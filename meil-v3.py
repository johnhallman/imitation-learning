# MEIL - Maximum Entropy Imitation Learning (Proximal Policy Optimization based implementation)
# Author: John Hallman

import numpy as np
import tensorflow as tf
import json
import gym
import argparse
import time
import joblib
import sys
import os
import os.path as osp
from scipy import stats # kernel density estimation in high dimensions
import spinup.algos.ppo.core as core
from spinup.utils.run_utils import setup_logger_kwargs
from spinup.utils.logx import EpochLogger, restore_tf_graph # restore_tf_graph necessary to load models from directories
from spinup.utils.mpi_tf import MpiAdamOptimizer, sync_all_params
from spinup.utils.mpi_tools import mpi_fork, mpi_avg, proc_id, mpi_statistics_scalar, num_procs
from spinup_ppo_copy import PPOBuffer # no need to copy code if there is no change



OPTIMIZATION_METHODS = [
    "sac", # Soft Actor-Critic
    "ppo", # Proximal Policy Optimization
    "td3", # TD3
    "ddpg", # Deep Deterministic Policy Gradients
    "trpo" # Trust Region Policy Optimization
]

DENSITY_ESTIMATORS = [
    "gaussian_kde", # Kernel Density Estimator with Gaussian Kernel
    "discrete_pca" # Randomly project onto low-dimensional subspace, then discretize
]



# load tf model from filepath
def load_policy(sess, fpath):
    model = restore_tf_graph(sess, osp.join(fpath, 'simple_save'))
    get_action = lambda x : sess.run(model['pi'], feed_dict={model['x']: x[None,:]})[0]
    return get_action



# evaluate a given policy on env over n rounds
def evaluate_reward(env, policy, n):
    r_cum = []
    for i in range(n):
        o = env.reset()
        s = 0.0
        for t in range(args.steps):
            a = policy(o)
            o, r, d, _ = env.step(a)
            s += r
            if d or t+1 == args.steps:
                r_cum.append(s / (t+1))
                break
    return r_cum


# evaluate a given policy on env over n rounds
def evaluate_distribution(env, n, expert, reward):
    r_cum = []
    for i in range(n):
        o = env.reset()
        s = 0.0
        for t in range(args.steps):
            a = expert(o)
            o, _, d, _ = env.step(a)
            s += np.log(reward(o))
            if d or t+1 == args.steps:
                r_cum.append(s / (t+1))
                break
    return r_cum



# returns policy with highest reward
def reward_validation(WORKING_DIR, args):
    scores = []
    policy_dirs = os.listdir(WORKING_DIR)
    with tf.Session() as sess:
        env = gym.make(args.env)
        for sub_dir in policy_dirs:
            policy = load_policy(sess, os.path.join(WORKING_DIR, sub_dir))
            score_array = evaluate_reward(env, policy, args.validate_rounds)
            scores.append(np.mean(score_array))
        env.close()
    opt_ind = np.argmax(scores)
    print("Reward validation results, optimal index: {}".format(opt_ind))
    print(scores)
    return os.path.join(WORKING_DIR, policy_dirs[opt_ind])



# class for a Gaussian KDE distribution
class Gaussian_Density:
    def __init__(self):
        self.trajects = None
        self.weights = None
        self.kernel = None
        self.num_samples = 0
        self.num_trajects = 0

    def train(self, env, policy, N, gamma, iter_length, weighted=True):
        trajects, weights = [], []
        eps = lambda g, k: 1.0 if g == 1.0 else max(g ** k / (1 - g), 1e-6)
        for i in range(N):
            obs = env.reset()
            for t in range(iter_length):
                trajects.append(obs)
                action = policy(obs)
                obs, reward, done, info = env.step(action)
                self.num_samples += 1
                if done:
                    weights += [eps(gamma, k) for k in range(t+1)]
                    break
            self.num_trajects += 1
        self.trajects = trajects
        self.weights = weights
        self.kernel = stats.gaussian_kde(np.array(trajects).T, weights=weights) if weighted \
            else stats.gaussian_kde(np.array(trajects).T)

    def merge(self, distributions, alphas, weighted=True): # merges distributions in p according to weights
        all_t, all_w = [], []
        for d, a in zip(distributions, alphas):
            all_t += d.trajects
            all_w += a * d.weights
        self.trajects = all_t
        self.weights = all_w
        self.kernel = stats.gaussian_kde(np.array(all_t).T, weights=all_w) if weighted \
            else stats.gaussian_kde(np.array(all_t).T)

    def kernel(self, s):
        assert self.kernel != None
        return self.kernel.pdf(s)[0]

    def density(self):
        assert self.kernel != None
        return lambda s: self.kernel.pdf(s)[0]



# runs customized, removed: save_freq, seed
def ppo(BASE_DIR, expert_density, env_fn, actor_critic=core.mlp_actor_critic, ac_kwargs=dict(), 
        steps_per_epoch=1000, epochs=10, gamma=0.99, clip_ratio=0.2, pi_lr=3e-4,
        vf_lr=1e-3, train_pi_iters=50, train_v_iters=50, lam=0.97, max_ep_len=1000,
        target_kl=0.01, data_n=10):

    data = {} # ALL THE DATA

    logger_kwargs = setup_logger_kwargs(args.dir_name, data_dir=BASE_DIR)
    logger = EpochLogger(**logger_kwargs)
    logger.save_config(locals())

    env = env_fn()
    obs_dim = env.observation_space.shape
    act_dim = env.action_space.shape
    
    # Share information about action space with policy architecture
    ac_kwargs['action_space'] = env.action_space

    # Inputs to computation graph
    x_ph, a_ph = core.placeholders_from_spaces(env.observation_space, env.action_space)
    adv_ph, ret_ph, logp_old_ph = core.placeholders(None, None, None)

    # Main outputs from computation graph
    pi, logp, logp_pi, v = actor_critic(x_ph, a_ph, **ac_kwargs)

    # Need all placeholders in *this* order later (to zip with data from buffer)
    all_phs = [x_ph, a_ph, adv_ph, ret_ph, logp_old_ph]

    # Every step, get: action, value, and logprob
    get_action_ops = [pi, v, logp_pi]

    # Experience buffer
    local_steps_per_epoch = int(steps_per_epoch / num_procs())
    buf = PPOBuffer(obs_dim, act_dim, local_steps_per_epoch, gamma, lam)

    # Count variables
    var_counts = tuple(core.count_vars(scope) for scope in ['pi', 'v'])
    logger.log('\nNumber of parameters: \t pi: %d, \t v: %d\n'%var_counts)

    # PPO objectives
    ratio = tf.exp(logp - logp_old_ph)          # pi(a|s) / pi_old(a|s)
    min_adv = tf.where(adv_ph>0, (1+clip_ratio)*adv_ph, (1-clip_ratio)*adv_ph)
    pi_loss = -tf.reduce_mean(tf.minimum(ratio * adv_ph, min_adv))
    v_loss = tf.reduce_mean((ret_ph - v)**2)

    # Info (useful to watch during learning)
    approx_kl = tf.reduce_mean(logp_old_ph - logp)      # a sample estimate for KL-divergence, easy to compute
    approx_ent = tf.reduce_mean(-logp)                  # a sample estimate for entropy, also easy to compute
    clipped = tf.logical_or(ratio > (1+clip_ratio), ratio < (1-clip_ratio))
    clipfrac = tf.reduce_mean(tf.cast(clipped, tf.float32))

    # Optimizers
    train_pi = MpiAdamOptimizer(learning_rate=pi_lr).minimize(pi_loss)
    train_v = MpiAdamOptimizer(learning_rate=vf_lr).minimize(v_loss)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    # Sync params across processes
    sess.run(sync_all_params())

    # Setup model saving
    logger.setup_tf_saver(sess, inputs={'x': x_ph}, outputs={'pi': pi, 'v': v})

    # update rule
    def update():
        inputs = {k:v for k,v in zip(all_phs, buf.get())}
        pi_l_old, v_l_old, ent = sess.run([pi_loss, v_loss, approx_ent], feed_dict=inputs)

        # Training
        for i in range(train_pi_iters):
            _, kl = sess.run([train_pi, approx_kl], feed_dict=inputs)
            kl = mpi_avg(kl)
            if kl > 1.5 * target_kl:
                logger.log('Early stopping at step %d due to reaching max kl.'%i)
                break
        logger.store(StopIter=i)
        for _ in range(train_v_iters):
            sess.run(train_v, feed_dict=inputs)

        # Log changes from update
        pi_l_new, v_l_new, kl, cf = sess.run([pi_loss, v_loss, approx_kl, clipfrac], feed_dict=inputs)
        logger.store(LossPi=pi_l_old, LossV=v_l_old, 
                     KL=kl, Entropy=ent, ClipFrac=cf,
                     DeltaLossPi=(pi_l_new - pi_l_old),
                     DeltaLossV=(v_l_new - v_l_old))

    policy_distr = Gaussian_Density()
    policy = lambda s: np.random.uniform(-2.0, 2.0, size=env.action_space.shape) # random policy
    policy_distr.train(env, policy, args.trajects, args.distr_gamma, args.iter_length)
    density = policy_distr.density()

    data[0] = {'pol_s': policy_distr.num_samples, 'pol_t': policy_distr.num_trajects}

    dist_rewards = []

    # repeat REIL for given number of rounds
    for i in range(args.rounds):

        message = "\nRound {} out of {}\n".format(i + 1, args.rounds)
        reward = lambda s: expert_density(s) / (density(s) + args.eps)

        dist_rewards.append(reward)

        start_time = time.time()
        o, old_r, d, ep_ret, ep_len = env.reset(), 0, False, 0, 0
        r = reward(o) # custom reward

        # Main loop: collect experience in env and update/log each epoch
        for epoch in range(epochs):
            for t in range(local_steps_per_epoch):

                a, v_t, logp_t = sess.run(get_action_ops, feed_dict={x_ph: o.reshape(1,-1)})

                # save and log
                buf.store(o, a, r, v_t, logp_t)
                logger.store(VVals=v_t)

                o, old_r, d, _ = env.step(a[0])
                r = reward(o)
                ep_ret += r
                ep_len += 1

                terminal = d or (ep_len == max_ep_len)
                if terminal or (t==local_steps_per_epoch-1):
                    if not(terminal):
                        print('Warning: trajectory cut off by epoch at %d steps.'%ep_len)
                    # if trajectory didn't reach terminal state, bootstrap value target
                    last_val = old_r if d else sess.run(v, feed_dict={x_ph: o.reshape(1,-1)})
                    last_val = reward(o)
                    buf.finish_path(last_val)
                    if terminal:
                        # only save EpRet / EpLen if trajectory finished
                        logger.store(EpRet=ep_ret, EpLen=ep_len)
                    o, old_r, d, ep_ret, ep_len = env.reset(), 0, False, 0, 0
                    r = reward(o)

            # store model!
            if (epoch == epochs-1): logger.save_state({'env': env}, None)

            # Perform PPO update!
            update()

            # Log info about epoch
            logger.log_tabular('Epoch', epoch)
            logger.log_tabular('EpRet', with_min_and_max=True)
            logger.log_tabular('EpLen', average_only=True)
            logger.log_tabular('VVals', with_min_and_max=True)
            logger.log_tabular('TotalEnvInteracts', (epoch+1)*steps_per_epoch)
            logger.log_tabular('LossPi', average_only=True)
            logger.log_tabular('LossV', average_only=True)
            logger.log_tabular('DeltaLossPi', average_only=True)
            logger.log_tabular('DeltaLossV', average_only=True)
            logger.log_tabular('Entropy', average_only=True)
            logger.log_tabular('KL', average_only=True)
            logger.log_tabular('ClipFrac', average_only=True)
            logger.log_tabular('StopIter', average_only=True)
            logger.log_tabular('Time', time.time()-start_time)
            logger.dump_tabular()
            print(message)

        policy = lambda state: sess.run(get_action_ops, feed_dict={x_ph: state.reshape(1,-1)})[0][0]
        data[i] = {'pol_s': policy_distr.num_samples, 'pol_t': policy_distr.num_trajects}
        data[i]['rewards'] = evaluate_reward(env, policy, data_n)

        if i != args.rounds - 1:
            policy_distr.train(env, policy, args.trajects, args.distr_gamma, args.iter_length)
            density = policy_distr.density()

    return data, dist_rewards



# takes optimization method, density estimator, other arguments, and runs Maximum
# Entropy Imitation Learning algorithm, then returns best directory after validation
def meil(BASE_DIR, WORKING_DIR, EXPERT_DIR, args):

    env = gym.make(args.env)
    with tf.Session() as sess:
        expert_distribution = Gaussian_Density()
        expert = load_policy(sess, EXPERT_DIR)
        expert_distribution.train(env, expert, args.expert_trajects, args.distr_gamma, args.iter_length)
        expert_density = expert_distribution.density()
    env.close()

    data, dist_rewards = ppo(BASE_DIR, expert_density, lambda : gym.make(args.env), 
        actor_critic=core.mlp_actor_critic, ac_kwargs=dict(hidden_sizes=[args.hid]*args.l), 
        gamma=args.gamma, steps_per_epoch=args.steps, epochs=args.epochs, data_n=args.data_n)

    for k, sub_dict in data.items():
        sub_dict['exp_s'] = expert_distribution.num_samples
        sub_dict['exp_t'] = expert_distribution.num_trajects

    print("\nFinished training! Computing RE loss...\n")

    env = gym.make(args.env)
    tf.reset_default_graph()
    with tf.Session() as sess:
        expert = load_policy(sess, EXPERT_DIR)
        for i in range(args.rounds):
            data[i]['RE-loss'] = evaluate_distribution(env, args.data_n, expert, dist_rewards[i])
    env.close()

    return data



# run Hazan distribution imitation learning algorithm
if __name__ == "__main__":

    t_total = time.time()

    # load and visualize policy
    def run_policy_from_dir(policy_dir, env_name, n=1):
        if policy_dir==None or not os.path.isdir(policy_dir):
            raise Exception("Given policy directory doesn't exist!")
        with tf.Session() as sess:
            env = gym.make(env_name)
            policy = load_policy(sess, policy_dir)
            for i in range(n):
                o = env.reset()
                for t in range(1000):
                    env.render()
                    time.sleep(1e-3)
                    a = policy(o)
                    o, _, d, _ = env.step(a)
            env.close()


    # parses arguments
    parser = argparse.ArgumentParser()

    # main arguments
    parser.add_argument('--env', type=str, default='LunarLanderContinuous-v2')
    parser.add_argument('--load_model', action='store_true') # if enabled, only runs pretrained model without training
    parser.add_argument('--dir_name', type=str, default='') # location to store trained models in
    parser.add_argument('--expert_dir_name', type=str, default='')
    parser.add_argument('--rounds', type=int, default=5) # number of rounds of Frank-Wolfe algorithm -- 5
    parser.add_argument('--validate_rounds', type=int, default=50) # number of rounds to run validation
    parser.add_argument('--data_n', type=int, default=10) # number of rounds to run validation

    # density estimator arguments
    parser.add_argument('--distr_gamma', type=float, default=1.0)
    parser.add_argument('--expert_trajects', type=int, default=1) # number of expert trajectories
    parser.add_argument('--trajects', type=int, default=25) # number of regular trajectories
    parser.add_argument('--iter_length', type=int, default=3000)

    # policy trainer arguments
    parser.add_argument('--epochs', type=int, default=50) # regular epochs -- 50
    parser.add_argument('--hid', type=int, default=32) # dimension of hidden layers in Actor-Critic neural networks
    parser.add_argument('--l', type=int, default=5)
    parser.add_argument('--gamma', type=float, default=0.99) # training PPO gamma
    parser.add_argument('--steps', type=int, default=3000)
    parser.add_argument('--eps', type=float, default=1e-5) # epsilon added to reward d*(s)/(d_t(s) + eps)
    args = parser.parse_args()

    BASE_DIR = os.path.join(os.getcwd(), "models")
    WORKING_DIR = os.path.join(BASE_DIR, args.dir_name) 
    EXPERT_DIR = os.path.join(BASE_DIR, args.expert_dir_name)

    # visualize pretrained model and exit
    if args.load_model:
        assert os.path.isdir(WORKING_DIR)
        print("\nRunning model in directory " + WORKING_DIR)
        run_policy_from_dir(WORKING_DIR, args.env)
        sys.exit(0)

    print("\nMEIL, store results in directory " + WORKING_DIR)
    if os.path.isdir(WORKING_DIR):
        raise Exception("Data directory alread exists! (--dir_name)")
    if args.expert_dir_name == None or not os.path.isdir(EXPERT_DIR):
        raise Exception("Invalid expert directory! (--expert_dir_name)")

    # train policy and store model
    print("\n----- Run MEIL algorithm -----\n")

    t_alg = time.time()
    data = meil(BASE_DIR, WORKING_DIR, EXPERT_DIR, args)
    t_alg = time.time() - t_alg

    # store reward data in json file
    filename = os.path.join("experiment_data", args.dir_name + ".json" )# os.path.join(WORKING_DIR, "reward_data.json")
    with open(filename, 'w') as file:
        json.dump(data, file)

    print("\n\nTraining complete!!!")
    print("Visualizing...\n")

    # visualize policy
    run_policy_from_dir(WORKING_DIR, args.env)

    t_total = time.time() - t_total
    print("Final runtimes, algorithm and entire program: {}, {}\n".format(t_alg, t_total))




