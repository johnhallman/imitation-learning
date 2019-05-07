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

import tensorflow as tf 
from scipy import stats # kernel density estimation in high dimensions
import spinup.algos.ppo.core as core
from spinup.utils.run_utils import setup_logger_kwargs
from spinup.utils.logx import EpochLogger, restore_tf_graph # restore_tf_graph necessary to load models from directories
from spinup.utils.mpi_tf import MpiAdamOptimizer, sync_all_params
from spinup.utils.mpi_tools import mpi_fork, mpi_avg, proc_id, mpi_statistics_scalar, num_procs
from spinup_ppo_copy import PPOBuffer # no need to copy code if there is no change
from keras import Model
from keras.models import Sequential, load_model
from keras.layers import Dense, Input
from keras.optimizers import Adam, Adagrad, Adadelta
from spinup.utils.logx import restore_tf_graph


# load tf model from filepath
def load_policy(sess, fpath):
    model = restore_tf_graph(sess, os.path.join(fpath, 'simple_save'))
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


class Policy(Model):
    def __init__(self, model):
        super(Policy, self).__init__(name='policy')
        self.model = model

    @classmethod # note: minimum number of layers is 2!
    def initialize(cls, state_dim, action_dim, hidden_dim, layers):
        model = Sequential()
        model.add(Dense(hidden_dim, input_dim=state_dim, activation='relu'))
        for i in range(layers - 2):
            model.add(Dense(hidden_dim, activation='relu'))
        model.add(Dense(action_dim, activation='linear'))
        opt = Adam()
        model.compile(loss='mean_squared_error', optimizer=opt, metrics=['mse'])
        return cls(model)

    def train(self, training_data, epochs):
        self.model.fit(np.array(training_data[0]), np.array(training_data[1]), epochs=epochs, batch_size=32)

    def change_model(self, new_model):
        self.model = new_model

    def predict(self, x):
        return self.model.predict(np.expand_dims(x, axis=0))[0]

    def print_params(self, print_weights=False):
        print("Params:")
        print(self.model.summary())
        if print_weights:
            for weight in self.model.get_weights(): print(weight)


# evaluate a given policy on env over n rounds
def evaluate_reward(env, policy, n):
    cum, r_data = 0.0, []
    for i in range(n):
        o = env.reset()
        round_data, r_round = [], 0.0
        for t in range(args.steps):
            a = policy.predict(o)
            o, r, d, _ = env.step(a)
            round_data.append(r)
            r_round += r
            if d or t+1 == args.steps:
                cum += r_round / (t+1)
                break
        r_data.append(round_data)
    return r_data, cum / n



# simulate an expert for T rounds, return trajectories
def simulate_trajectories(rounds, data, env, T, policy, time_limit=3000):
    exp_t, exp_s = 0, 0
    obs_data = [[], []]
    for traject_round in range(T):
        observation = env.reset()
        samples = 0
        for t in range(time_limit):
            action = policy(observation)

            obs_data[0].append(observation)
            obs_data[1].append(action)
            observation, reward, done, info = env.step(action)
            samples += 1
            if done:
                break
        exp_t += 1
        exp_s += samples

    for i in range(rounds):
        data[i] = {'exp_t':exp_t, 'exp_s':exp_s}

    return obs_data


# visualize and render policy
def visualize(args, env, pi_opt):
    for i in range(1):
        o = env.reset()
        for t in range(1000):
            env.render()
            time.sleep(1e-3)
            a = pi_opt.predict(o)
            o, _, d, _ = env.step(a)


# run Hazan distribution imitation learning algorithm
if __name__ == "__main__":

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
    parser.add_argument('--dir_name', type=str, default='') # location to store trained models in
    parser.add_argument('--expert_dir_name', type=str, default='')

    parser.add_argument('--rounds', type=int, default=5) # number of rounds to sample/train --- 5
    parser.add_argument('--expert_trajects', type=int, default=10) # number of expert trajectories
    parser.add_argument('--epochs', type=int, default=10) # ---- 10

    parser.add_argument('--steps', type=int, default=3000)
    parser.add_argument('--k_validation', type=int, default=20)
    parser.add_argument('--data_n', type=int, default=10) # number of rounds to run validation
    parser.add_argument('--hid', type=int, default=32) # dimension of hidden layers in Actor-Critic neural networks
    parser.add_argument('--l', type=int, default=5)
    args = parser.parse_args()

    BASE_DIR = os.path.join(os.getcwd(), "models")
    WORKING_DIR = os.path.join(BASE_DIR, args.dir_name) 
    EXPERT_DIR = os.path.join(BASE_DIR, args.expert_dir_name)


    if os.path.isdir(WORKING_DIR):
        raise Exception("Data directory alread exists! (--dir_name)")
    if args.expert_dir_name == None or not os.path.isdir(EXPERT_DIR):
        raise Exception("Invalid expert directory! (--expert_dir_name)")


    data = {} # BIG DATA

    env = gym.make(args.env)
    state_dim, action_dim = env.observation_space.shape[0], env.action_space.shape[0]

    with tf.Session() as sess:
        expert = load_policy(sess, EXPERT_DIR)
        expert_trajects = simulate_trajectories(args.rounds, data, env, args.expert_trajects, expert)
        policy = Policy.initialize(state_dim, action_dim, args.hid, args.l)
        for i in range(args.rounds):
            policy.train(expert_trajects, args.epochs)
            data[i]['rewards'] = evaluate_reward(env, policy, args.data_n)
    
        print("\nDone training! Now visualize...")
        visualize(args, env, policy)
        policy.model.save(WORKING_DIR + ".h5")

    # store reward data in json file
    filename = os.path.join("experiment_data", args.dir_name + ".json" )# os.path.join(WORKING_DIR, "reward_data.json")
    with open(filename, 'w') as file:
        json.dump(data, file)

    env.close()
    print("\n\nProgram complete!!!")




