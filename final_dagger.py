# Test DAgger algorithm on arbitrary environment
# Generalized full implementation
# Author: John Hallman

import numpy as np
import sys
import os
import time
import gym
import math
import json
import argparse
import joblib

# import ML packages
import tensorflow as tf 
from keras import Model
from keras.models import Sequential, load_model
from keras.layers import Dense, Input
from keras.optimizers import Adam, Adagrad, Adadelta
from spinup.utils.logx import restore_tf_graph


# just cause why not
class Expert:
	def __init__(self, policy):
		self.policy = policy

	def predict(self, x):
		return self.policy(x)


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
            a = policy.predict(o)
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
	def train(cls, state_dim, action_dim, training_data, layers=5, hidden_dim=32, epochs=10):
		model = Sequential()
		model.add(Dense(hidden_dim, input_dim=state_dim, activation='relu'))
		for i in range(layers - 2):
			model.add(Dense(hidden_dim, activation='relu'))
		model.add(Dense(action_dim, activation='linear'))
		opt = Adam()
		model.compile(loss='mean_squared_error', optimizer=opt, metrics=['mse'])
		model.fit(np.array(training_data[0]), np.array(training_data[1]), epochs=epochs, batch_size=32)

		return cls(model)

	def change_model(self, new_model):
		self.model = new_model

	def predict(self, x):
		return self.model.predict(np.expand_dims(x, axis=0))[0]

	def print_params(self, print_weights=False):
		print("Params:")
		print(self.model.summary())
		if print_weights:
			for weight in self.model.get_weights(): print(weight)


# simulate an expert for T rounds, return trajectories
def simulate_trajectories(index, data, env, D, T, pi_n, pi_expert, beta, time_limit=1000):

	if index > 0:
		prev = data[index - 1]
		data[index] = {'exp_s':prev['exp_s'], 'exp_t':prev['exp_t'], 'pol_s':prev['pol_s'], 'pol_t':prev['pol_t']}
	else:
		data[index] = {'exp_s':0, 'exp_t':0, 'pol_s':0, 'pol_t':0}

	for traject_round in range(T):
		observation = env.reset()
		random_outcome = np.random.binomial(1, beta)
		policy = pi_expert if (random_outcome == 1) else pi_n
		samples = 0
		for t in range(time_limit):
			action = policy.predict(observation)
			D[0].append(observation)
			D[1].append(pi_expert.predict(observation))
			observation, reward, done, info = env.step(action)
			samples += 1
			if done:
				break
		
		data[index]['exp_t'] += 1
		data[index]['exp_s'] += samples
		if not random_outcome:
			data[index]['pol_t'] += 1
			data[index]['pol_s'] += samples

	return


# run validation on set of policies in order to select the best one (test each policy k times)
def policy_validation(env, policies, k_validate):
	print("Running validation")
	n = len(policies)
	r_mean = []
	for i in range(n):
		r_policy = 0.0
		for k in range(k_validate):
			r_round = 0.0
			obs = env.reset()
			for t in range(1000):
				action = policies[i].predict(obs)
				obs, reward, done, info = env.step(action)
				r_round += reward
				if done or t+1 == 1000:
					r_policy += r_round / (t+1)
					break
			r_policy += r_round / k_validate
		r_mean.append(r_policy)

	i_opt = np.argmax(r_mean)
	return i_opt, r_mean[i_opt]


# T is the length of a single round/trajectory, N is the number of times to run the algorithm
# p is the probability that we use to decrease beta with at each round
# pi_expert the policy of the expert (np array of length 4)
# pi_start the first policy (initialized randomly if None)
def DAgger(env, T, N, p, state_dim, action_dim, data_n, WORKING_DIR, EXPERT_DIR, layers=5, hidden_dim=32, epochs=10):

	# D is set of trajectories, beta is mixing probability, checkpoint for how far through training we are
	D, pi_trained = [[], []], []
	beta, checkpoint = 1.0, 10

	data = {}

	# step 1: collect trajectories from pi_expert, (no need to select initial policy)
	with tf.Session() as sess:
		expert_policy = load_policy(sess, EXPERT_DIR)
		pi_expert = Expert(expert_policy)
		simulate_trajectories(0, data, env, D, T, None, pi_expert, beta) # note beta = 1.0! (hence no need for index)
		pi_new = Policy.train(state_dim, action_dim, layers=layers, hidden_dim=hidden_dim, training_data=D, epochs=epochs)
		pi_trained.append(pi_new)
		data[0]['rewards'] = evaluate_reward(env, pi_new, data_n)

		print("Starting Training for {} rounds!".format(N))
		for n in range(N):
			beta = beta * p
			simulate_trajectories(n + 1, data, env, D, T, pi_new, pi_expert, beta)
			pi_new = Policy.train(state_dim, action_dim, layers=layers, hidden_dim=hidden_dim, training_data=D, epochs=epochs)
			pi_trained.append(pi_new)
			data[n + 1]['rewards'] = evaluate_reward(env, pi_new, data_n)

			if (100 * (n + 1) / N >= checkpoint):
				while checkpoint < 100 * (n + 1) / N: checkpoint += 10
				print("{}% towards completion".format(checkpoint))
		i_opt, r_opt = policy_validation(env, pi_trained, k_validate)
		pi_opt = pi_trained[i_opt]
		pi_opt.model.save(os.path.join(WORKING_DIR, "model.h5"))

	return data # return data on experiments


# visualize and render policy
def visualize(args, env, pi_opt, pi_expert):
	obs = env.reset()
	for t in range(args.steps):
		env.render()
		time.sleep(1e-3) # slows down process to make it more visible
		action = pi_opt.predict(obs)
		obs, reward, done, info = env.step(action)
		if done:
			print("Model ran {} time steps".format(t+1))
			break


# train policy using DAgger
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

	def run_policy(env, policy):
		o = env.reset()
		for t in range(1000):
			env.render()
			time.sleep(1e-3)
			a = policy.predict(o)
			o, _, d, _ = env.step(a)

	# parses arguments
	parser = argparse.ArgumentParser()

	# main arguments
	parser.add_argument('--env', type=str, default='LunarLanderContinuous-v2')
	parser.add_argument('--load_model', action='store_true') # if enabled, only runs pretrained model without training
	parser.add_argument('--dir_name', type=str, default='') # location to store trained models in
	parser.add_argument('--expert_dir_name', type=str, default='')
	parser.add_argument('--rounds', type=int, default=4) # number of rounds of Frank-Wolfe algorithm -- 5
	parser.add_argument('--validate_rounds', type=int, default=50) # number of rounds to run validation -- 50
	parser.add_argument('--data_n', type=int, default=10) # number of rounds to run validation

	# density estimator arguments
	parser.add_argument('--trajects', type=int, default=10) # number of regular trajectories -- 50
	parser.add_argument('--iter_length', type=int, default=3000)
	parser.add_argument('--policy_prob', type=float, default=0.8)

	# policy trainer arguments
	parser.add_argument('--epochs', type=int, default=10) # -- 10
	parser.add_argument('--hid', type=int, default=32) # dimension of hidden layers in Actor-Critic neural networks
	parser.add_argument('--l', type=int, default=5)
	parser.add_argument('--steps', type=int, default=3000)
	args = parser.parse_args()

	print("\n---- Running DAgger ----\n")

	BASE_DIR = os.path.join(os.getcwd(), "models")
	WORKING_DIR = os.path.join(BASE_DIR, args.dir_name) 
	EXPERT_DIR = os.path.join(BASE_DIR, args.expert_dir_name)

	if not os.path.isdir(WORKING_DIR):
		os.makedirs(WORKING_DIR)

	# parameters
	T, N, prob, k_validate = args.trajects, args.rounds, args.policy_prob, args.validate_rounds # 1, 1, 0.8, 10
	model_layers, hidden_dim, epochs = args.l, args.hid, args.epochs # 3, 5, 20

	# initialize environment
	env = gym.make(args.env)
	state_dim, action_dim = env.observation_space.shape[0], env.action_space.shape[0]


	# run pretrained model vs train new model
	if args.load_model:
		assert os.path.isdir(WORKING_DIR)
		print("\nRunning model in directory " + WORKING_DIR)
		run_policy_from_dir(WORKING_DIR, args.env)
		sys.exit(0)

	else:
		# train policy
		t_alg = time.time()
		data = DAgger(env, T, N, prob, state_dim, action_dim, args.data_n, WORKING_DIR, EXPERT_DIR, layers=model_layers, 
			hidden_dim=hidden_dim, epochs=epochs)
		t_alg = time.time() - t_alg

		print("\nTraining complete! Visualizing...")
		pi_opt = Policy(load_model(os.path.join(WORKING_DIR, "model.h5")))
		run_policy(env, pi_opt)

		filename = os.path.join("experiment_data", args.dir_name + ".json") # os.path.join(WORKING_DIR, "reward_data.json")
		with open(filename, 'w') as file:
			json.dump(data, file)

		t_total = time.time() - t_total

		print("\nStored data, program complete!")
		print("Final runtimes, algorithm and entire program: {}, {}\n".format(t_alg, t_total))

    

	env.close()


