# Test DAgger algorithm on arbitrary environment
# Generalized full implementation
# Author: John Hallman

import numpy as np
import sys
import os
import time
import gym
import math

# import ML packages
import tensorflow as tf 
from keras import Model
from keras.models import Sequential, load_model
from keras.layers import Dense, Input, Activation
from keras.optimizers import Adam, Adagrad, Adadelta


class Policy(Model):
	def __init__(self, model):
		super(Policy, self).__init__(name='policy')
		self.model = model

	@classmethod
	def train(cls, state_dim, action_dim, training_data, layers=10, hidden_dim=None, epochs=10):
		model = Sequential()
		if layers > 1:
			if not hidden_dim: hidden_dim = max(state_dim, action_dim)
			model.add(Dense(hidden_dim, input_dim=state_dim, kernel_initializer='normal', activation='relu'))
			for i in range(layers - 2):
				model.add(Dense(hidden_dim, kernel_initializer='normal', activation='relu'))
			model.add(Dense(action_dim, kernel_initializer='normal', activation='linear'))
		else:
			model.add(Dense(action_dim, input_dim=state_dim, kernel_initializer='normal', activation='sigmoid'))
		opt = Adam()
		model.compile(loss='mean_squared_error', optimizer=opt, metrics=['mse'])
		model.fit(np.array(training_data[0]), np.array(training_data[1]), epochs=epochs, batch_size=32)

		return cls(model)

	def change_model(self, new_model):
		self.model = new_model

	def predict(self, input):
		return self.model.predict(np.expand_dims(input, axis=0))[0]

	def print_params(self, print_weights=False):
		print("Params:")
		print(self.model.summary())
		if print_weights:
			for weight in self.model.get_weights(): print(weight)


# expert policy
class Expert(Policy):
	def __init__(self, filename, state_dim, action_dim):
		model = Sequential()
		model.add(Dense(16, input_dim=state_dim))
		model.add(Activation('relu'))
		model.add(Dense(16))
		model.add(Activation('relu'))
		model.add(Dense(16))
		model.add(Activation('relu'))
		model.add(Dense(action_dim))
		model.add(Activation('linear'))
		model.load_weights(filename)
		Policy.__init__(self, model)



# simulate an expert for T rounds, return trajectories
def simulate_trajectories(D, T, pi_n, pi_expert, beta, time_limit=1000):
	for traject_round in range(T):
		observation = env.reset()
		policy = pi_expert if (np.random.binomial(1, beta) == 1) else pi_n
		for t in range(time_limit):
			action = policy.predict(observation)
			D[0].append(observation)
			D[1].append(pi_expert.predict(observation))
			observation, reward, done, info = env.step(action)
			if done:
				break
	return


# run validation on set of policies in order to select the best one (test each policy k times)
def policy_validation(policies, k_validate, expert_policy):
	print("Running validation")
	n = len(policies)
	pred_mean = []
	for i in range(n):
		pred_current = []
		for k in range(k_validate):
			data = []
			obs = env.reset()
			for t in range(1000):
				data.append(obs)
				action = policies[i].predict(obs)
				obs, reward, done, info = env.step(action)
				if done:
					break
			policy_preds = np.array([policies[i].predict(obs) for obs in data])
			expert_preds = np.array([expert_policy.predict(obs) for obs in data])
			pred_current.append(((policy_preds - expert_preds)**2).mean())
		pred_mean.append(np.mean(np.array(pred_current)))

	i_opt = np.argmin(pred_mean)
	return i_opt, pred_mean[i_opt]


# T is the length of a single round/trajectory, N is the number of times to run the algorithm
# p is the probability that we use to decrease beta with at each round
# pi_expert the policy of the expert (np array of length 4)
# pi_start the first policy (initialized randomly if None)
def DAgger(T, N, p, state_dim, action_dim, pi_expert, layers=1, hidden_dim=None, epochs=10):

	# D is set of trajectories, beta is mixing probability, checkpoint for how far through training we are
	D, pi_trained = [[], []], []
	beta, checkpoint = 1.0, 10

	# step 1: collect trajectories from pi_expert, (no need to select initial policy)
	simulate_trajectories(D, T, None, pi_expert, beta) # note beta = 1.0!
	pi_new = Policy.train(state_dim, action_dim, D, layers=layers, hidden_dim=hidden_dim, epochs=epochs)
	pi_trained.append(pi_new)

	print("Starting Training for {} rounds and {} trajectories per round!".format(N, T))
	for n in range(N):
		beta = beta * p
		simulate_trajectories(D, T, pi_trained[-1], pi_expert, beta)
		pi_new = Policy.train(state_dim, action_dim, D, layers=layers, hidden_dim=hidden_dim, epochs=epochs)
		pi_trained.append(pi_new)
		if (100 * (n + 1) / N >= checkpoint):
			print("{}% towards completion".format(checkpoint))
			#print(pi_new.print_params())
			while checkpoint <= 100 * (n + 1) / N: checkpoint += 10

	return pi_trained # pick best policy


# visualize and render policy
def visualize(env, pi_opt, pi_expert):
	data = []
	obs = env.reset()
	for t in range(1000):
		data.append(obs)
		env.render()
		time.sleep(0.03) # slows down process to make it more visible
		action = pi_opt.predict(obs)
		obs, reward, done, info = env.step(action)
		if done:
			print("Model ran {} time steps".format(t+1))
			break

	preds = np.array([pi_opt.predict(obs)[0] for obs in data])
	correct = np.array([pi_expert.predict(obs)[0] for obs in data])
	print("MSE: {}".format(((preds - correct)**2).mean()))


# train policy using DAgger
if __name__ == "__main__":

	print("Flatten:")

	# $1 = environment, $2 = expert model, $3 policy model (optional)
	assert(len(sys.argv) == 3 or len(sys.argv) == 4)
	expert_path = "models/" + sys.argv[2]
	if len(sys.argv) == 4:
		policy_path = "models/" + sys.argv[3]
	env_name = sys.argv[1]

	# parameters
	T, N, prob, k_validate = 10, 10, 0.8, 10
	model_layers, hidden_dim, epochs = 4, 5, 20

	# initialize environment
	env = gym.make(env_name)
	state_dim, action_dim = env.observation_space.shape[0], env.action_space.shape[0]
	pi_expert = Expert(expert_path, state_dim, action_dim)

	# run pretrained model vs train new model
	if len(sys.argv) > 3 and os.path.isfile(policy_path):
		print("Loading pre-trained model from " + policy_path)
		opt_model = load_model(policy_path)
		pi_opt = Policy(opt_model)
		visualize(env, pi_opt, pi_expert)

	else:
		# train policy
		policies = DAgger(T, N, prob, state_dim, action_dim, pi_expert, layers=model_layers, hidden_dim=hidden_dim, epochs=epochs)
		i_opt, pred_opt = policy_validation(policies, k_validate, pi_expert)
		pi_opt = policies[i_opt]

		print("\n\nTraining complete!!!")
		print("Best policy was index {} out of {} with mse {}".format(i_opt, N, pred_opt))
		visualize(env, pi_opt, pi_expert)

		if len(sys.argv) == 4:
			print("Saving model to file " + policy_path)
			pi_opt.model.save(policy_path)
    

	env.close()


