# Test DAgger algorithm on cartpole environment

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
from keras.layers import Dense, Input
from keras.optimizers import Adam, Adagrad, Adadelta



class NN_Policy(Model):
	def __init__(self, state_dim, action_dim, layers=1, hidden_dim=None, training_data=None, epochs=10):
		super(NN_Policy, self).__init__(name='nn_policy')
		assert(layers > 0)
		
		self.model = Sequential()
		if layers > 1:
			if not hidden_dim: hidden_dim = max(state_dim, action_dim)
			self.model.add(Dense(hidden_dim, input_dim=state_dim, kernel_initializer='normal', activation='relu'))
			for i in range(layers - 2):
				self.model.add(Dense(hidden_dim, kernel_initializer='normal', activation='relu'))
			self.model.add(Dense(action_dim, kernel_initializer='normal', activation='sigmoid'))
		else:
			self.model.add(Dense(action_dim, input_dim=state_dim, kernel_initializer='normal', activation='sigmoid'))

		if training_data:
			opt = Adam()
			self.model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
			self.model.fit(np.array(training_data[0]), np.array(training_data[1]), epochs=epochs, batch_size=32)

	def change_model(self, new_model):
		self.model = new_model

	def predict(self, input):
		probs = self.model.predict(np.expand_dims(input, axis=0))[0]
		return np.argmax(probs)

	def print_params(self):
		print("Params:")
		print(self.model.summary())
		#for weight in self.model.get_weights():
		#	print(weight)



 # Cartpole expert model with v = np.array([0.1, 0.1, 0.5, 0.1])
class Expert_Policy(Model):
	def __init__(self, env_name):
		super(Expert_Policy, self).__init__(name='expert_policy')

		W, b = np.array([[0.1], [0.1], [0.5], [0.1]]), np.array([0.0])
		x = Input(shape=(4,))
		y = Dense(1, kernel_initializer='normal', activation='sigmoid', weights=[W,b])(x)
		self.model = Model(x, y)

	def predict(self, input):
		return np.rint(self.model.predict(np.expand_dims(input, axis=0))[0, 0]).astype(int)

	def print_params(self):
		print("Params:")
		print(self.model.summary())
		#for weight in self.model.get_weights():
		#	print(weight)

	def model():
		return self.model



# simulate an expert for T rounds, return trajectories
def simulate_trajectories(D, T, pi_n, pi_expert, beta, time_limit=1000):
	for traject_round in range(T):
		observation = env.reset()
		policy = pi_expert if (np.random.binomial(1, beta) == 1) else pi_n
		for t in range(time_limit):
			action = policy.predict(observation)
			D[0].append(observation)
			D[1].append(np.array([1 - pi_expert.predict(observation), pi_expert.predict(observation)]))
			observation, reward, done, info = env.step(action)
			if done:
				break
	return


# run validation on set of policies in order to select the best one (test each policy k times)
def policy_validation(policies, k_validate, expert_policy):
	print("Running validation")
	n = len(policies)
	t_mean, pred_mean = [], []
	for i in range(n):
		t_current, pred_current = [], []
		for k in range(k_validate):
			data = []
			obs = env.reset()
			for t in range(1000):
				data.append(obs)
				action = policies[i].predict(obs)
				obs, reward, done, info = env.step(action)
				if done:
					t_current.append(t)
					break
			preds = np.array([policies[i].predict(obs) for obs in data])
			correct = np.array([expert_policy.predict(obs) for obs in data])
			pred_current.append(np.mean(preds == correct))
		t_mean.append(np.mean(np.array(t_current)))
		pred_mean.append(np.mean(np.array(pred_current)))

	i_opt = np.argmax(np.where(np.argwhere(t_mean == np.amax(t_mean)), pred_mean, np.zeros(n)))
	return i_opt, t_mean[i_opt]+1, pred_mean[i_opt]


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
	pi_new = NN_Policy(state_dim, action_dim, layers=layers, hidden_dim=hidden_dim, training_data=D, epochs=epochs)
	pi_trained.append(pi_new)

	print("Starting Training for {} rounds and {} trajectories per round!".format(N, T))
	for n in range(N):
		beta = beta * p
		simulate_trajectories(D, T, pi_trained[-1], pi_expert, beta)
		pi_new = NN_Policy(state_dim, action_dim, layers=layers, hidden_dim=hidden_dim, training_data=D, epochs=epochs)
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

	preds = np.array([pi_opt.predict(obs) for obs in data])
	correct = np.array([pi_expert.predict(obs) for obs in data])
	print("Selected vs correct action accuracy: {}".format(np.mean(preds == correct)))


# train policy using DAgger and expert policy ([0.1, 0.1, 0.5, 0.1])
if __name__ == "__main__":
	assert(len(sys.argv) >= 2)
	if len(sys.argv) > 2: 
		assert(sys.argv[2][-3:] == ".h5")
		filepath = "models/" + sys.argv[2]

	# parameters
	T, N, prob, k_validate = 2, 1, 0.8, 10
	model_layers, hidden_dim, epochs = 3, 5, 10

	# initialize environment
	env_name = sys.argv[1]
	env = gym.make(env_name)
	state_dim, action_dim = env.observation_space.shape[0], env.action_space.n
	pi_expert = Expert_Policy(env_name)

	# run pretrained model vs train new model
	if len(sys.argv) > 2 and os.path.isfile(filepath):
		print("Loading pre-trained model from " + filepath)
		opt_model = load_model(filepath)
		pi_opt = NN_Policy(state_dim, action_dim)
		pi_opt.change_model(opt_model)
		visualize(env, pi_opt, pi_expert)

	else:
		# train policy
		policies = DAgger(T, N, prob, state_dim, action_dim, pi_expert, layers=model_layers, hidden_dim=hidden_dim, epochs=epochs)
		i_opt, t_opt, pred_opt = policy_validation(policies, k_validate, pi_expert)
		pi_opt = policies[i_opt]

		print("\n\nTraining complete!!!")
		print("Best policy was index {} out of {} with time and pred average {}, {}".format(i_opt, N, t_opt, pred_opt))
		visualize(env, pi_opt, pi_expert)

		if len(sys.argv) > 2:
			print("Saving model to file " + filepath)
			pi_opt.model.save(filepath)
    

	env.close()

