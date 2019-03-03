# Test DAgger algorithm on cartpole environment

import numpy as np
import sys
import time
import gym
import math



# trains policy on trajectories based on initial policy, alpha is training rate
# uses SQUARE-LOSS instead of logistic loss as cartpole isn't a true classification problem
def train_policy(trajects, state_dim, initial_policy=None, alpha=0.005):

	# project policy onto unit vector if too large
	project = lambda v: v / np.linalg.norm(v) if (np.linalg.norm(v) > 1) else v

	if not initial_policy:
		initial_policy = np.zeros(state_dim)
		#initial_policy = np.random.uniform(size=(state_dim,)) / np.sqrt(state_dim)

	for trajectory in trajects:
		for time_point in trajectory:
			x, y = np.array(time_point[0]), time_point[1] # x = state, y = action (dot product, not (0, 1) classification)
			y_hat = np.dot(x, initial_policy) # prediction
			initial_policy -= alpha * (y - y_hat) * x
			initial_policy = project(initial_policy)
	return initial_policy


# simulate an expert for T rounds, return trajectories
def simulate_trajectories(T, pi_n, pi_expert, beta, time_limit=1000):
	expert_policy = lambda obs: 0 if (pi_expert.dot(obs) < 0) else 1
	trained_policy = lambda obs: 0 if (pi_n.dot(obs) < 0) else 1
	trajects = []
	for traject_round in range(T):
		observation = env.reset()
		curr_traject = []
		policy = expert_policy if (np.random.binomial(1, beta) == 1) else trained_policy
		for t in range(time_limit):
			action = policy(observation)
			curr_traject.append([observation, expert_policy(observation)])
			observation, reward, done, info = env.step(action)
			if done:
				break
		trajects.append(curr_traject)
	return trajects


# T is the length of a single round/trajectory, N is the number of times to run the algorithm
# p is the probability that we use to decrease beta with at each round
# pi_expert the policy of the expert (np array of length 4)
# pi_start the first policy (initialized randomly if None)
def DAgger(T, N, p, pi_expert, pi_start=None):

	# D is set of trajectories, beta is mixing probability
	D, beta, state_dim = [], 1.0, 4 # state_dim = 4 for cartpole
	if pi_start == None:
		pi_start = np.zeros(state_dim)
		#pi_start = np.random.normal(size=(state_dim,)) / np.sqrt(state_dim)

	pi_trained = [pi_start]
	checkpoint = 10
	print("Starting Training for {} rounds and {} trajectories per round!".format(N, T))
	print("Initial policy vector:")
	print(pi_start)
	for n in range(N):
		beta = beta * p
		D += simulate_trajectories(T, pi_trained[-1], pi_expert, beta)
		pi_new = train_policy(D, state_dim)
		pi_trained.append(pi_new)
		if (100 * (n + 1) / N >= checkpoint):
			print("{}% towards completion, current vector:".format(checkpoint))
			print(pi_trained[-1])
			checkpoint += 10

	# pi_opt = validation(pi_set) # complete later
	pi_opt = pi_trained[-1] # return last policy
	return pi_opt



# train policy using DAgger and expert policy ([0.1, 0.1, 0.5, 0.1])
if __name__ == "__main__":

	# parameters
	T, N, p, pi_expert = 1000, 5, 0.99, np.array([0.1, 0.1, 0.5, 0.1])

	# learn the optimal vector
	env = gym.make('CartPole-v1')
	pi_learned = DAgger(T, N, p, pi_expert)

	# test resulting policy
	print("Training complete, learned policy is:")
	print(pi_learned)

	obs = env.reset()
	for t in range(1000):
	    env.render()
	    time.sleep(0.1) # slows down process to make it more visible
	    val = obs.dot(pi_learned)
	    action = 0 if (val < 0) else 1
	    obs, reward, done, info = env.step(action)
	    if done:
	        print("Final episode: lasted {} timesteps, data: {}".format(t+1, obs))
	        break
    
	env.close()


