# PPO Implementation
# Author: John Hallman

import numpy as np
import sys
import os
import time
import gym
import math
import matplotlib.pyplot as plt

# import ML packages
import tensorflow as tf 


# returns a full model that can be trained over a given gym environment
class PPO:
	def __init__(self, env):
		self.env = env
		sess = tf.Session()


	def train(epochs=10, batch_size=32):
		with tf.variable_scope('actor'):
			input_ph, output_ph, output_pred = actor()
			mse = tf.reduce_mean(0.5 * tf.square(output_pred - output_ph))
			opt = tf.train.AdamOptimizer().minimize(mse)
			sess.run(tf.global_variables_initializer())
			saver = tf.train.Saver()

			for training_step in range(10000):
				# get a random subset of the training data
				indices = np.random.randint(low=0, high=len(inputs), size=batch_size)
				input_batch = inputs[indices]
				output_batch = outputs[indices]

				# run the optimizer and get the mse
				_, mse_run = sess.run([opt, mse], feed_dict={input_ph: input_batch, output_ph: output_batch})

				# print the mse every so often
				if training_step % 1000 == 0:
					print('{0:04d} mse: {1:.3f}'.format(training_step, mse_run))
					save_path = saver.save(sess, "/tmp/model.ckpt")
					print("Model saved in path: %s" % save_path)


# actor in the actor critic model
def actor(state_dim, action_dim):

	input_ph = tf.placeholder(dtype=tf.float32, shape=[None, state_dim])
	output_ph = tf.placeholder(dtype=tf.float32, shape=[None, action_dim])

	W0 = tf.get_variable(name='W0', shape=[state_dim, 100], initializer=tf.contrib.layers.xavier_initializer())
	W1 = tf.get_variable(name='W1', shape=[100, 50], initializer=tf.contrib.layers.xavier_initializer())
	W2 = tf.get_variable(name='W2', shape=[50, action_dim], initializer=tf.contrib.layers.xavier_initializer())

	b0 = tf.get_variable(name='b0', shape=[100], initializer=tf.constant_initializer(0.))
	b1 = tf.get_variable(name='b1', shape=[50], initializer=tf.constant_initializer(0.))
	b2 = tf.get_variable(name='b2', shape=[action_dim], initializer=tf.constant_initializer(0.))

	weights = [W0, W1, W2]
	biases = [b0, b1, b2]
	activations = [tf.nn.relu, tf.nn.relu, None]

	# computation graph
	layer = input_ph
	for W, b, activation in zip(weights, biases, activations):
		layer = tf.matmul(layer, W) + b
		if activation is not None:
			layer = activation(layer)
	output_pred = layer

	return input_ph, output_ph, output_pred





