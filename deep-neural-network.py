from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import argparse
import sys

# import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from numpy import random
from numpy import array, newaxis

# hello = tf.constant('Hello, TensorFlow!')
# sess = tf.Session()
# # print(sess.run(hello))
#
# x1 = tf.constant([5])
# x2 = tf.constant([6])
#
# result = tf.multiply(x1, x2)
# # result = x1 * x2
#
# # print(sess.run(result))
# sess.close()

# ------------------------------------------------------------------------------

# Define variables
alpha = 1
theta = 0

# Define a single scalar Normal distribution.
# gaussian_function = tf.distributions.Normal(loc= 0., scale= 3.)

# Define the argument of the sinusoidal function
# sin_argument = tf.add(tf.multiply(alpha,x), theta)

# Defining the sin function, though this seems to be a predefined function
# sin_function = sin(
#     x,
#     name=None
# )

# Define Solution function
# S = gaussian_function * sin(sin_argument)

# Define the random Noise function
# G = tf.random_normal([2, 3], seed=1234)

# Define the Input function
# f = S + G

# ------------------------------------------------------------------------------

mu, sigma = 0, 1 # mean and standard deviation
gaussian = np.random.normal(mu, sigma, 1000)

# print("gauss =", gaussian[0])

alpha = random.rand(1)
# print("alpha =", alpha)
theta = random.rand(1)
# print("theta =", theta)

x = np.arange(0, 1/8129, 10)
f = gaussian * np.sin(alpha*x + theta)
f = np.multiply(gaussian, np.sin(np.add(np.multiply(alpha, x), theta)))
# print(g)

Fs = 8000
f = 5
sample = 8000
x = np.arange(sample)
y = np.sin(2 * np.pi * f * x / Fs)
# plt.plot(x, y)
# plt.xlabel('sample(n)')
# plt.ylabel('voltage(V)')
# plt.show()
