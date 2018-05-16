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

#-------------------------------------------------------------------------------

# f = Ae^(-((t-B)^2)/C^2) * sin(Dt + E)
# fix A, B, C, D, and E
# Create a loop which loops through 8000 values for t between 0 and 1 and
# calculates the value f for each of these 8000 values
# Shift the function by replacing t with (t - F) and calculate the value of
# f for each the same 8000 values of t. Repeat this.
# This should be stored in a matrix

# NOT THE FOLLOWING
# [
#     [
#         [1, 2, ... 8129]
#         [1, 2, ... 8129]
#         [1, 2, ... 8129]
#     ],
#     [
#         [1, 2, ... 8129]
#         [1, 2, ... 8129]
#         [1, 2, ... 8129]
#     ]
# ]
# THE FOLLOWING IS RIGHT
# [
#     [1, 2, ... 8129],   # Wave 1
#     [1, 2, ... 8129],   # Wave 1 shifted
#     [1, 2, ... 8129],   # Wave 1 shifted again
#     [1, 2, ... 8129],   # Wave 2
#     [1, 2, ... 8129],   # Wave 2 shifted
#     [1, 2, ... 8129]    # Wave 2 shifted again
# ]

# np.array --> fill with 0s (np.zeroes) to set the dimension, and then fill
# each corresponding value with the value of the function.
# save this array in a txt file, then load batch by batch into the neural network

waves_clean = []
waves_shifted_one = []
waves_shifted_two = []
waves_shifted_three = []

time = np.linspace(0, 1, 11)    # Change the 11 to 8130 to get a intervals of 8129

def myfunc(t):
    return A * np.exp(-(np.square((t-F)-B))/(np.square(C))) * np.sin(D*t + E)

vfunc = np.vectorize(myfunc)

for x in range(0, 4):           # Change the 4 to 10,000 for 10,000 data sets
    A = np.random.rand(1)
    B = np.random.rand(1)
    C = np.random.rand(1)
    D = 100 + np.random.rand(1)
    E = np.random.rand(1)
    F = 0

    for x in range(0, 4):   # 4 as 1 clean and 3 shifted

        if (F == 0):
            waves_clean.append(vfunc(time))
        elif (F == 0.25):
            waves_shifted_one.append(vfunc(time))
        elif (F == 0.50):
            waves_shifted_two.append(vfunc(time))
        elif (F == 0.75):
            waves_shifted_three.append(vfunc(time))

        F += 0.25                # Should this be random as well?

    F = 0

# plt.plot(waves_clean, time)
# plt.show()
# print(waves_shifted_one)        # array of 4 arrays as data size is currently 4

# print("Clean")
# print(waves_clean)
# print("Shifted #1")
# print(waves_shifted_one)
# print("Shifted #2")
# print(waves_shifted_two)
# print("Shifted #3")
# print(waves_shifted_three)

np.savetxt('waves_clean.txt', waves_clean, fmt='%.5e')
np.savetxt('waves_shifted_one.txt', waves_shifted_one, fmt='%.5e')
np.savetxt('waves_shifted_two.txt', waves_shifted_two, fmt='%.5e')
np.savetxt('waves_shifted_three.txt', waves_shifted_three, fmt='%.5e')

np.set_printoptions(precision = 17)
data_clean = np.loadtxt("waves_clean.txt", dtype = np.float64)
data_shifted_one = np.loadtxt("waves_shifted_one.txt", dtype = np.float64)
data_shifted_two = np.loadtxt("waves_shifted_two.txt", dtype = np.float64)
data_shifted_three = np.loadtxt("waves_shifted_three.txt", dtype = np.float64)

batch_size = 2
# print("---")
# print(data_clean.shape)
# print(data_shifted_one.shape)
# print(data_shifted_two.shape)
# print(data_shifted_three.shape)             # these 4 print out (4, 11)
# print(data_clean[0:2].shape)
# print(data_shifted_one[0:2].shape)
# print(data_shifted_two[0:2].shape)
# print(data_shifted_three[0:2].shape)        # these 4 print out (2, 11)

# I could restructure the generation of waves to create a [11, 3] matrix for each clean
# wave instead of three separate files, and create a larger array which holds all of
# these arrays.
# order of the axes shouldn't matter -- could try to figure out the dimensions for w and
# b such that wX + b where x.shape = [3, 2, 11] gives me an output with shape [1, 2, 11]

batch = array([data_shifted_one[0:batch_size], data_shifted_two[0:batch_size], data_shifted_three[0:batch_size]])
# print(batch.shape)      # prints out (3, 2, 11) as 2 is batch_size
# print(batch)
# # print(batch[0][0].shape)
batch = np.transpose(batch, (1, 2, 0))
# print(batch.shape)      # prints out (2, 11, 3) where 2 is the batch_size
# print(batch)

# print(batch[0].shape)
# print(batch[0][0].shape)
#
# print(batch.shape)
# noised_batch = batch
# print(noised_batch.shape)

noisy_shifted_one = data_shifted_one
noisy_shifted_two = data_shifted_two
noisy_shifted_three = data_shifted_three

for i in range(0, 4):       # 4 to be replaced by 10,000

    for j in range(0, 11):  # 11 to be replaced by 8130

        mu, sigma = 0, 1 # mean and standard deviation
        noise = np.random.normal(mu, sigma, 1000)
        noisy_shifted_one[i][j] += noise[0]
        noise = np.random.normal(mu, sigma, 1000)
        noisy_shifted_two[i][j] += noise[0]
        noise = np.random.normal(mu, sigma, 1000)
        noisy_shifted_three[i][j] += noise[0]
        # add a random value, or should normal distribution be added?

# print(noisy_shifted_one.shape)  # (4, 11) where 4 should be 10,000
# print(noised_batch.shape)

x = tf.placeholder(tf.float32, [None, 11, 3])   # None for variable batch_size
W = tf.Variable(tf.random_normal([batch_size, 3, 1]))
b = tf.Variable(tf.random_normal([1]))
y = tf.matmul(x, W) + b
# print(y)

y_ = tf.placeholder(tf.int64, [batch_size, 11, 1]) # Label, i.e. Correct answer

# May not be able to use this as this has something to do with Softmax, which
# is only for creating an output of probabilities which add up to 1.
cross_entropy = tf.losses.mean_squared_error(labels=y_, predictions=y)
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

sess = tf.InteractiveSession()
tf.global_variables_initializer().run()

# print(data_clean.shape)

epochs = 2
for m in range(0, epochs):
    min = 0
    max = batch_size
    for n in range(0, int(4/batch_size)):       # here the 4 will be replaced by 10,000
        current_batch_x = array([noisy_shifted_one[min:max], noisy_shifted_two[min:max], noisy_shifted_three[min:max]])
        batch_xs = np.transpose(current_batch_x, (1, 2, 0))
        current_batch_y = data_clean[min:max]
        batch_ys = current_batch_y[..., newaxis]    # to change (2, 11) to (2, 11, 1)
        _, acc_train = sess.run([train_step, cross_entropy], feed_dict={x: batch_xs, y_: batch_ys})
        # print(acc_train)
        # when i = 1, [0:2], when i = 2, [2:4]
        min += batch_size
        max += batch_size
        print("Epoch: " + str(m) + ", iteration: " + str(n))
        print(sess.run(
              cross_entropy, feed_dict={
                  x: batch_xs,
                  y_: batch_ys
              }))

#-------------------------------------------------------------------------------

  # Test trained model

waves_clean_test = []
waves_shifted_one_test = []
waves_shifted_two_test = []
waves_shifted_three_test = []

for _ in range(0, 4):           # Change the 4 to 10,000 for 10,000 data sets
     A = np.random.rand(1)
     B = np.random.rand(1)
     C = np.random.rand(1)
     D = 100 + np.random.rand(1)
     E = np.random.rand(1)
     F = 0

     for _ in range(0, 4):

         if (F == 0):
             waves_clean_test.append(vfunc(time))
         elif (F == 0.25):
             waves_shifted_one_test.append(vfunc(time))
         elif (F == 0.50):
             waves_shifted_two_test.append(vfunc(time))
         elif (F == 0.75):
             waves_shifted_three_test.append(vfunc(time))

         F += 0.25                # Should this be random as well?

     F = 0

# batch_test = array([data_shifted_one_test, data_shifted_two_test, data_shifted_three_test])
# print(batch.shape)      # prints out (3, 2, 11) as 2 is batch_size
# print(batch)
# # print(batch[0][0].shape)
# batch_test_x = np.transpose(batch_test, (1, 2, 0))

noisy_shifted_one_test = waves_shifted_one_test
noisy_shifted_two_test = waves_shifted_two_test
noisy_shifted_three_test = waves_shifted_three_test

for i in range(0, 4):

    for j in range(0, 11):

        mu, sigma = 0, 1 # mean and standard deviation
        noise = np.random.normal(mu, sigma, 1000)
        noisy_shifted_one_test[i][j] += noise[0]
        noise = np.random.normal(mu, sigma, 1000)
        noisy_shifted_two_test[i][j] += noise[0]
        noise = np.random.normal(mu, sigma, 1000)
        noisy_shifted_three_test[i][j] += noise[0]

batch_test = array([noisy_shifted_one_test, noisy_shifted_two_test, noisy_shifted_three_test])
batch_test_x = np.transpose(batch_test, (1, 2, 0))

np.savetxt('waves_clean_test.txt', waves_clean_test, fmt='%.5e')
data_clean_test = np.loadtxt("waves_clean_test.txt", dtype = np.float64)

# print(data_clean_test.shape)
batch_ys = data_clean_test[..., newaxis]
batch_ys = batch_ys[:batch_size, :, :]
batch_test_x_new = batch_test_x[:batch_size, :, :]
# print("-------------------------------------------------------------------")
# print(batch_ys.shape)           # (2, 11, 1)
# print(batch_test_x_new.shape)   # (2, 11, 3)
# print(x.shape)
# print(y_.shape)
# print("-------------------------------------------------------------------")

# print(y)
# print("-------------------------------------------------------------------")
float_y_ = tf.cast(y_, dtype=tf.float32)
# print(float_y_)

print(y.shape)
print(y_.shape)

correct_prediction = tf.equal(y, float_y_)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print(sess.run(
      cross_entropy, feed_dict={
          x: batch_test_x_new,
          y_: batch_ys
      }))
