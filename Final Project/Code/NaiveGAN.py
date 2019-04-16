import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd
from keras.layers import Input
from keras.models import Model, Sequential
from keras.layers.core import Dense, Dropout
from keras.layers.advanced_activations import LeakyReLU
from keras.datasets import mnist
from keras.optimizers import Adam
from keras import initializers


def get_optimizer():
    return Adam(lr=0.0002, beta_1=0.5)

def get_generator(optimizer):
    generator = Sequential()
    generator.add(Dense(64, input_dim=random_dim, kernel_initializer=initializers.RandomNormal(stddev=0.02)))
    generator.add(LeakyReLU(0.2))

    generator.add(Dense(128))
    generator.add(LeakyReLU(0.2))

    generator.add(Dense(256))
    generator.add(LeakyReLU(0.2))

    generator.add(Dense(X_dim, activation='tanh'))
    generator.compile(loss='mean_absolute_error', optimizer=optimizer)
    return generator

def get_discriminator(optimizer):
    discriminator = Sequential()
    discriminator.add(Dense(256, input_dim=X_dim, kernel_initializer=initializers.RandomNormal(stddev=0.02)))
    discriminator.add(LeakyReLU(0.2))
    discriminator.add(Dropout(0.3))

    discriminator.add(Dense(128))
    discriminator.add(LeakyReLU(0.2))
    discriminator.add(Dropout(0.3))

    discriminator.add(Dense(64))
    discriminator.add(LeakyReLU(0.2))
    discriminator.add(Dropout(0.3))

    discriminator.add(Dense(1, activation='tanh'))
    discriminator.compile(loss='binary_crossentropy', optimizer=optimizer)
    return discriminator

def get_gan_network(discriminator, random_dim, generator, optimizer):
    # We initially set trainable to False since we only want to train either the
    # generator or discriminator at a time
    discriminator.trainable = False
    # gan input (noise) will be 100-dimensional vectors
    gan_input = Input(shape=(random_dim,))
    # the output of the generator (an image)
    x = generator(gan_input)
    # get the output of the discriminator (probability if the image is real or not)
    gan_output = discriminator(x)
    gan = Model(inputs=gan_input, outputs=gan_output)
    gan.compile(loss='mean_absolute_error', optimizer=optimizer)
    gan.summary()
    return gan

def train(X, Y, epochs=1, batch_size=128, train_percent = 0.9):
    # Get the training and testing data

    # Split the training data into batches of size 128

    fold = int(X.shape[0]*train_percent)
    x_train = 	X[:fold]
    x_test = 	X[fold+1:]
    y_train = 	Y[:fold]
    y_test = 	Y[fold+1:]

    batch_count = x_train.shape[0] // batch_size

    # Build our GAN netowrk
    adam = get_optimizer()
    generator = get_generator(adam)
    discriminator = get_discriminator(adam)
    gan = get_gan_network(discriminator, random_dim, generator, adam)

    for e in range(1, epochs+1):
        print('-'*15, 'Epoch %d' % e, '-'*15)
        for _ in range(batch_count):
            # Get a random set of input noise and images
            noise = np.random.normal(0, 1, size=[batch_size, random_dim])
            image_batch = x_train[np.random.randint(0, x_train.shape[0], size=batch_size)]

            # Generate fake data
            generated_images = generator.predict(noise)
            X = np.concatenate([image_batch, generated_images])

            # Labels for generated and real data
            y_dis = np.zeros(2*batch_size)
            # One-sided label smoothing
            y_dis[:batch_size] = 0.9

            # Train discriminator
            discriminator.trainable = True
            discriminator.train_on_batch(X, y_dis)

            # Train generator
            noise = np.random.normal(0, 1, size=[batch_size, random_dim])
            y_gen = np.ones(batch_size)
            discriminator.trainable = False
            gan.train_on_batch(noise, y_gen)

        y_gen = np.ones(x_test.shape[0])
        print("Test Loss:" + str(discriminator.test_on_batch(x_test, y_gen)))

baseDirectory = '../'
dataDirectory = baseDirectory + "Data/"

f1 = pd.read_csv(dataDirectory + 'CompleteData_Upod1.csv')
f2 = pd.read_csv(dataDirectory + 'CompleteData_Upod2.csv')

Xchannels = ['Fig1r', 'Fig2r', 'Temp', 'RH']
Ychannels = ['Acet', 'Benz', 'Form', 'Meth', 'Tol']

X = f1[Xchannels].as_matrix()
Y = f1[Ychannels].as_matrix()

# To make sure that we can reproduce the experiment and get the same results
np.random.seed(10)

# The dimension of our random noise vector.
random_dim = 100
X_dim = X.shape[1]
Y_Dim = Y.shape[1]

train(X,Y,500,64)