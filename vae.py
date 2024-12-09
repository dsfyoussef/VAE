 # -*- coding: utf-8 -*-
"""
Created on Thu Nov 21 13:48:12 2024

@author: youss
"""

import tensorflow as tf
from keras import layers, Model

# 1. Encoder
class Encoder(Model):
    def __init__(self, latent_dim):
        super(Encoder, self).__init__()
        self.conv1 = layers.Conv2D(32, kernel_size=5, activation=layers.LeakyReLU(0.02), padding="same")
        self.conv2 = layers.Conv2D(64, kernel_size=5, activation=layers.LeakyReLU(0.02), strides=2, padding="same")
        self.flatten = layers.Flatten()
        self.dense = layers.Dense(1024, activation="selu")
        self.mu = layers.Dense(latent_dim)
        self.log_var = layers.Dense(latent_dim)

    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.conv2(x)
        x = self.flatten(x)
        x = self.dense(x)
        mu = self.mu(x)
        log_var = self.log_var(x)
        return mu, log_var


# 2. Sampling Layer
class Sampling(layers.Layer):
    def call(self, inputs):
        mu, log_var = inputs
        epsilon = tf.random.normal(shape=tf.shape(mu))  # Sample from normal distribution
        return mu + tf.exp(0.5 * log_var) * epsilon


# 3. Decoder
class Decoder(Model):
    def __init__(self, latent_dim):
        super(Decoder, self).__init__()
        self.dense1 = layers.Dense(1024, activation="selu")
        self.dense2 = layers.Dense(8192, activation="selu")
        self.reshape = layers.Reshape((4, 4, 512))
        self.deconv1 = layers.Conv2DTranspose(256, kernel_size=5, strides=2, activation=layers.LeakyReLU(0.02), padding="same")
        self.deconv2 = layers.Conv2DTranspose(128, kernel_size=5, strides=2, activation=layers.LeakyReLU(0.02), padding="same")
        self.deconv3 = layers.Conv2DTranspose(64, kernel_size=5, strides=2, activation=layers.LeakyReLU(0.02), padding="same")
        self.deconv4 = layers.Conv2DTranspose(32, kernel_size=5, strides=2, activation=layers.LeakyReLU(0.02), padding="same")
        self.output_layer = layers.Conv2DTranspose(3, kernel_size=5, activation="sigmoid", padding="same")

    def call(self, z):
        x = self.dense1(z)
        x = self.dense2(x)
        x = self.reshape(x)
        x = self.deconv1(x)
        x = self.deconv2(x)
        x = self.deconv3(x)
        x = self.deconv4(x)
        return self.output_layer(x)


# 4. VAE
class VAE(Model):
    def __init__(self, encoder, decoder):
        super(VAE, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.sampling = Sampling()

    def call(self, inputs):
        mu, log_var = self.encoder(inputs)
        z = self.sampling([mu, log_var])
        reconstructed = self.decoder(z)
        return reconstructed, mu, log_var



