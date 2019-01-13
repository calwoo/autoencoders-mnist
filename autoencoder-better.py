import tensorflow as tf 
import numpy as np 
import matplotlib.pyplot as plt 

import functools

# import mnist from tensorflow datasets
mnist = tf.keras.datasets.mnist
(x_train, y_train),(x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

# autoencoder
class Model:
    def __init__(self, encoding_dim, learning_rate):
        self.inputs = tf.placeholder(tf.float32, [None, 28*28])
        self.encoding_dim = encoding_dim
        self.learning_rate = learning_rate
        self.init = tf.global_variables_initializer()
        self.sess = tf.Session()

        self.hidden = self.encoder(self.inputs)
        self.outputs = self.decoder(self.hidden)

        # self.loss = tf.reduce_mean(tf.square(self.outputs - self.inputs))
        """
        We are going to use the binary cross-entropy loss per-pixel.
        """
        cross_entropies = self.inputs * tf.log(self.outputs) + (1-self.inputs) * tf.log(1-self.outputs)
        self.loss = -tf.reduce_mean(cross_entropies)
        self.optimizer = tf.train.AdamOptimizer(learning_rate).minimize(self.loss)

    def encoder(self, inputs):
        with tf.name_scope("encoder") as scope:
            hidden = tf.layers.dense(
                inputs,
                units=self.encoding_dim,
                activation=tf.nn.relu)
        return hidden

    def decoder(self, hidden):
        with tf.name_scope("decoder") as scope:
            outputs = tf.layers.dense(
                hidden,
                units=784,
                activation=tf.nn.sigmoid)
        return outputs

    def test(self, slice):
        n = slice.shape[0]  # how many digits we will display
        plt.figure(figsize=(20, 4))
        for i in range(n):
            # display original
            ax = plt.subplot(2, n, i + 1)
            plt.imshow(slice[i].reshape(28, 28))
            plt.gray()
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)

            # display reconstruction
            compressed_imgs = self.sess.run(self.outputs, feed_dict={self.inputs:slice})
            ax = plt.subplot(2, n, i + 1 + n)
            plt.imshow(compressed_imgs[i].reshape(28, 28))
            plt.gray()
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
        plt.show()

    def train(self, inputs, epochs=50):
        self.sess.run(tf.global_variables_initializer())
        for i in range(epochs):
            _loss, _ = self.sess.run([self.loss, self.optimizer],
                feed_dict={self.inputs:inputs})
            print("epoch %d, loss = %.03f" % (i, _loss))

learning_rate = 0.01
model = Model(32, learning_rate)
model.train(x_train.reshape(-1, 784))

slice = x_test.reshape(-1, 784)[:10]
model.test(slice)