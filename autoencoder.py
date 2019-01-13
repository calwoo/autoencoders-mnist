import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt 

# import mnist from tensorflow datasets
mnist = tf.keras.datasets.mnist
(x_train, y_train),(x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

# autoencoders are essentially just neural networks with hidden layers smaller than the input/output layers.
# this allows them to potentially learn features in an unsupervised manner.
class Model:
    def __init__(self, encoding_dim, learning_rate):
        self.inputs = tf.placeholder(tf.float32, [None, 28*28])
        self.encoding_dim = encoding_dim
        self.learning_rate = learning_rate
        self.sess = tf.Session()

        self.hidden = tf.layers.dense(
            self.inputs,
            self.encoding_dim,
            activation=tf.nn.relu)
        self.outputs = tf.layers.dense(
            self.hidden,
            784)
        self.loss = tf.reduce_mean(tf.square(self.outputs - self.inputs))
        self.optimizer = tf.train.AdamOptimizer(learning_rate).minimize(self.loss)

    def encoder(self, inputs):
        return self.sess.run(self.hidden, feed_dict={self.inputs: inputs})

    def decoder(self, hidden):
        return self.sess.run(sess.outputs, feed_dict={self.hidden: hidden})

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

    def train(self, inputs, epochs=30):
        self.sess.run(tf.global_variables_initializer())
        for i in range(epochs):
            _loss, _ = self.sess.run([self.loss, self.optimizer],
                feed_dict={self.inputs:inputs})
            print("epoch %d, loss = %.03f" % (i, _loss))

learning_rate = 0.05
model = Model(32, learning_rate)
model.train(x_train.reshape(-1, 784))

slice = x_test.reshape(-1, 784)[:10]
model.test(slice)

    