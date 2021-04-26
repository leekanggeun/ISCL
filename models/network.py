import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import Model
import sys
sys.path.append('ISCL/')
from utils.normalization import BIN, IN, BN

class Res_Block(Model):
    def __init__(self, num_filters, activation, initializer='he_normal'):
        super(Res_Block, self).__init__()
        self.activation = activation
        self.initializer = initializer
        self.h2 = layers.Conv2D(num_filters, (3,3), (1,1), activation=None, padding="valid", use_bias=True, kernel_initializer=self.initializer)
        self.h3 = layers.Conv2D(num_filters, (3,3), (1,1), activation=None, padding="valid", use_bias=True, kernel_initializer=self.initializer)

        self.s1 = BIN(activation)
        self.s2 = BIN(activation)

    def call(self, inputs, training=True):
        padding = tf.constant([[0,0,], [1, 1,], [1,1,], [0,0]])

        h2 = self.h2(tf.pad(inputs, padding, "SYMMETRIC"))
        h2 = self.s1(h2, training=training)

        h3 = self.h3(tf.pad(h2, padding, "SYMMETRIC"))
        h3 = self.s2(h3, act=False, training=training)

        return inputs+h3

class Extractor(Model):
    def __init__(self, start_neuron=16, out_channel=1, initializer='he_normal', n_layers=20):
        super(Extractor, self).__init__()
        self.conv = []
        self.bin = []
        self.initializer = initializer
        self.n_layers = n_layers
        self.conv.append(layers.Conv2D(start_neuron, (3,3), (1,1), activation=tf.nn.leaky_relu, padding="same", use_bias=True, kernel_initializer=self.initializer))
        for i in range(0,self.n_layers-2):
            self.conv.append(layers.Conv2D(start_neuron, (3,3), (1,1), activation=None, padding="same", use_bias=True, kernel_initializer=self.initializer))
            self.bin.append(BIN(tf.nn.leaky_relu))
        self.conv.append(layers.Conv2D(out_channel, (3,3), (1,1), activation=None, padding="same", use_bias=True, kernel_initializer=self.initializer))
    def call(self, input, training=True):
        h = self.conv[0](input)
        for conv, bin in zip(self.conv[1:-1], self.bin):
            h = conv(h)
            h = bin(h, training=training)
        h = self.conv[-1](h)
        return h

class Discriminator(Model):
    def __init__(self, start_neuron=16, initializer='he_normal'):
        super(Discriminator, self).__init__()
        self.start_neuron = start_neuron
        self.initializer = initializer
        self.conv1 = layers.Conv2D(self.start_neuron, (4,4), (2,2), activation=tf.nn.leaky_relu, padding="same", use_bias=True, kernel_initializer=self.initializer) # 32 32
        self.conv2 = layers.Conv2D(self.start_neuron*2, (4,4), (2,2), activation=None, padding="same", use_bias=True, kernel_initializer=self.initializer) # 16 16
        self.conv3 = layers.Conv2D(self.start_neuron*4, (4,4), (2,2), activation=None, padding="same", use_bias=True, kernel_initializer=self.initializer) # 8 8
        self.conv4 = layers.Conv2D(self.start_neuron*8, (4,4), (2,2), activation=None, padding="same", use_bias=True, kernel_initializer=self.initializer) # 4 4
        self.conv5 = layers.Conv2D(self.start_neuron*16, (4,4), (1,1), activation=None, padding="valid", use_bias=True, kernel_initializer=self.initializer) # 1 1
        self.conv6 = layers.Conv2D(1, (1,1), (1,1), activation=None, padding="valid", use_bias=True, kernel_initializer=self.initializer) # 1 1

        self.s1 = BIN(tf.nn.leaky_relu)
        self.s2 = BIN(tf.nn.leaky_relu)
        self.s3 = BIN(tf.nn.leaky_relu)
        self.s4 = BIN(tf.nn.leaky_relu)
        

    def call(self, input, training=True):
        h = self.conv1(input)
        h = self.conv2(h)
        h = self.s1(h, training=training)
        h = self.conv3(h)
        h = self.s2(h, training=training)
        h = self.conv4(h)
        h = self.s3(h, training=training)
        h = self.conv5(h)
        h = self.s4(h, training=training)
        h = self.conv6(h)
        return h


class Generator(Model):
    def __init__(self, output_dim, start_neuron=32, nres=3, initializer='he_normal', last=tf.nn.tanh):
        super(Generator, self).__init__()
        self.start_neuron = start_neuron
        self.initializer = initializer
        self.nres = nres
        self.conv1 = layers.Conv2D(self.start_neuron*1, (7,7), (1,1), activation=None, padding="valid", use_bias=True,kernel_initializer=self.initializer) # 64 64 
        self.s1 = BIN(activation=tf.nn.leaky_relu)
        self.conv2 = layers.Conv2D(self.start_neuron*2, (4,4), (2,2), activation=None, padding="valid", use_bias=True, kernel_initializer=self.initializer) # 32 32 
        self.s2 = BIN(activation=tf.nn.leaky_relu)
        self.conv3 = layers.Conv2D(self.start_neuron*4, (4,4), (2,2), activation=None, padding="valid", use_bias=True, kernel_initializer=self.initializer) # 16 16 
        self.s3 = BIN(activation=tf.nn.leaky_relu)
        
        self.res = []
        for i in range(0,nres):
            self.res.append(Res_Block(self.start_neuron*4, tf.nn.leaky_relu, initializer=self.initializer)) # 16 16 
        self.conv4 = layers.Conv2DTranspose(start_neuron*2, (3,3), activation=None, strides=(2,2), padding="same", use_bias=True, kernel_initializer=self.initializer) # 32 32 
        self.s4 = BIN(activation=tf.nn.leaky_relu)
        self.conv5 = layers.Conv2DTranspose(start_neuron*1, (3,3), activation=None, strides=(2,2), padding="same", use_bias=True, kernel_initializer=self.initializer) # 64 64 
        self.s5 = BIN(activation=tf.nn.leaky_relu)
        self.conv6 = layers.Conv2D(output_dim, (7,7), (1,1), activation=last, padding="valid", use_bias=True, kernel_initializer=self.initializer)

    def call(self, input, training=True):
        padding1 = tf.constant([[0,0,], [3,3,], [3,3,], [0,0]])
        padding2 = tf.constant([[0,0,], [1,1,], [1,1,], [0,0]])
        h = self.conv1(tf.pad(input, padding1, "SYMMETRIC"))
        h = self.s1(h, training=training)
        h = self.conv2(tf.pad(h, padding2, "SYMMETRIC"))
        h = self.s2(h, training=training)
        h = self.conv3(tf.pad(h, padding2, "SYMMETRIC"))
        h = self.s3(h, training=training)
        for h1 in self.res:
            h = h1(h, training=training)
        h = self.conv4(h)
        h = self.s4(h, training=training)
        h = self.conv5(h)
        h = self.s5(h, training=training)
        h = self.conv6(tf.pad(h, padding1, "SYMMETRIC")) 
        return h
