import sys
sys.path.append("/Library/Frameworks/Python.framework/Versions/3.7/lib/python3.7/site-packages")
import os
import numpy as np
import tensorflow as tf
import numpy as np
from preprocessing import *

class RWB(tf.keras.Model):
    def __init__(self):
        super(RWB, self).__init__()
        self.batch_size = 300
        self.epochs = 1
        self.stride = (1,1)

        self.regularizer = tf.keras.regularizers.l2(5e-4)

        # conv layers
        self.conv_1 = tf.keras.layers.Conv2D(500, (5,5), self.stride, padding="SAME", 
            activation="relu", kernel_regularizer=self.regularizer, use_bias=True)
        self.conv_2 = tf.keras.layers.Conv2D(500, (5,5), self.stride, padding="SAME", 
            activation="relu", kernel_regularizer=self.regularizer, use_bias=True)
        self.conv_3 = tf.keras.layers.Conv2D(500, (5,5), self.stride, padding="SAME", 
            activation="relu", kernel_regularizer=self.regularizer, use_bias=True)
        self.conv_4 = tf.keras.layers.Conv2D(500, (5,5), self.stride, padding="SAME", 
            activation="relu", kernel_regularizer=self.regularizer, use_bias=True)
        self.conv_5 = tf.keras.layers.Conv2D(500, (5,5), self.stride, padding="SAME", 
            activation="relu", kernel_regularizer=self.regularizer, use_bias=True)
        self.conv_6 = tf.keras.layers.Conv2D(500, (5,5), self.stride, padding="SAME", 
            activation="relu", kernel_regularizer=self.regularizer, use_bias=True)

        # pool layer
        self.pool = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=None, padding="SAME")

        # batch normalization layer
        self.batch_norm = tf.keras.layers.BatchNormalization()

        # dense layers
        self.dense_1 = tf.keras.layers.Dense(3000, activation="relu") # maybe tanh
        self.dense_2 = tf.keras.layers.Dense(3000, activation="relu")
        self.dense_3 = tf.keras.layers.Dense(320, activation="softmax") 

        self.optimizer = tf.keras.optimizers.SGD(learning_rate=5e-4, momentum=0.9)

    def call(self, inputs):
        conv1 = self.conv_1(inputs)
        norm_conv1 = self.batch_norm(conv1)
        pooled1 = self.pool(norm_conv1)

        conv2 = self.conv_2(pooled1)
        norm_conv2 = self.batch_norm(conv2)
        pooled2 = self.pool(norm_conv2)

        conv3 = self.conv_3(inputs)
        norm_conv3 = self.batch_norm(conv3)
        pooled3 = self.pool(norm_conv3)

        conv4 = self.conv_4(pooled3)
        norm_conv4 = self.batch_norm(conv4)
        pooled4 = self.pool(norm_conv4)

        conv5 = self.conv_5(pooled4)
        norm_conv5 = self.batch_norm(conv5)
        pooled5 = self.pool(norm_conv5)

        conv6 = self.conv_6(pooled5)
        norm_conv6 = self.batch_norm(conv6)

        conv6 = tf.reshape(norm_conv6, [len(inputs), -1]) # maybe?

        dense1 = self.dense_1(conv6)
        dense2 = self.dense_2(dense1)
        logits = self.dense_3(dense2)

        return logits

    def loss(self, logits, labels):
        loss = tf.reduce_mean(tf.keras.losses.sparse_categorical_crossentropy(labels, logits))
        return loss

    def accuracy(self, logits, labels):
        correct_predictions = tf.equal(tf.argmax(logits, 1), tf.argmax(labels, 1))
        return tf.reduce_sum(tf.cast(correct_predictions, tf.float32))


def train(model, train_inputs, train_labels):
    indices = tf.random.shuffle(range(len(train_inputs)))
    train_inputs = tf.gather(train_inputs, indices)
    train_labels = tf.gather(train_labels, indices)

    # divide images into batches
    to_split = list(filter(lambda x: x % model.batch_size == 0,
                           range(len(train_inputs))))
    input_batches = np.split(train_inputs, to_split)[1:]
    label_batches = np.split(train_labels, to_split)[1:]

    # for every batch, compute and descend gradients for model's weights
    for i in range(len(input_batches)):
        cur_input_batch = input_batches[i]
        cur_label_batch = label_batches[i]

        cur_input_batch = tf.image.random_flip_left_right(cur_input_batch)

        with tf.GradientTape() as tape:
            # print("curr input batch: ", cur_input_batch)
            predictions = model.call(cur_input_batch)
            loss = model.loss(predictions, cur_label_batch)

        gradients = tape.gradient(loss, model.trainable_variables)
        model.optimizer.apply_gradients(zip(gradients, model.trainable_variables))


def test(model, test_inputs, test_labels):
    to_split = list(filter(lambda x: x % model.batch_size == 0,
                           range(len(test_inputs))))
    input_batches = np.split(test_inputs, to_split)[1:]
    label_batches = np.split(test_labels, to_split)[1:]

    total_accuracy = 0
    # for every batch, compute and descend gradients for model's weights
    for i in range(len(input_batches)):
        cur_input_batch = input_batches[i]
        cur_label_batch = label_batches[i]

        predictions = model.call(cur_input_batch)
        batch_accuracy = model.accuracy(probs, test_labels)
        total_accuracy += batch_accuracy

    avg_accuracy = total_accuracy / len(test_labels)

    return avg_accuracy
    

def main():

    # load train and test data
    train_inputs, train_labels, test_inputs, test_labels = get_data()
    
    # create Model
    m = RWB()

    for i in range(m.epochs):
        train(m, train_inputs, train_labels)
        print("finished training epoch %d" % i)

    print("testing...")
    acc = test(m, test_inputs, test_labels)
    print("model received an accuracy of: %s" % str(acc))


if __name__ == '__main__':
    main()
