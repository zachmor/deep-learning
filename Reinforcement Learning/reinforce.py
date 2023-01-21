import os
import gym
import numpy as np
import tensorflow as tf

# Killing optional CPU driver warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# DO NOT ALTER MODEL CLASS OUTSIDE OF TODOs. OTHERWISE, YOU RISK INCOMPATIBILITY
# WITH THE AUTOGRADER AND RECEIVING A LOWER GRADE.


class Reinforce(tf.keras.Model):
    def __init__(self, state_size, num_actions):
        """
        The Reinforce class that inherits from tf.keras.Model
        The forward pass calculates the policy for the agent given a batch of states.

        :param state_size: number of parameters that define the state. You don't necessarily have to use this, 
                           but it can be used as the input size for your first dense layer.
        :param num_actions: number of actions in an environment
        """
        super(Reinforce, self).__init__()
        self.num_actions = num_actions        
        
        # TODO: Define network parameters and optimizer
        self.hidden_size = 500
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.002)

        self.l1 = tf.keras.layers.Dense(self.hidden_size, activation="relu")#with or without you
        self.l2 = tf.keras.layers.Dense(self.num_actions, activation="softmax", kernel_regularizer="l1") 

    @tf.function
    def call(self, states):
        """
        Performs the forward pass on a batch of states to generate the action probabilities.
        This returns a policy tensor of shape [episode_length, num_actions], where each row is a
        probability distribution over actions for each state.

        :param states: An [episode_length, state_size] dimensioned array
        representing the history of states of an episode
        :return: A [episode_length,num_actions] matrix representing the probability distribution over actions
        of each state in the episode
        """
        # TODO: implement this ~
        #print("states: ", states)
        return self.l2(self.l1(states))

    def loss(self, states, actions, discounted_rewards):
        """
        Computes the loss for the agent. Make sure to understand the handout clearly when implementing this.

        :param states: A batch of states of shape [episode_length, state_size]
        :param actions: History of actions taken at each timestep of the episode (represented as an [episode_length] array)
        :param discounted_rewards: Discounted rewards throughout a complete episode (represented as an [episode_length] array)
        :return: loss, a TensorFlow scalar
        """
        # TODO: implement this uWu
        # Hint: Use gather_nd to get the probability of each action that was actually taken in the episode.
        prbs_all = self.call(np.array(states))
        #print("Prbs_all: ", prbs_all)
        prbs_actions = [prbs_all[i,action] for i,action in enumerate(actions)]
        #print("prbs_actions: ", prbs_actions)
        #print("actions: ", actions)
        loss = tf.reduce_sum(tf.multiply(-tf.math.log(prbs_actions), discounted_rewards))
        #print("loss: ", loss)
        return loss