import numpy as np
from Reward import Reward
from checks import check_valid_state
import os
import sys
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from visualize import visualize_trajectory

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, Input, Flatten
import tensorflow_probability as tfp

tfd = tfp.distributions


class GFNAgent(Model):
    """Example Generative Flow Network as described in:
    https://arxiv.org/abs/2106.04399 and
    https://arxiv.org/abs/2201.13259
    """

    def __init__(
        self,
        initial_state,
        n_hidden=32,
        name="",
        epochs=100,
        lr=0.005,
        # not sure
        gamma=0.5,
        env_dim=2,
        env_r0=0.01,
    ):
        """Initialize GFlowNet agent.
        :param env_dim: (int) Number of dimensions in the reward environment
        :param env_len: (int) Length of each dimension in the environment
        :param env_r0: (float) r0 value in the environment
        :param name: (str) Agent name
        :param n_hidden: (int) Number of nodes in hidden layer of neural network
        :param gamma: (float) Mixture proportion when sampling from forward policy,
                      and mixing with uniform distribution
        :param epochs: (int) Number of epochs to complete during training cycle
        :param lr: (float) Learning rate
        :return: (None) Initialized class object
        """
        super().__init__()
        # I understand
        self.height, self.width = initial_state[0].shape
        assert self.width > 2
        assert self.height > 2
        self.name = name
        self.action_space = (
            2 * self.width * self.height * 4 + 1
        )  # sping times each site times 4 directions + exit action
        self.n_hidden = n_hidden
        self.epochs = epochs
        self.env_reward = Reward(lattice_width=self.width, lattice_height=self.height)
        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=lr, decay_steps=10000, decay_rate=0.8
        )
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
        self.dim = (
            self.action_space - 1
        )  # backwards move, all in action space minus exit action
        self.gamma = gamma
        self.stop_action = self.action_space - 1
        self.data = {"positions": None, "actions": None, "rewards": None}
        self.get_model()
        self.max_trajectory_len = 10  # 10?

        # I think I do
        # self.clear_eval_data()

        # Not sure
        # assert env_dim > 1
        # assert 0 <= gamma <= 1
        # self.env_len = env_len

    def get_model(self):
        """Initialize neural network using TensorFlow2 functional API.
        :return: (None)
        """
        # Accept one-hot encoded states as input
        input_ = Input(shape=(2, self.height, self.width), name="input")
        flatten_1 = Flatten()(input_)
        dense_1 = Dense(
            units=self.n_hidden,
            activation="relu",
            kernel_regularizer="l2",
            name="dense_1",
        )(flatten_1)
        dense_2 = Dense(
            units=self.n_hidden,
            activation="relu",
            kernel_regularizer="l2",
            name="dense_2",
        )(dense_1)
        # Output log probabilities using log_softmax activation
        fpm = Dense(
            units=self.action_space, activation="log_softmax", name="forward_policy"
        )(dense_2)
        bpm = Dense(units=self.dim, activation="log_softmax", name="backward_policy")(
            dense_2
        )
        # Z0 is a single learned value
        self.z0 = tf.keras.Variable(0.0, name="z0")
        # Model output will be a list of tensors for both forward and backward
        self.model = Model(input_, [fpm, bpm])
        # # We'll be using the uniform distribution to add
        # # more exploration to our forward policy
        self.unif = tfd.Uniform(
            low=[0] * self.action_space, high=[1] * self.action_space
        )

    def call(self, state):
        return self.model(state)

    def mask_forward_actions(self, lattice):
        """Build boolean mask with zeros over coordinates at the edge of environment.
        :param batch_of_positions: (nd.array) Array of coordinates
        :return: (nd.array) Mask over coordinates at the edge of environment with same
                 shape == batch_of_positions.shape
        """
        num_spins, height, width = lattice.shape
        # Total actions: 2 spins, height * width positions, 4 directions
        actions = np.zeros(num_spins * height * width * 4 + 1, dtype=int)
        actions[num_spins * height * width * 4] = 1
        # Directions are encoded as: 0 - up, 1 - down, 2 - left, 3 - right
        for spin in range(num_spins):
            for h in range(height):
                for w in range(width):
                    if lattice[spin, h, w] == 1:
                        # Calculate wrapped indices
                        up = (h - 1) % height
                        down = (h + 1) % height
                        left = (w - 1) % width
                        right = (w + 1) % width

                        # Position in the actions array
                        base_idx = (spin * height * width + h * width + w) * 4

                        # Set actions based on availability of target position
                        # Up
                        actions[base_idx + 0] = 1 if lattice[spin, up, w] == 0 else 0
                        # Down
                        actions[base_idx + 1] = 1 if lattice[spin, down, w] == 0 else 0
                        # Left
                        actions[base_idx + 2] = 1 if lattice[spin, h, left] == 0 else 0
                        # Right
                        actions[base_idx + 3] = 1 if lattice[spin, h, right] == 0 else 0

        return actions

        ######
        # batch_size = array_state.shape[0]
        # # Check that we're not up against the edge of the environment
        # action_mask = batch_of_positions < (self.env_len - 1)
        # # The "stop" action is last and always allowed, so we append a 1 at the end)
        # stop_column = np.ones((batch_size, 1))
        # return np.append(action_mask, stop_column, axis=1)

    def mask_and_norm_forward_actions(self, lattice, batch_forward_probs):
        """Remove actions that would move outside the environment, and re-normalize
        probabilities so that they sum to one.
        :param batch_positions: (nd.array) Array of coordinates
        :param batch_forward_probs: (nd.array) Array of probabilites over actions
        :return: (nd.array) Masked and re-normalized probabilities
        """
        mask = self.mask_forward_actions(lattice=lattice)
        masked_actions = mask * batch_forward_probs.numpy()
        # Normalize masked probabilities so that they again sum to 1
        normalized_actions = masked_actions / np.sum(
            masked_actions, axis=1, keepdims=True
        )
        return normalized_actions

    def apply_action(self, state, action):
        new_state = np.copy(state)
        spin = (action // (self.width * self.height * 4)) % 2
        height = (action // (self.width * 4)) % self.height
        width = (action // 4) % self.width
        direction = action % 4
        new_state[spin, height, width] = 0
        match direction:
            case 0:
                new_state[spin, (height - 1) % self.height, width] = 1
            case 1:
                new_state[spin, (height + 1) % self.height, width] = 1
            case 2:
                new_state[spin, height, (width - 1) % self.width] = 1
            case 3:
                new_state[spin, height, (width + 1) % self.width] = 1
        return new_state


if __name__ == "__main__":
    initial_state = np.array(
        [
            [[0, 1, 0, 1, 0], [1, 0, 1, 0, 1], [0, 0, 0, 0, 1]],
            [[1, 0, 1, 1, 1], [0, 0, 0, 0, 1], [0, 0, 1, 0, 0]],
        ]
    )
    trajectory = [initial_state]
    path = []
    model = GFNAgent(initial_state=initial_state)
    state = initial_state

    for episode in range(model.max_trajectory_len):
        starting_state = tf.convert_to_tensor(state)
        batched_starting_state = tf.expand_dims(starting_state, axis=0)
        # tensor_starting_state = tf.reshape(tensor_starting_state, shape=(1, 2, 3, 5))
        model_forward_logits = model(batched_starting_state)[0]
        model_forward_probs = tf.math.exp(model_forward_logits)
        normalized_actions = model.mask_and_norm_forward_actions(
            lattice=state, batch_forward_probs=model_forward_probs
        )
        action = tfd.Categorical(probs=normalized_actions).sample()
        action_one_hot = tf.one_hot(action, model.action_space).numpy()
        action_int = np.argmax(action_one_hot)
        path.append(action_one_hot)

        if action_int == model.stop_action or episode == (model.max_trajectory_len - 1):
            visualize_trajectory(trajectory=trajectory)
            # reward = model.env_reward.potential_reward(state)
            break

        state = model.apply_action(state=state, action=action_int)
        trajectory.append(state)

    # print(f"{reward=} after {len(path)} actions")
