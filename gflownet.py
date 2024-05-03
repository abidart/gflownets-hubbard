import numpy as np
from Reward import Reward
from checks import check_valid_state
import os
import sys
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from visualize import visualize_trajectory
import tqdm

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
        height,
        width,
        epochs,
        index_log,
        n_hidden=32,
        name="",
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
        self.height, self.width = height, width
        assert self.width > 1
        assert self.height > 1
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
        self.max_trajectory_len = 10
        self.index_log = index_log

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
        self.logz = tf.keras.Variable(0.0, name="logz")
        # Model output will be a list of tensors for both forward and backward
        self.model = Model(input_, [fpm, bpm])

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
                        if lattice[spin, up, w] == 0:
                            actions[base_idx + 0] = 1
                        # Down
                        if lattice[spin, down, w] == 0:
                            actions[base_idx + 1] = 1
                        # Left
                        if lattice[spin, h, left] == 0:
                            actions[base_idx + 2] = 1
                        # Right
                        if lattice[spin, h, right] == 0:
                            actions[base_idx + 3] = 1

        return actions

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
            # TODO(Alan): Remove assertions once bug in masking is fixed
            case 0:
                assert new_state[spin, (height - 1) % self.height, width] == 0
                new_state[spin, (height - 1) % self.height, width] = 1
            case 1:
                assert new_state[spin, (height + 1) % self.height, width] == 0
                new_state[spin, (height + 1) % self.height, width] = 1
            case 2:
                assert new_state[spin, height, (width - 1) % self.width] == 0
                new_state[spin, height, (width - 1) % self.width] = 1
            case 3:
                assert new_state[spin, height, (width + 1) % self.width] == 0
                new_state[spin, height, (width + 1) % self.width] = 1
        return new_state

    def grad(self, total_P_F, total_P_B, reward):
        """Calculate gradients based on loss function values. Notice the z0 value is
        also considered during training.
        :param batch: (tuple of ndarrays) Output from self.train_gen() (positions, rewards)
        :return: (tuple) (loss, gradients)
        """
        with tf.GradientTape() as tape:
            loss = self.trajectory_balance_loss(
                total_P_F=total_P_F, total_P_B=total_P_B, reward=reward
            )
            grads = tape.gradient(loss, self.trainable_variables + [self.logz])
        return loss, grads

    def trajectory_balance_loss(self, total_P_F, total_P_B, reward):
        return tf.pow(
            self.logz
            + total_P_F
            - tf.clip_by_value(tf.math.log(reward), -20, tf.float32.max)
            - total_P_B,
            2,
        )

    def train(self, initial_states):
        """Run a training loop of `length self.epochs`.
        At the end of each epoch, save weights if loss is better than any previous epoch.
        At the end of training, read in the best weights.
        :param verbose: (bool) Print additional messages while training
        :return: (None) Updated model parameters
        """

        # use datasets to set up batches using array of initial states (check out generate_valid_states)
        dataset = tf.data.Dataset.from_tensor_slices(initial_states)
        dataset = dataset.shuffle(buffer_size=100).batch(5) # can change batch size here

        # uses epochs instead of episodes, maybe we could stick with episodes since reinforcement learning
        for epoch in range(self.epochs):
            print(f"Epoch {epoch + 1}/{self.epochs}")
            epoch_losses = []
            trajectory = []

            # gradient tape, loss, and optimization for each batch
            for batch_initial_states in dataset:
                with tf.GradientTape() as tape:
                    losses = []
                    for initial_state in batch_initial_states:

        # ALL FOLLOWING LOGIC SWALLOWED INTO BATCH IMPLEMENTATION
        # for episode in tqdm.tqdm(range(self.epochs), ncols=40):
        #     # Each episode starts with an "initial state"
        #     trajectory = [] // moved trajectory up for each epoch
        #     path = [] // got rid of path, cat.sample() does same thing

        #     if episode % self.index_log == 0:
        #         trajectory.append(initial_state)

                        state = initial_state
                        state_tensor = self.state_to_tensor(initial_state)
                        # Predict P_F, P_B
                        P_F_logit, P_B_logit = self.model(state_tensor)
                        P_F_probs, P_B_probs = tf.exp(P_F_logit), tf.exp(P_B_logit)
                        total_P_F = 0
                        total_P_B = 0
                        # reward = 0 // goes after next for loop

                        for trajectory_len in range(self.max_trajectory_len):
                # TODO(Alan): understand if we should use logits or probabilities
                # Here P_F is logits, so we want the Categorical to compute the softmax for us
                            normalized_actions_probs = self.mask_and_norm_forward_actions(
                                lattice=state, batch_forward_probs=P_F_probs
                            )
                            cat = tfd.Categorical(probs=normalized_actions_probs)
                            action = cat.sample()

                            log_prob_f = cat.log_prob(action)
                            total_P_F += log_prob_f

                            trajectory.append(state)
                        
                            action_one_hot = tf.one_hot(action, self.action_space).numpy()
                            action_int = np.argmax(action_one_hot)


                            if action_int == self.stop_action or trajectory_len == (
                                self.max_trajectory_len - 1
                            ):
                                if epoch % self.index_log == 0:
                                    visualize_trajectory(
                                        trajectory=trajectory,
                                        filename=f"training_gifs/episode_{epoch}.gif",
                                    )

                            new_state = self.apply_action(state=state, action=action_int) # error often happens here because action chosen was invalid
                            P_F_logit, P_B_logit = self.model(self.state_to_tensor(new_state))
                            log_prob_b = tfd.Categorical(probs=tf.exp(P_B_logit)).log_prob(action) # error often happens here because action 120 is outside of range [1,120)
                            total_P_B += log_prob_b

                            state = new_state

                        reward = tf.cast(self.env_reward.potential_reward(state), dtype=tf.float32)
                        loss = self.trajectory_balance_loss(total_P_F, total_P_B, reward)
                        losses.append(loss)

                    batch_loss = tf.reduce_mean(losses)
                    grads = tape.gradient(batch_loss, self.trainable_variables + [self.logz])
                    self.optimizer.apply_gradients(zip(grads, self.trainable_variables + [self.logz]))

                epoch_losses.append(batch_loss.numpy())

            avg_loss = sum(epoch_losses) / len(epoch_losses)
            print(f"Average Loss for Epoch {epoch + 1}: {avg_loss}")

            ### end of batch training implementation ###

            

                # action_one_hot = tf.one_hot(action, self.action_space).numpy()
                # action_int = np.argmax(action_one_hot)
                # if episode % self.index_log == 0:
                #     path.append(action_one_hot)

                # if action_int == self.stop_action or trajectory_len == (
                #     self.max_trajectory_len - 1
                # ):
                #     if episode % self.index_log == 0:
                #         visualize_trajectory(
                #             trajectory=trajectory,
                #             filename=f"training_gifs/episode_{episode}.gif",
                #         )
                    # TODO(Alan): fix reward
                    # reward = tf.cast(self.env_reward.potential_reward(state), dtype=tf.float32)
                    # # reward = (
                    # #     tf.convert_to_tensor([42], dtype=tf.float32)
                    # #     if state[0][0][0] == 1 and state[1][0][0] == 1
                    # #     else tf.convert_to_tensor([0], dtype=tf.float32)
                    # # )
                    # break

                # new_state = self.apply_action(state=state, action=action_int)

                # if episode % self.index_log == 0:
                #     trajectory.append(new_state)

                # new_state_tensor = self.state_to_tensor(new_state)
                # # Accumulate the P_F sum
                # total_P_F += cat.log_prob(action)

                # We recompute P_F and P_B for new_state
                # P_F_logit, P_B_logit = self.model(new_state_tensor)
                # P_F_probs, P_B_probs = tf.exp(P_F_logit), tf.exp(P_B_logit)
                # Here we accumulate P_B, going backwards from `new_state`. We're also just
                # going to use opposite semantics for the backward policy. I.e., for P_F action
                # `i` just added the face part `i`, for P_B we'll assume action `i` removes
                # face part `i`, this way we can just keep the same indices.

                # total_P_B += tfd.Categorical(probs=P_B_probs).log_prob(action)

                # Continue iterating
                # state = new_state

            # We're done with the trajectory, let's compute its loss. Since the reward can
            # sometimes be zero, instead of log(0) we'll clip the log-reward to -20.
            # loss, grads = self.grad(total_P_F, total_P_B, reward)
            # self.optimizer.apply_gradients(
            #     zip(grads, self.trainable_variables + [self.logz])
            # )
            # if episode % self.index_log == 0:
            #     print(
            #         f"\nEpisode {episode}, loss = {loss.numpy()}, logZ ={self.logz.numpy()}"
            #     )

    def state_to_tensor(self, state):
        starting_state = tf.convert_to_tensor(state)
        batched_starting_state = tf.expand_dims(starting_state, axis=0)
        return batched_starting_state
    
    def generate_valid_states(self, height, width, num_e, num_states):

        assert num_e <= width*height, "Number of elements (num_e) cannot exceed total number of cells (width * height)"

        states = np.zeros((num_states, 2, height, width))

        for i in range(num_states):
            lattice = np.zeros((2, height, width))
            random_up = np.random.choice(height * width, size=num_e, replace=False)
            random_down = np.random.choice(height * width, size=num_e, replace=False)
        
            for ind in random_up:
                lattice[0, ind // width, ind % width] = 1

            for ind in random_down:
                lattice[1, ind // width, ind % width] = 1

            states[i]=lattice
        
        return states

        


if __name__ == "__main__":
    initial_state = np.array(
        [
            [[0, 0, 1, 1, 0], [0, 1, 1, 0, 1], [0, 0, 0, 0, 1]],
            [[0, 0, 1, 1, 1], [0, 0, 1, 0, 1], [0, 0, 1, 0, 0]],
        ]
    )
    # initial_state = np.array(
    #     [
    #         [[1, 0], [1, 1]],
    #         [[1, 1], [1, 0]],
    #     ]
    # )
    height, width = initial_state[0].shape
    model = GFNAgent(height=height, width=width, epochs=51, index_log=10)
    initial_states = model.generate_valid_states(height=model.height, width=model.width, num_e=7, num_states=10)
    model.train(initial_states=initial_states)
