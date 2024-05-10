import numpy as np
from matplotlib import pyplot as plt

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, Input, Flatten
import tensorflow_probability as tfp
from Reward import Reward
import os
from visualize import visualize_trajectory, draw_lattice

os.environ["TF_USE_LEGACY_KERAS"] = "1"

# INITIAL_LATTICE = np.array(
#     [
#         [[1, 0], [1, 1]],
#         [[1, 1], [1, 0]],
#     ]
# )

INITIAL_LATTICE = np.array(
    [
        [[1, 1, 1], [1, 0, 0], [0, 0, 0]],
        [[1, 1, 1], [1, 0, 0], [0, 0, 0]],
    ]
)

# INITIAL_LATTICE = np.array(
#     [
#         [[0, 0, 1, 1, 0], [0, 1, 1, 0, 1], [0, 0, 0, 0, 1]],
#         [[0, 0, 1, 1, 1], [0, 0, 1, 0, 1], [0, 0, 1, 0, 0]],
#     ]
# )


class GFNAgent(Model):
    """Example Generative Flow Network as described in:
    https://arxiv.org/abs/2106.04399 and
    https://arxiv.org/abs/2201.13259
    """

    tfd = tfp.distributions

    def __init__(
        self,
        epochs=100,
        n_hidden=32,
        lr=0.005,
        max_trajectory_len=10,
    ):
        """Initialize GFlowNet agent.
        :param max_trajectory_len: (int) Max length for each sampled path
        :param n_hidden: (int) Number of nodes in hidden layer of neural network
        :param epochs: (int) Number of epochs to complete during training cycle
        :param lr: (float) Learning rate
        :return: (None) Initialized class object
        """
        super().__init__()
        self.initial_lattice = (
            INITIAL_LATTICE  # all trajectories will start from this lattice
        )
        self.num_spins, self.height, self.width = INITIAL_LATTICE.shape
        self.max_trajectory_len = max_trajectory_len
        self.action_space = (
            2 * self.width * self.height * 4 + 1
        )  # the number of valid actions
        self.stop_action = (
            self.action_space - 1
        )  # the index of the action representing the "exit" action which terminates the path
        self.dim = (
            self.action_space - 1
        )  # dimension of the space representing all possible non-exit actions (used for one-hot encoding of actions)
        self.n_hidden = n_hidden
        self.epochs = epochs

        self.env_reward = Reward(lattice_width=self.width, lattice_height=self.height)
        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=lr, decay_steps=10000, decay_rate=0.8
        )
        self.optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=lr_schedule)

        self.data = {
            "positions": None,
            "actions": None,
            "rewards": None,
        }  # stores generated data for offline training
        self.clear_eval_data()

        ### MODEL
        self.flatten = Flatten()
        self.dense_1 = Dense(
            units=self.n_hidden,
            activation="relu",
            kernel_regularizer="l2",
            name="dense_1",
        )
        self.dense_2 = Dense(
            units=self.n_hidden,
            activation="relu",
            kernel_regularizer="l2",
            name="dense_2",
        )
        self.fpm = Dense(
            units=self.action_space,
            activation="log_softmax",
            name="forward_policy",
        )
        self.bpm = Dense(
            units=self.dim, activation="log_softmax", name="backward_policy"
        )
        self.z0 = tf.Variable(0.0, name="z0")

    def call(self, input_state, bidirectional=False, backwards=False):
        flattened = self.flatten(input_state)
        dense_1_out = self.dense_1(flattened)
        dense_2_out = self.dense_2(dense_1_out)
        if bidirectional:
            out = self.fpm(dense_2_out), self.bpm(dense_2_out)
        else:
            out = self.fpm(dense_2_out) if not backwards else self.bpm(dense_2_out)
        return out

    @staticmethod
    def lattice_to_tensor(lattice):
        """
        Convert input lattice to a tf.Tensor object.
        :param lattice: (np.ndarray) 3D Numpy array representing a lattice
        :return: tf.Tensor object
        """
        starting_state = tf.convert_to_tensor(lattice)
        batched_starting_state = tf.expand_dims(starting_state, axis=0)
        return batched_starting_state

    @staticmethod
    def lattices_to_tensor(lattices):
        """
        Convert array of lattices to a tf.Tensor object for batch processing.
        :param lattices: (np.ndarray) Array of 3D Numpy arrays representing lattices
        :return: tf.Tensor object
        """
        array = np.array(lattices)
        batch_of_lattices = tf.convert_to_tensor(array)
        return batch_of_lattices

    def create_forward_actions_mask(self, batch_of_lattices: list):
        """Build list of boolean masks with zeros for invalid actions and ones for valid actions, for each
        lattice in the input array of lattices.
        :param batch_of_lattices: (nd.array) Array of lattices
        :return: (nd.array) Array of masks corresponding to the lattices in input
        """
        num_spins, height, width = batch_of_lattices[0].shape

        batch_of_actions = []

        for lattice in batch_of_lattices:
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
            batch_of_actions.append(actions)
        return batch_of_actions

    def mask_and_norm_forward_actions(self, batch_of_lattices, batch_forward_probs):
        """Remove invalid actions and re-normalize probabilities so that they sum to one.
        :param batch_positions: (nd.array) Array of lattices
        :param batch_forward_probs: (nd.array) Array of probabilities over actions
        :return: (nd.array) Masked and re-normalized probabilities
        """
        mask = self.create_forward_actions_mask(batch_of_lattices=batch_of_lattices)
        masked_actions = mask * batch_forward_probs.numpy()
        # Normalize masked probabilities so that they again sum to 1
        normalized_actions = masked_actions / np.sum(
            masked_actions, axis=1, keepdims=True
        )
        return normalized_actions

    def apply_action(self, state, action):
        """Generate the state that would result from applying the given action to the given state.
        :param state: (np.ndarray) 3D Numpy array representing the starting state
        :param action: One-hot encoding of the action to be applied
        :return: state resulting from applying the action to the input state
        """
        new_state = np.copy(state)
        spin = (action // (self.width * self.height * 4)) % 2
        height = (action // (self.width * 4)) % self.height
        width = (action // 4) % self.width
        direction = action % 4
        new_state[spin, height, width] = 0
        match direction:
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

    def sample_trajectories(self, batch_size=3):
        """Sample `batch_size` trajectories using the current policy.
        :param batch_size: (int) Number of trajectories to sample
        :return: (tuple of nd.array) (trajectories, one_hot_actions, rewards)
        """
        still_sampling = [True] * batch_size
        positions = [self.initial_lattice] * batch_size
        trajectories = [positions.copy()]
        one_hot_actions = []
        batch_rewards = np.zeros((batch_size,))
        for step in range(self.max_trajectory_len - 1):
            tensor_positions = self.lattices_to_tensor(positions)
            # Use forward policy to get log probabilities over actions, these actions could include invalid actions
            model_fwrd_logits = self.call(tensor_positions)
            model_fwrd_probs = tf.math.exp(model_fwrd_logits)

            # Impossible actions as not valid
            normalized_actions = self.mask_and_norm_forward_actions(
                positions, model_fwrd_probs
            )
            # Select actions randomly, proportionally to input probabilities
            actions = self.tfd.Categorical(probs=normalized_actions).sample()
            actions_one_hot = tf.one_hot(actions, self.action_space).numpy()
            # Update positions based on selected actions
            for i, act_i in enumerate(actions):
                if act_i == (self.action_space - 1) and still_sampling[i]:
                    still_sampling[i] = False
                elif not still_sampling[i]:
                    positions[i] = trajectories[step][i]
                    actions_one_hot[i] = 0
                else:
                    action_int = act_i.numpy()
                    positions[i] = self.apply_action(
                        state=positions[i], action=action_int
                    )
            trajectories.append(positions.copy())
            one_hot_actions.append(actions_one_hot)
        for index in range(batch_size):
            batch_rewards[index] = self.env_reward.potential_reward(positions[index])
        return (
            np.stack(trajectories, axis=1),
            np.stack(one_hot_actions, axis=1),
            batch_rewards,
        )

    def create_backwards_actions_mask(self, lattice):
        """Build boolean mask with zeros over coordinates at the edge of environment.
        :param batch_of_positions: (nd.array) Array of coordinates
        :return: (nd.array) Mask over coordinates at the edge of environment with same
                 shape == batch_of_positions.shape
        """
        num_spins, height, width = lattice.shape

        # Total actions: 2 spins, height * width positions, 4 directions
        actions = np.zeros(num_spins * height * width * 4, dtype=int)
        # Directions are encoded as: 0 - up, 1 - down, 2 - left, 3 - right
        for spin in range(num_spins):
            for h in range(height):
                for w in range(width):
                    if lattice[spin, h, w] == 0:
                        # Calculate wrapped indices
                        up = (h - 1) % height
                        down = (h + 1) % height
                        left = (w - 1) % width
                        right = (w + 1) % width

                        # Position in the actions array
                        base_idx = (spin * height * width + h * width + w) * 4

                        # Set actions based on availability of target position
                        # Up
                        if lattice[spin, up, w] == 1:
                            actions[base_idx + 0] = 1
                        # Down
                        if lattice[spin, down, w] == 1:
                            actions[base_idx + 1] = 1
                        # Left
                        if lattice[spin, h, left] == 1:
                            actions[base_idx + 2] = 1
                        # Right
                        if lattice[spin, h, right] == 1:
                            actions[base_idx + 3] = 1

        return actions

    def mask_and_norm_backward_actions(self, lattice, backward_probs):
        """Remove invalid backward actions and re-normalize probabilities so that they sum to one.
        :param batch_positions: (nd.array) Array of lattices
        :param batch_forward_probs: (nd.array) Array of probabilities over backward actions
        :return: (nd.array) Masked and re-normalized probabilities
        """
        assert isinstance(lattice, np.ndarray)
        mask = self.create_backwards_actions_mask(lattice=lattice)
        masked_actions = mask * backward_probs.numpy()
        # Normalize masked probabilities so that they again sum to 1
        normalized_actions = masked_actions / np.sum(
            masked_actions, axis=1, keepdims=True
        )
        return normalized_actions

    def unapply_action(self, state, action):
        """Recover the state that would lead to the given state if the given action were applied.
        :param state: (np.ndarray) 3D Numpy array representing the final state
        :param action: One-hot encoding of the action that was applied
        :return: state that would result in the given state if the given action were applied
        """
        new_state = np.copy(state)
        spin = (action // (self.width * self.height * 4)) % 2
        height = (action // (self.width * 4)) % self.height
        width = (action // 4) % self.width
        direction = action % 4
        new_state[spin, height, width] = 1
        match direction:
            case 0:
                assert new_state[spin, (height - 1) % self.height, width] == 1
                new_state[spin, (height - 1) % self.height, width] = 0
            case 1:
                assert new_state[spin, (height + 1) % self.height, width] == 1
                new_state[spin, (height + 1) % self.height, width] = 0
            case 2:
                assert new_state[spin, height, (width - 1) % self.width] == 1
                new_state[spin, height, (width - 1) % self.width] = 0
            case 3:
                assert new_state[spin, height, (width + 1) % self.width] == 1
                new_state[spin, height, (width + 1) % self.width] = 0
        return new_state

    def back_sample_trajectory(self, lattice):
        """Follow current backward policy from a position back to the origin.
        Returns them in "forward order" such that origin is first.
        :param lattice: (nd.array) starting lattice for the back trajectory
        :return: (tuple of nd.array) (positions, actions)
        """
        # Attempt to trace a path back to the origin from a given position
        assert isinstance(lattice, np.ndarray)
        positions = [lattice]
        actions = [tf.one_hot(self.stop_action, self.action_space).numpy()]
        cur_lattice = lattice.copy()
        for step in range(self.max_trajectory_len - 1):
            # Use backward policy to get log probabilities over non-termination-actions
            tensor_lattice = self.lattice_to_tensor(lattice=cur_lattice)
            model_back_logits = self.call(tensor_lattice, backwards=True)
            model_back_probs = tf.math.exp(model_back_logits)
            # Don't select impossible actions (like moving out of the environment)
            normalized_actions = self.mask_and_norm_backward_actions(
                cur_lattice, model_back_probs
            )
            # Select most likely action
            action = np.argmax(normalized_actions)
            action_one_hot = tf.one_hot(action, self.action_space).numpy()
            # Update position based on selected action
            try:
                new_lattice = self.unapply_action(state=cur_lattice, action=action)
            except:
                print("error")
            # Convert position to one-hot encoding
            # Stop tracing if at origin
            if np.all(cur_lattice == self.initial_lattice):
                break
            cur_lattice = new_lattice
            positions.append(cur_lattice)
            actions.append(action_one_hot)
        positions.reverse()
        actions.reverse()
        return (np.array(positions), np.array(actions))

    def get_last_position(self, trajectory):
        """Identify the termination coordinates for a trajectory.
        :param trajectory: (nd.array) Array of coordinates/positions
        :return: (nd.array) Last position
        """
        assert len(trajectory.shape) == 4
        return trajectory[-1]

    def sample(self, num_to_sample, evaluate=False):
        """Sample trajectories using the current policy and save to
        `self.data` or `self.eval_data`.
        :param num_to_sample: (int) Number of samples to collect
        :param evaluate: (bool) If False, trajectories are de-duplicated, and output is saved to `self.eval_data`
        :return: (None) Data saved internally
        """
        assert num_to_sample > 0
        trajectories, actions, rewards = self.sample_trajectories(
            batch_size=num_to_sample
        )
        positions = np.stack([self.get_last_position(x) for x in trajectories], axis=0)
        if not evaluate:
            if self.data["positions"] is not None:
                # (batch, len_trajectory, env dimensions)
                self.data["trajectories"] = np.append(
                    self.data["trajectories"], trajectories, axis=0
                )
                # (batch, env dimensions)
                self.data["positions"] = np.append(
                    self.data["positions"], positions, axis=0
                )
                # (batch, len_trajectory-1, action dimensions)
                self.data["actions"] = np.append(self.data["actions"], actions, axis=0)
                # (batch,)
                self.data["rewards"] = np.append(self.data["rewards"], rewards, axis=0)
            else:
                self.data["trajectories"] = trajectories
                self.data["positions"] = positions
                self.data["actions"] = actions
                self.data["rewards"] = rewards
            # Ensure that training data do not contain duplicates
            # (simply to make training faster)
            u_positions, u_indices = np.unique(positions, axis=0, return_index=True)
            self.data["trajectories"] = self.data["trajectories"][u_indices]
            self.data["positions"] = u_positions
            self.data["actions"] = self.data["actions"][u_indices]
            self.data["rewards"] = self.data["rewards"][u_indices]
        else:
            # For evaluating frequencies we have to keep duplicates
            self.eval_data["trajectories"] = trajectories
            self.eval_data["positions"] = positions
            self.eval_data["actions"] = actions
            self.eval_data["rewards"] = rewards

    def clear_eval_data(self):
        """Refresh self.eval_data dictionary."""
        self.eval_data = {"positions": None, "actions": None, "rewards": None}

    def train_gen(self, batch_size=10):
        """Generator object that feeds shuffled samples from `self.data` to training loop.
        :return: (tuple of ndarrays) (positions, rewards)
        """
        data_len = self.data["rewards"].shape[0]
        assert data_len > 0
        iterations = int(data_len // batch_size) + 1
        shuffle = np.random.choice(data_len, size=data_len, replace=False)
        for i in range(iterations):
            # Pick a random batch of training data
            samples = shuffle[i * batch_size : (i + 1) * batch_size]
            sample_positions = self.data["positions"][samples]
            sample_rewards = tf.convert_to_tensor(
                self.data["rewards"][samples], dtype="float32"
            )
            yield (sample_positions, sample_rewards)

    def trajectory_balance_loss(self, batch):
        """Calculate Trajectory Balance Loss function as described in
        https://arxiv.org/abs/2201.13259.
        :param batch: (tuple of nd.arrays) Output from self.train_gen() (positions, rewards)
        :return: (list) Loss function as tensor for each value in batch
        """
        positions, rewards = batch
        losses = []
        for i, position in enumerate(positions):
            reward = rewards[i]
            # Sample a trajectory for the given position using backward policy
            trajectory, back_actions = self.back_sample_trajectory(position)
            # Generate policy predictions for each position in trajectory
            tf_traj = tf.convert_to_tensor(trajectory[:, ...], dtype="float32")
            forward_policy, back_policy = self.call(tf_traj, bidirectional=True)
            # Use "back_actions" to select corresponding forward probabilities
            forward_probs = tf.reduce_sum(
                tf.multiply(forward_policy, back_actions), axis=1
            )
            # Get backward probabilities for the sampled trajectory
            backward_probs = tf.reduce_sum(
                tf.multiply(back_policy[1:, :], back_actions[:-1, : self.dim]), axis=1
            )
            # Add a constant backward probability for transitioning from the termination state
            backward_probs = tf.concat([backward_probs, [0]], axis=0)
            # take log of product of probabilities (i.e. sum of log probabilities)
            sum_forward = tf.reduce_sum(forward_probs)
            sum_backward = tf.reduce_sum(backward_probs)
            # Calculate trajectory balance loss function and add to batch loss
            numerator = self.z0 + sum_forward
            denominator = tf.math.log(reward) + sum_backward
            tb_loss = tf.math.pow(numerator - denominator, 2)
            losses.append(tb_loss)
        return losses

    def grad(self, batch):
        """Calculate gradients based on loss function values. Notice the z0 value is
        also considered during training.
        :param batch: (tuple of ndarrays) Output from self.train_gen() (positions, rewards)
        :return: (tuple) (loss, gradients)
        """
        with tf.GradientTape() as tape:
            loss = self.trajectory_balance_loss(batch)
            grads = tape.gradient(loss, self.trainable_variables + [self.z0])
        return loss, grads

    def train(self, verbose=True):
        """Run a training loop of `length self.epochs`.
        At the end of each epoch, save weights if loss is better than any previous epoch.
        At the end of training, read in the best weights.
        :param verbose: (bool) Print additional messages while training
        :return: (None) Updated model parameters
        """
        if verbose:
            print("Start training...")
        # Keep track of loss during training
        train_loss_results = []
        best_epoch_loss = 10**10
        model_weights_path = "./checkpoints/gfn_checkpoint/gfn.weights.h5"
        for epoch in range(self.epochs):
            epoch_loss_avg = tf.keras.metrics.Mean()
            sampler = self.train_gen()
            # Iterate through shuffled batches of deduplicated training data
            for batch in sampler:
                loss_values, gradients = self.grad(batch)
                self.optimizer.apply_gradients(
                    zip(gradients, self.trainable_variables + [self.z0])
                )
                losses = []
                for sample in loss_values:
                    losses.append(sample.numpy())
                epoch_loss_avg(np.mean(losses))
            # If current loss is better than any previous, save weights
            if epoch_loss_avg.result() < best_epoch_loss:
                self.save_weights(model_weights_path)
                best_epoch_loss = epoch_loss_avg.result()

            train_loss_results.append(epoch_loss_avg.result())
            if verbose and epoch % 10 == 0:
                print(f"Epoch: {epoch} Loss: {epoch_loss_avg.result()}")
        # Load best weights
        self.load_weights(model_weights_path)


if __name__ == "__main__":
    agent = GFNAgent(epochs=100)
    agent.sample(5000)
    agent.train()
    agent.sample(10, evaluate=True)
    for index in range(len(agent.eval_data["positions"])):
        trajectory = agent.eval_data["trajectories"][index]
        reward = agent.eval_data["rewards"][index]
        visualize_trajectory(
            trajectory=trajectory,
            filename=f"eval_trajectories/larger_hidden_layer/{index}_trajectory_{agent.height}_x_{agent.width}.gif",
            reward=reward,
        )
        plt.show()
    print("Done")
