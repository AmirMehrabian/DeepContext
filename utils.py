from tensorflow.keras import layers, models
import numpy as np
import tensorflow as tf


def model_builder(context_dim, num_actions, output_size=1):
    input_dim = context_dim + num_actions  # context + one-hot action

    model = models.Sequential([
        layers.Input(shape=(input_dim,)),
        layers.Dense(128, activation='relu'),
        layers.Dense(64, activation='relu'),
        layers.Dense(64, activation='relu'),
        layers.Dense(64, activation='relu'),
        layers.Dense(output_size)  # Output: predicted reward
    ])

    # Compile model with MSE loss and Adam optimizer
    model.compile(optimizer='adam', loss='mse')

    return model


def context_builder(r_state, p_jam, p_signal):
    c0 = np.degrees(np.acos(np.abs(r_state))) / 90
    return np.array([c0[0], p_jam, p_signal])


def model_feeder(context, action_index, number_actions):
    one_hot_action = tf.one_hot(action_index, number_actions)
    tf_context = tf.convert_to_tensor(context, dtype=tf.float32)
    tf_input = tf.concat([tf_context, one_hot_action], axis=0)
    return tf_input.numpy().reshape(1, -1)


def model_feeder_no_action(context):
    tf_context = tf.convert_to_tensor(context, dtype=tf.float32)

    return tf_context.numpy().reshape(1, -1)


def epsilon_greedy(epsilon, q_values):
    """
    epsilon: float in [0, 1] — probability of exploration
    q_values: array of estimated rewards for each action

    Returns: selected action (int)
    """
    if np.random.rand() < epsilon:
        # Explore: choose random action
        return np.random.randint(len(q_values))
    else:
        # Exploit: choose best action
        return np.argmax(q_values)


#
# # Test of epsilon greedy
# epsilon = 0.01
# q_values = [0.1, 0.5, 0.3, 0.4]  # dummy reward estimates for 3 actions
#
# explore_count = 0
# exploit_count = 0
# action_counts = np.zeros(len(q_values), dtype=int)
# iter = int(1e5)
# for _ in range(iter):
#     action = epsilon_greedy(epsilon, q_values)
#     action_counts[action] += 1
#
#     # Check if it was an exploration
#     if q_values[action] != max(q_values):
#         explore_count += 1
#     else:
#         exploit_count += 1

# print("Total Explores (random):", explore_count/iter,'- epsilon: ', (explore_count/iter)/(1-1/len(q_values)))
# print("Total Exploits (best action):", exploit_count/iter)
# print("Action counts:", action_counts)

class ReplayBuffer:

    def __init__(self, capacity, input_model_size, output_model_size=1):
        self.capacity = capacity
        self.replay_input_buffer = np.array([]).reshape(-1, input_model_size)
        self.replay_output_buffer = np.array([]).reshape(-1, output_model_size)

    def add_to_buffer(self, input_sample, output_sample):
        if self.replay_input_buffer.shape[0] < self.capacity:
            self.replay_input_buffer = np.concatenate([self.replay_input_buffer, input_sample], axis=0)
            self.replay_output_buffer = np.concatenate([self.replay_output_buffer, output_sample], axis=0)

        elif self.replay_input_buffer.shape[0] >= self.capacity:
            self.replay_input_buffer = np.concatenate([self.replay_input_buffer[1:], input_sample], axis=0)
            self.replay_output_buffer = np.concatenate([self.replay_output_buffer[1:], output_sample], axis=0)

    def sample_from_buffer(self, batch_size):
        if batch_size > self.replay_input_buffer.shape[0]:
            batch_size = self.replay_input_buffer.shape[0]

        indices = np.random.choice(self.replay_input_buffer.shape[0], size=batch_size, replace=False)
        return self.replay_input_buffer[indices], self.replay_output_buffer[indices]


def create_grid_rbf_centers(n1=5, n2=5, n3=5):
    """
    Creates a grid of RBF centers for 3D input space.

    - n1: number of grid points along dim 0 (range 0 to 1)
    - n2, n3: number of grid points along dim 1 & 2 (range -1 to 1)

    Returns:
        centers: np.ndarray of shape (n1*n2*n3, 3)
    """
    x1 = np.linspace(0.0, 1.0, n1)  # for dim 0
    x2 = np.linspace(-0.2, 1.0, n2)  # for dim 1
    x3 = np.linspace(-0.2, 1.0, n3)  # for dim 2

    # Create a meshgrid of shape (n1, n2, n3)
    grid_x1, grid_x2, grid_x3 = np.meshgrid(x1, x2, x3, indexing='ij')

    # Flatten into shape (n1*n2*n3, 3)
    centers = np.stack([grid_x1, grid_x2, grid_x3], axis=-1).reshape(-1, 3)
    return centers


from sklearn.metrics.pairwise import rbf_kernel


# Define some example fixed RBF centers

# Fixed gamma value (controls kernel width)


# Function to encode 3D context using RBF kernel
def encode_with_rbf(context_batch, centers, gamma=0.5):
    """
    Parameters:
        context_batch: np.ndarray of shape (batch_size, 3)
        centers: np.ndarray of shape (num_centers, 3)
        gamma: float, RBF gamma parameter

    Returns:
        np.ndarray of shape (batch_size, num_centers): RBF features
    """
    return rbf_kernel(context_batch, centers, gamma=gamma)


class ReplayBuffer_CNN:

    def __init__(self, capacity, input_model_size, output_model_size=1):
        self.capacity = capacity
        self.replay_input_buffer = np.empty((0, *input_model_size))
        self.replay_output_buffer = np.array([]).reshape(-1, output_model_size)

    def add_to_buffer(self, input_sample, output_sample):
        if self.replay_input_buffer.shape[0] < self.capacity:
            self.replay_input_buffer = np.concatenate([self.replay_input_buffer, input_sample], axis=0)
            self.replay_output_buffer = np.concatenate([self.replay_output_buffer, output_sample], axis=0)


        elif self.replay_input_buffer.shape[0] >= self.capacity:
            self.replay_input_buffer = np.concatenate([self.replay_input_buffer[1:], input_sample], axis=0)
            self.replay_output_buffer = np.concatenate([self.replay_output_buffer[1:], output_sample], axis=0)

    def sample_from_buffer(self, batch_size):
        if batch_size > self.replay_input_buffer.shape[0]:
            batch_size = self.replay_input_buffer.shape[0]

        indices = np.random.choice(self.replay_input_buffer.shape[0], size=batch_size, replace=False)
        return self.replay_input_buffer[indices], self.replay_output_buffer[indices]


def model_builder_cnn(input_dim, output_size=1):
    model = models.Sequential([
        layers.Input(shape=input_dim),
        #  layers.Dense(128, activation='relu'),
        # layers.Dense(64, activation='relu'),

        layers.Conv3D(32, (5, 5, 5), activation='relu', padding='same'),
        layers.MaxPooling3D((2, 2, 2)),
        layers.Conv3D(32, (3, 3, 3), activation='relu', padding='same'),
        layers.GlobalAveragePooling3D(),
        layers.Dense(output_size)  # Output: predicted reward
    ])

    # Compile model with MSE loss and Adam optimizer
    model.compile(optimizer='adam', loss='mse')

    return model
