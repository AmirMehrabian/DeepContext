import numpy as np
from config import config_dict, step_dict
from utils import epsilon_greedy, context_builder, create_grid_rbf_centers, model_builder_cnn, model_builder, \
    model_feeder, encode_with_rbf, ReplayBuffer_CNN, ReplayBuffer, model_feeder_no_action
from env_response import env_response
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

dim1_bin = 8
dim2_bin = 8
dim3_bin = 8

gamma = 0.5
buffer_capacity = 5000
batch_size = 2500
learning_interval = 10
epoch = 20

# model = model_builder(3, 3)
# model.summary()

step_list = step_dict['eval_episodes_pisodes']

action_set = config_dict['action_set']
print(action_set)
number_actions = len(action_set)
number_context = 3

input_model_size = number_context
# Epsilon setting
epsilon_init = 0.99
epislon_min = 0
epsilon_decay = 0.025

num_episodes = 100
avg_error = []
avg_rev = []

# replay_input_buffer = np.array([]).reshape(-1,input_model_size)
# replay_output_buffer = np.array([]).reshape(-1,1)

num_features = 3
avg_curve = np.zeros(step_list.shape[1])

model_list = []
buffers = []
A_matrices = dict()
b_vectors = dict()
theta_vectors = dict()
x_vector = dict()
for index_action, action in enumerate(action_set):
    A_matrices[index_action] = np.eye(num_features)
    b_vectors[index_action] = np.zeros(num_features)
    theta_vectors[index_action] = np.zeros(num_features)




eps_zero_count = 0
all_avg_error = 0
for episode_index in range(num_episodes):

    epsilon = max(epislon_min, epsilon_init - (episode_index * epsilon_decay))

    print(f"Episode: {episode_index + 1}, Epsilon: {epsilon}")

    # Randomly selecting the action and first episode
    action_index = np.random.choice(number_actions)
    config_dict['action_index'] = action_index
    config_dict['num_pilot_block'] = action_set[action_index]

    rnd_step_index = np.random.choice(step_list.shape[1])
    print(rnd_step_index)
    config_dict['N_tc'] = step_list[0][rnd_step_index]
    config_dict['snrj'] = step_list[1][rnd_step_index]
    config_dict['snrs'] = step_list[2][rnd_step_index]
    # print(config_dict)

    # Initialize the first context
    total_rev, r_state, p_jam, p_signal = env_response(config_dict)


    avg_vec = []
    agg_err = 0
    agg_rev = 0
    for counter, step_params in enumerate(np.array(step_list).T):

        est_reward_vector = np.zeros(number_actions)
        context = context_builder(r_state, p_jam, p_signal)
        for index_action, action in enumerate(action_set):
             x_vector[index_action] = context
             theta_vectors[index_action] = np.linalg.pinv(A_matrices[index_action])@ b_vectors[index_action]
             est_reward_vector[index_action] = x_vector[index_action]@theta_vectors[index_action]
             est = x_vector[index_action]@theta_vectors[index_action]
             est2 = x_vector[index_action].T @ theta_vectors[index_action]
             #print(est,est.shape)
             #print('est2', est2, est2.shape)


        action_index = epsilon_greedy(epsilon, est_reward_vector)
        config_dict['action_index'] = action_index
        config_dict['num_pilot_block'] = action_set[action_index]
        # print(counter, est_reward_vector,action_index)
        if counter % 20 == 0:
            print(counter, end=', ')
            #print(est_reward_vector, action_index)

        # Observing new env params based on step_params
        config_dict['N_tc'] = step_params[0]
        config_dict['snrj'] = step_params[1]
        config_dict['snrs'] = step_params[2]

        total_rev, r_state, p_jam, p_signal = env_response(config_dict)
        outer_x = np.outer(x_vector[action_index], x_vector[action_index])
        #print(outer_x.shape, outer_x)
        A_matrices[action_index] = A_matrices[action_index] + outer_x
        b_vectors[action_index] = b_vectors[action_index] + total_rev*x_vector[action_index]


        # print(f'c: {counter}, a_i: {action_index}, e_i:{abs(total_rev-est_reward_vector[action_index])}')
        agg_err = agg_err + abs(total_rev - est_reward_vector[action_index])
        agg_rev = agg_rev + total_rev
        avg_vec = np.append(avg_vec, total_rev)
        # adding data to buffer


    avg_error = np.append(avg_error, agg_err / step_list.shape[1])
    avg_rev = np.append(avg_rev, agg_rev / step_list.shape[1])
    print('\n', f'avg_err: {agg_err / step_list.shape[1]}, avg_rev: {agg_rev / step_list.shape[1]} ')
    print("-" * 50)
    if epsilon <= 0.0001:
        avg_curve = avg_curve + avg_vec
        all_avg_error = all_avg_error + agg_err / step_list.shape[1]
        eps_zero_count += 1


print('total_average_error: ', all_avg_error/eps_zero_count)
plt.figure(1)
plt.plot(list(range(num_episodes)), avg_error)
plt.xlabel("Episodes")
plt.ylabel("Average Error")
plt.grid(True)

plt.figure(2)
plt.plot(list(range(step_list.shape[1])), avg_curve / eps_zero_count)
plt.xlabel("Steps")
plt.ylabel("Average Error")
plt.grid(True)

plt.figure(3)
plt.plot(list(range(num_episodes)), avg_rev)
plt.xlabel("Episodes")
plt.ylabel("Average Rev")
plt.grid(True)


print(np.mean(avg_curve / eps_zero_count))
