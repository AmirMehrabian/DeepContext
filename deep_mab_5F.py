from tensorflow.keras import layers, models
import numpy as np
from config import config_dict, step_dict
from utils import epsilon_greedy, context_builder, create_grid_rbf_centers, model_builder_cnn, model_builder, \
    model_feeder, encode_with_rbf, ReplayBuffer_CNN, ReplayBuffer, model_feeder_no_action, context_builder_5features
from env_response import env_response
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
number_context =5

input_model_size = number_context
# Epsilon setting
epsilon_init = 0.99
epislon_min = 0
epsilon_decay = 0.03

num_episodes = 45
avg_error = []
avg_rev = []

# replay_input_buffer = np.array([]).reshape(-1,input_model_size)
# replay_output_buffer = np.array([]).reshape(-1,1)


avg_curve = np.zeros(step_list.shape[1])

model_list = []
buffers = []
for index_action, action in enumerate(action_set):
    model_list.append(model_builder(number_context, 0))
    model_list[index_action].summary()
    buffers.append(ReplayBuffer(capacity=buffer_capacity, input_model_size=input_model_size))

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
    context = context_builder_5features(r_state, p_jam, p_signal)

    avg_vec = []
    agg_err = 0
    agg_rev = 0
    for counter, step_params in enumerate(np.array(step_list).T):

        est_reward_vector = []

        for index_action, action in enumerate(action_set):
            model = model_list[index_action]
            model_input = model_feeder_no_action(context)
            model_output = model.predict(model_input, verbose=False)
            est_reward_vector = np.append(est_reward_vector, model_output.reshape(-1))

        action_index = epsilon_greedy(epsilon, est_reward_vector)
        config_dict['action_index'] = action_index
        config_dict['num_pilot_block'] = action_set[action_index]
        # print(counter, est_reward_vector,action_index)
        if counter % 20 == 0:
            print(counter, end=', ')

        # Observing new env params based on step_params
        config_dict['N_tc'] = step_params[0]
        config_dict['snrj'] = step_params[1]
        config_dict['snrs'] = step_params[2]

        # taking action in that context
        total_rev, r_state, p_jam, p_signal = env_response(config_dict)

        # new_input_sample = model_feeder(context, index_action, number_actions)
        new_input_sample = model_feeder_no_action(context)
        model = model_list[action_index]
        buffer = buffers[action_index]

        # print(f'c: {counter}, a_i: {action_index}, e_i:{abs(total_rev-est_reward_vector[action_index])}')
        agg_err = agg_err + abs(total_rev - est_reward_vector[action_index])
        agg_rev = agg_rev + total_rev
        avg_vec = np.append(avg_vec, total_rev)
        # adding data to buffer
        buffer.add_to_buffer(new_input_sample, total_rev.reshape(-1, 1))

        # observing new context
        context = context_builder_5features(r_state, p_jam, p_signal)

        if counter % learning_interval == 0 and counter > 0:
            batch_input, batch_output = buffer.sample_from_buffer(batch_size)
            model.fit(batch_input, batch_output, epochs=epoch, verbose=False)

    avg_error = np.append(avg_error, agg_err / step_list.shape[1])
    avg_rev = np.append(avg_rev, agg_rev / step_list.shape[1])
    print('\n', f'avg_err: {agg_err / step_list.shape[1]}, avg_rev: {agg_rev / step_list.shape[1]} ')
    print("-" * 50)
    if epsilon <= 0.0001:
        avg_curve = avg_curve + avg_vec
        all_avg_error = all_avg_error + agg_err / step_list.shape[1]
        eps_zero_count += 1


print('total_average_error: ', all_avg_error/eps_zero_count)

print(np.mean(avg_curve / eps_zero_count))

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

plt.show()
