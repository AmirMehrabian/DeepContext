from tensorflow.keras import layers, models
import numpy as np
from config import config_dict, step_dict
from utils import epsilon_greedy, context_builder, create_grid_rbf_centers, model_builder_cnn, model_builder, \
    model_feeder, encode_with_rbf, ReplayBuffer_CNN, ReplayBuffer, model_feeder_no_action
from env_response import env_response
import matplotlib.pyplot as plt

dim1_bin = 8
dim2_bin = 8
dim3_bin = 8

gamma = 0.5

centers = create_grid_rbf_centers(n1=dim1_bin, n2=dim2_bin, n3=dim3_bin)
buffer_capacity = 5000
batch_size = 3000
learning_interval = 10
epoch = 30

# model = model_builder(3, 3)
# model.summary()

step_list = step_dict['eval_episodes_pisodes']

action_set = config_dict['action_set']
print(action_set)
number_actions = len(action_set)
number_centers = centers.shape[0]
number_context = number_centers

input_model_size = number_context
# Epsilon setting
epsilon_init = 0.99
epislon_min = 0
epsilon_decay = 0.025

num_episodes = 60
avg_error = []
avg_rev = []

# replay_input_buffer = np.array([]).reshape(-1,input_model_size)
# replay_output_buffer = np.array([]).reshape(-1,1)

model = model_builder(number_context, 0, output_size=number_actions)
model.summary()

buffer = ReplayBuffer(capacity=buffer_capacity, input_model_size=input_model_size, output_model_size=number_actions)
avg_curve = np.zeros(step_list.shape[1])
eps_zero_count = 0
for episode_index in range(num_episodes):

    epsilon = max(epislon_min, epsilon_init - (episode_index * epsilon_decay))
    print(f"Episode: {episode_index + 1} , Epsilon: {epsilon}")

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
    context_uncoded = context_builder(r_state, p_jam / 40, p_signal / 40)
    context = encode_with_rbf(context_uncoded.reshape(1, -1), centers, gamma)

    agg_err = 0
    agg_rev = 0
    avg_vec = []

    for counter, step_params in enumerate(np.array((step_list)).T):

        est_reward_vector = []

        model_input = model_feeder_no_action(context)
        model_output = model.predict(model_input, verbose=False)
        est_reward_vector = model_output.reshape(-1)

        action_index = epsilon_greedy(epsilon, est_reward_vector)

        config_dict['action_index'] = action_index
        config_dict['num_pilot_block'] = action_set[action_index]
        # print(counter, est_reward_vector,action_index)

        # print(counter, end=', '
        # Observing new env params based on step_params
        config_dict['N_tc'] = step_params[0]
        config_dict['snrj'] = step_params[1]
        config_dict['snrs'] = step_params[2]

        # taking action in that context
        total_rev, r_state, p_jam, p_signal = env_response(config_dict)

        # new_input_sample = model_feeder(context, index_action, number_actions)
        new_input_sample = model_feeder_no_action(context)

        # print(f'c: {counter}, a_i: {action_index}, e_i:{abs(total_rev-est_reward_vector[action_index])}')
        est_rew = est_reward_vector[action_index]
        agg_err = agg_err + abs(total_rev - est_reward_vector[action_index])
        agg_rev = agg_rev + total_rev
        avg_vec = np.append(avg_vec, total_rev)
        # new_output_sample = np.zeros(number_actions)
        new_output_sample = est_reward_vector
        new_output_sample[action_index] = total_rev
        # adding data to buffer

        # print(f'act_ind: {action_index},\n q_vetor:  {est_reward_vector},\n new_output: {new_output_sample} \n')
        buffer.add_to_buffer(new_input_sample, new_output_sample.reshape(-1, number_actions))

        # observing new context
        context_uncoded = context_builder(r_state, p_jam / 40, p_signal / 40)
        context = encode_with_rbf(context_uncoded.reshape(1, -1), centers, gamma)

        if counter % learning_interval == 0 and counter > 0:
            batch_input, batch_output = buffer.sample_from_buffer(batch_size)
            model.fit(batch_input, batch_output, epochs=epoch, verbose=False)

        print('E', episode_index, '- s', counter, ',\n ')
        print('step_params: ', step_params, '- actions_ind: ', action_index, '\n Q_vec: ', est_reward_vector)
        print('\n curr_reward:', total_rev, '- Error: ',   total_rev - est_rew)
        print('-' * 50)

    if epsilon <= 0.0001:
        avg_curve = avg_curve + avg_vec
        eps_zero_count += 1

    avg_error = np.append(avg_error, agg_err / step_list.shape[1])
    avg_rev = np.append(avg_rev, agg_rev / step_list.shape[1])
    print('\n', f'avg_err: {agg_err / step_list.shape[1]}, avg_rev: {agg_rev / step_list.shape[1]} ')
    print("-" * 50)

plt.figure(21)
plt.plot(list(range(num_episodes)), avg_error)
plt.xlabel("Episodes")
plt.ylabel("Average Error")
plt.grid(True)
plt.show()

plt.figure(22)
plt.plot(list(range(num_episodes)), avg_rev)
plt.xlabel("Episodes")
plt.ylabel("Average Rev")
plt.grid(True)
plt.show()

plt.figure(23)
plt.plot(list(range(step_list.shape[1])), avg_curve/eps_zero_count)
plt.xlabel("Steps")
plt.ylabel("Average Error")
plt.grid(True)
plt.show()

print(np.mean(avg_curve/eps_zero_count))