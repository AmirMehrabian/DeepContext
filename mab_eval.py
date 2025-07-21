import numpy as np
from config import config_dict, step_dict
from utils import epsilon_greedy
from env_response import env_response
import matplotlib
matplotlib.use('TkAgg')  # or 'Qt5Agg' or another supported GUI backend
import matplotlib.pyplot as plt



step_list = step_dict['eval_episodes_pisodes']

action_set = config_dict['action_set']
print(action_set)
number_actions = len(action_set)

# Epsilon setting
epsilon =0.15

learning_rate = 0.3

num_episodes = 100
avg_error = []
avg_rev = []


avg_curve = np.zeros(step_list.shape[1])

Q_vector = np.zeros(number_actions)

eps_zero_count = 0
all_avg_error = 0

for episode_index in range(num_episodes):


    Q_vector = np.zeros(number_actions)
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
    for counter, step_params in enumerate(np.array((step_list)).T):

        est_reward_vector = []


        action_index = epsilon_greedy(epsilon, Q_vector)
        config_dict['action_index'] = action_index
        config_dict['num_pilot_block'] = action_set[action_index]
        # print(counter, est_reward_vector,action_index)
        if counter % 20 == 0:
           print(counter, end=', ')
            #print(Q_vector , action_index)

        # Observing new env params based on step_params
        config_dict['N_tc'] = step_params[0]
        config_dict['snrj'] = step_params[1]
        config_dict['snrs'] = step_params[2]

        # taking action in that context
        total_rev, r_state, p_jam, p_signal = env_response(config_dict)

        # new_input_sample = model_feeder(context, index_action, number_actions)
        Q_vector[action_index] = Q_vector[action_index] + learning_rate*(total_rev-Q_vector[action_index])

        # print(f'c: {counter}, a_i: {action_index}, e_i:{abs(total_rev-est_reward_vector[action_index])}')
        agg_err = agg_err + abs(total_rev - Q_vector[action_index])
        agg_rev = agg_rev + total_rev
        avg_vec = np.append(avg_vec, total_rev)
        # adding data to buffer


    avg_error = np.append(avg_error, agg_err / step_list.shape[1])
    avg_rev = np.append(avg_rev, agg_rev / step_list.shape[1])
    print('\n', f'avg_err: {agg_err / step_list.shape[1]}, avg_rev: {agg_rev / step_list.shape[1]} ')
    print("-" * 50)
    if epsilon <= 1:
        avg_curve = avg_curve + avg_vec
        all_avg_error = all_avg_error + agg_err / step_list.shape[1]
        eps_zero_count += 1

print('total_average_rew: ', np.mean(avg_curve / eps_zero_count))

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

plt.show()

