# Example configuration dictionary (values to be set as needed)

import numpy as np
config_dict = {
    "action_index":1,
    "num_pilot_block": 4,
    "num_JN":1,
    "snrs": 10,    # in dB
    "snrj": 20,     # in dB
    "N_tc": 1000,
    "Nt1": 20,
    "N_d1": 500,
    "K": 4,
    "Ma": 64,
    "mnak": 2.0 , # example parameter for the Nakagami channel
    "action_set": np.array([1, 2, 5])
}

# Example usage:
# Assuming you have defined utils.nakagami_channel elsewhere...
# total_rev, r_state, p_jam, p_signal = env_response(config_dict)
print(config_dict)


step_dict = {}
# Frames
step_dict['env_params'] = np.array([
    [1000, 20, 15],
    [1000, 20, 10],
    [1000, 20, 5],
    [1000, 40, 15],
    [1000, 40, 10],
    [1000, 40, 5],
    [3000, 20, 15],
    [3000, 20, 10],
    [3000, 20, 5],
    [3000, 40, 15],
    [3000, 40, 10],
    [3000, 40, 5],
    [5000, 20, 15],
    [5000, 20, 10],
    [5000, 20, 5],
    [5000, 40, 15],
    [5000, 40, 10],
    [5000, 40, 5],
    [10000, 20, 15],
    [10000, 20, 10],
    [10000, 20, 5],
    [10000, 40, 15],
    [10000, 40, 10],
    [10000, 40, 5],
]).T


step_dict['repeats'] = 50
part_repeat = 20
step_dict['N_tc_frame'] = np.concatenate([
    5000 * np.ones(part_repeat),
    3000 * np.ones(part_repeat),
    1000 * np.ones(part_repeat),
    5000 * np.ones(part_repeat),
    3000 * np.ones(part_repeat)
])
step_dict['snr_j_frame'] = np.concatenate([
    20 * np.ones(part_repeat),
    40 * np.ones(part_repeat),
    40 * np.ones(part_repeat),
    40 * np.ones(part_repeat),
    40 * np.ones(part_repeat)
])
step_dict['snr_s_frame'] = np.concatenate([
    10 * np.ones(part_repeat),
    5 * np.ones(part_repeat),
    20 * np.ones(part_repeat),
    20 * np.ones(part_repeat),
    20 * np.ones(part_repeat)
])

step_dict['eval_episodes_pisodes'] = np.vstack([step_dict['N_tc_frame'], step_dict['snr_j_frame'], step_dict['snr_s_frame']])



#print(config_dict)
