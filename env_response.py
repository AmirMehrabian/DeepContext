from scipy.special import j0  # Bessel function of zeroth order
from scipy.linalg import svd, null_space
from config import config_dict, step_dict
from scipy.stats import nakagami

def nakagami_channel(config, a, b):
  """
  Generates Nakagami-m fading channel coefficients.

  Args:
    config: A dictionary containing configuration parameters.
           It should include 'mnak' for the Nakagami-m parameter.
    a: Lower bound of the random variable.
    b: Upper bound of the random variable.

  Returns:
    Hnak: A complex array of channel coefficients.
  """
  m = config['mnak']

  # We will use numpy to generate random variables from Nakagami-m and Uniform distributions
  z = nakagami.rvs(m, scale=np.sqrt(1.0), size=(a, b)) # generating from Gamma distribution for Nakagami-m
  Q = np.random.uniform(0, 2*np.pi, size=(a,b)) # generating from Uniform distribution

  Hnak = z * np.exp(1j*Q)  # calculating channel coefficients as complex numbers
  return Hnak


import numpy as np

# Helper functions
def db2pow(db):
    """Convert decibels (dB) to power ratio."""
    return 10 ** (db / 10)

def pow2db(power):
    """Convert power ratio to decibels (dB)."""
    return 10 * np.log10(power)

def pskmod(symbols, M):
    """
    Simple PSK modulator.
    symbols: array of integers from 0 to M-1.
    Returns: modulated complex values.
    """
    return np.exp(1j * 2 * np.pi * symbols / M)

def pskdemod(signal, M):
    """
    Simple PSK demodulator.
    signal: array of received complex samples.
    Returns: estimated symbol indices (integers from 0 to M-1).
    """
    # Compute phase in the range [0, 2pi)
    phase = np.angle(signal) % (2 * np.pi)
    # Decide symbol by rounding
    symbols = np.round(phase / (2 * np.pi / M)) % M
    return symbols.astype(int)

def env_response(conf):
    """
    Rewritten MATLAB function in Python.

    Parameters
    ----------
    conf : dict
        Configuration dictionary containing keys:
          - num_pilot_block: int
          - snrs: float (in dB)
          - snrj: float (in dB)
          - N_tc: int
          - Nt1: int
          - N_d1: int
          - K: int
          - Ma: int
          - mnak: parameter for nakagami_channel

    Returns
    -------
    total_rev : float
        Total revenue (reward count)
    r_state : np.ndarray
        Array of state observations (autocorrelation metrics)
    p_jam : float
        Jammer power (in dB)
    p_signal : float
        Signal power (in dB)
    """
    # Unpack configuration
    nb = conf["num_pilot_block"]
    Kj = conf["num_JN"]
    snrs = conf["snrs"]
    snrj = conf["snrj"]
    N_tc = conf["N_tc"]
    Nt1 = conf["Nt1"]
    N_d1 = conf["N_d1"]
    K = conf["K"]
    Ma = conf["Ma"]
    mnak = conf["mnak"]

    sigma_w = 1
    # Number of data symbols in a time frame
    nd = N_d1 - (nb - 1) * Nt1
    # Length of a data block (assumed integer)
    Nt3 = int(nd / nb)
    nd_main = nd
    Nt3_main = Nt3
    total_rev = 0
    i_complex = 1j  # imaginary unit

    # tau
    Td2Tc_g = Nt3 / N_tc
    Ts2Tc_g = Td2Tc_g / Nt3

    vec_n = np.arange(1, N_d1 + 2 * Nt1 + 1)
    vec_rho_g = j0(2 * np.pi * vec_n * 0.423 * Ts2Tc_g)
    RHO_g = np.diag(vec_rho_g)
    RHO2_g = np.diag(np.sqrt(1 - vec_rho_g ** 2))

    if nb == 1:
        nb_vec = np.arange(1, nb + 1)  # [1,...,nb]
    else:
        nb_vec = np.arange(0, nb + 1)    # [0,...,nb]

    r_state = np.zeros(len(nb_vec), dtype=complex)
    cc = 0

    for sec in nb_vec:
        cc += 1
        if sec == 0:
            nd = N_d1
            Nt3 = nd  # when sec==0, use the full N_d1
        M_order = 4

        ampj = np.sqrt(db2pow(snrj) * sigma_w)
        amps = np.sqrt(db2pow(snrs) * sigma_w)

        # Noise matrices (complex Gaussian)
        W_t1 = np.sqrt(sigma_w / 2) * (np.random.randn(K, Nt1) + i_complex * np.random.randn(K, Nt1))
        W_t3 = np.sqrt(sigma_w / 2) * (np.random.randn(K, Nt3) + i_complex * np.random.randn(K, Nt3))
        W_p2 = np.sqrt(sigma_w / 2) * (np.random.randn(K, Nt1) + i_complex * np.random.randn(K, Nt1))

        # Generate random symbols (as integers) and modulate them using PSK
        syms_t1 = np.random.randint(0, M_order, size=Nt1)
        syms_t3 = np.random.randint(0, M_order, size=Nt3)
        s_t1 = pskmod(syms_t1, M_order).reshape(1,Nt1)
        s_t3 = pskmod(syms_t3, M_order).reshape(1,Nt3)

        # print("s_t1 shape:", s_t1.shape)
        # print("s_t3 shape:", s_t3.shape)

        # Channel realizations (assumed provided by nakagami_channel function)
        h_chan = nakagami_channel(conf, K, 1)

        j_t1 = pskmod(np.random.randint(0, M_order, size=Nt1), M_order).reshape(1,Nt1)
        j_t3 = pskmod(np.random.randint(0, M_order, size=Nt3), M_order).reshape(1,Nt3)
        j_p2 = pskmod(np.random.randint(0, M_order, size=Nt1), M_order).reshape(1,Nt1)

        # print("j_t1 shape:", j_t1.shape)
        # print("j_t3 shape:", j_t3.shape)
        # print("j_p2 shape:", j_p2.shape)
        # print("h_chan shape", h_chan.shape)

        g_chan_p = nakagami_channel(conf, K, 1)
        # Replicate g_chan_p horizontally: equivalent to repmat(g_chan_p, 1, 2*Nt1+Nt3)
        G_mat = np.tile(g_chan_p, (1, 2 * Nt1 + Nt3))
        # print("G_mat shape:", G_mat.shape, '\n', G_mat[:,1:3])


        X_mat = nakagami_channel(conf, K, 2 * Nt1 + Nt3)

        Gn = G_mat @ RHO_g[:2 * Nt1 + Nt3, :2 * Nt1 + Nt3] + X_mat @ RHO2_g[:2 * Nt1 + Nt3, :2 * Nt1 + Nt3]
        Gn_p = Gn[:, :Nt1]
        Gn_d = Gn[:, Nt1:Nt1 + Nt3]
        Gn_p2 = Gn[:, Nt1 + Nt3:2 * Nt1 + Nt3]

        # print(np.diag(j_t1).shape,np.diag(j_t1))

        # Received signals
        Y_t1 = (amps * h_chan @ s_t1) + (ampj * Gn_p @ np.diag(j_t1.reshape(-1))) + W_t1  # Pilot 1
        Y_t3 = (amps * h_chan @ s_t3) + (ampj * Gn_d @ np.diag(j_t3.reshape(-1))) + W_t3     # Data
        Y_p2 = (amps * h_chan @ s_t1) + (ampj * Gn_p2 @ np.diag(j_p2.reshape(-1))) + W_p2      # Next pilot

        # Control Channel
        Hc_t1 = nakagami_channel(conf, Ma, K)
        Wc_t1 = np.sqrt(sigma_w / 2) * (np.random.randn(Ma, Nt1) + i_complex * np.random.randn(Ma, Nt1))
        Wc_t3 = np.sqrt(sigma_w / 2) * (np.random.randn(Ma, Nt3) + i_complex * np.random.randn(Ma, Nt3))
        Wc_p2 = np.sqrt(sigma_w / 2) * (np.random.randn(Ma, Nt1) + i_complex * np.random.randn(Ma, Nt1))
        Yc_t1 = Hc_t1 @ Y_t1 + Wc_t1
        Yc_t3 = Hc_t1 @ Y_t3 + Wc_t3
        Yc_p2 = Hc_t1 @ Y_p2 + Wc_p2

        # Least squares estimation of channel matrix
        #Hc_t1_H = Hc_t1.conj().T
        #print(Hc_t1_H @ Hc_t1)
        #LS_H = LS_H = np.linalg.inv(Hc_t1_H @ Hc_t1) @ Hc_t1_H
        LS_H = np.linalg.pinv(Hc_t1)
        Y_t1_est = LS_H @ Yc_t1
        Y_t3_est = LS_H @ Yc_t3
        Y_p2_est = LS_H @ Yc_p2

        # Compute null space for s_t1 (treat s_t1 as a 1 x Nt1 row vector)
        #Sperp = null_space(s_t1[np.newaxis, :])

        Sperp = null_space(s_t1)
        # print(s_t1.shape, Sperp.shape)

        # Estimated jammer channel (projecting Y_t1_est onto the null space)
        est_gchan = Y_t1_est @ Sperp
        est_gchan_p2 = Y_p2_est @ Sperp

        # Calculate jammer power (using Frobenius norm)
        Pr_jam = np.linalg.norm(est_gchan, 'fro') ** 2 / (np.prod(est_gchan.shape))
        p_jam = pow2db(Pr_jam)

        # SVD to extract dominant components of estimated jammer channel
        U, _, _ = svd(est_gchan)
        est_gchan_1 = U[:, 0]
        U_p2, _, _ = svd(est_gchan_p2)
        est_gchan_p2 = U_p2[:, 0]

        if sec == 0:
            r_state[cc-1] = np.vdot(est_gchan_1, est_gchan_p2) # Adjusted index to be 0-based
            nd = nd_main
            Nt3 = Nt3_main
            continue
        r_state[cc-1] = np.vdot(est_gchan_1, est_gchan_p2) # Adjusted index to be 0-based

        # Compute null space of the row vector of est_gchan_1
       # perp_gchan = null_space(est_gchan_1[:, np.newaxis]).T # Reshaped est_gchan_1 to a column vector
        perp_gchan = np.conj(U[:,Kj:].T)
        jam_less = perp_gchan @ Y_t1_est
        p_signal = pow2db(np.linalg.norm(jam_less, 'fro') ** 2 / (np.prod(jam_less.shape)))

        #f_chan = (jam_less @ s_t1[:, np.newaxis] / (np.linalg.norm(s_t1) ** 2)) / amps
        f_chan = (jam_less @np.conj(s_t1.T) / (np.linalg.norm(s_t1) ** 2)) / amps

        hat_st3 = (f_chan.conj().T / (np.linalg.norm(f_chan) ** 2)) @ (perp_gchan @ Y_t3_est)

        hat_symt3 = pskdemod(hat_st3, M_order)

        # Count correctly received symbols:
        errors = np.sum(hat_symt3 != syms_t3)
        R = Nt3 - errors
        total_rev += R

    # Assuming p_jam and p_signal are calculated within the loop and we want the last values.
    # The MATLAB code calculates them inside the loop but also returns them as outputs,
    # implying the final values after the loop are the intended return values.
    # Let's keep the last calculated values.

    return total_rev, r_state, p_jam, p_signal

#
#
# vec_snrs = np.arange(-10,40,4)
# iter = 200
# Pse_vec = []
# corr_vec = []
# corr_vec_nb = []
# p_jam_vec = []
# p_signal_vec = []
# nbb = config_dict['num_pilot_block']
# nd = config_dict['N_d1']  - ( config_dict['num_pilot_block']-1)*config_dict['Nt1'];
# print('nd = ', nd)
#
# for new_snrs in vec_snrs:
#     print(new_snrs, end=', ')
#     config_dict['snrs'] = new_snrs
#     agg_error = 0
#     corr_agg = np.zeros(config_dict['num_pilot_block']+1)
#     p_jam_agg = 0
#     p_signal_agg = 0
#
#     for _ in range(iter):
#         total_rev, r_state, p_jam, p_signal = env_response(config_dict)
#         error = 1 - total_rev/nd
#         agg_error = agg_error +error
#         corr_agg = corr_agg + np.degrees(np.acos(np.abs(r_state)))/90
#         p_jam_agg = p_jam_agg + p_jam
#         p_signal_agg = p_signal_agg + p_signal
#
#     Pse_vec = np.append(Pse_vec, agg_error/iter)
#     corr_vec = np.append(corr_vec, corr_agg[0]/iter)
#     corr_vec_nb = np.append(corr_vec_nb, np.sum(corr_agg[1:])/(config_dict['num_pilot_block']*iter))
#     p_jam_vec = np.append(p_jam_vec, p_jam_agg/iter)
#     p_signal_vec = np.append(p_signal_vec, p_signal_agg/iter)
#
#
# print(config_dict)
#
#
# plt.semilogy(vec_snrs, Pse_vec)
# plt.xlabel("SNR_s")
# plt.ylabel("Average Error")
# plt.grid(True, which='both')
# plt.show()
#
#
# plt.plot(vec_snrs, corr_vec)
# plt.plot(vec_snrs, corr_vec_nb)
# plt.xlabel("SNR_s")
# plt.ylabel("Mean_Corr_vec")
# plt.grid(True)
# plt.legend(['nb=1', f'nb={nbb}'])
# plt.show()
#
#
#
# plt.plot(vec_snrs, p_signal_vec)
# plt.xlabel("SNR_s")
# plt.ylabel("Mean_SNR_S")
# plt.grid(True)
# plt.show()
#
# plt.plot(vec_snrs, p_jam_vec)
# plt.xlabel("SNR_s")
# plt.ylabel("Mean_SNR_J")
# plt.grid(True)
# plt.show()