from matplotlib import pyplot as plt
import torch
import numpy as np
import E1 as ee


dim = 2
H = np.identity(dim)
M = np.identity(dim)
C = 0.1 * np.identity(dim)
D = 0.1 * np.identity(dim)
R = np.identity(dim)
sigma = np.array([0.05,0.05])
T = 1
t = 0.5
lqr_monte = ee.LQR(H, M, C, D, R,  t, T)


# Calculate the initial value of the solution matrix S
def mc_S_0(lqr, time_tensor):
    time_grid = ee.get_time_grid(time_tensor,lqr.T, lqr.tau)
    S = lqr.solve_ricatti_ode(time_grid)
    return S[0]


def mc_time_steps(lqr_mc, batch_size, x_np):
    # Convert matrices to PyTorch tensors
    H = torch.tensor(lqr_mc.H)
    M = torch.tensor(lqr_mc.M)
    D_inv = torch.tensor(lqr_mc.D_inv)
    x_t = {}
    a_t = {}
    tss = {}

    for number_of_time_steps in [1, 10, 50, 100, 500, 1000, 5000]:
        # Change the number of time steps in the LQR problem
        lqr_monte.change_time_steps(number_of_time_steps)
        time_grid_mc = ee.get_time_grid(lqr_monte.t, lqr_monte.T, lqr_monte.tss)
        space_tensor_mc = torch.tensor(np.tile(x_np, batch_size).reshape(batch_size, 1, 2))
        x_t[number_of_time_steps] = [space_tensor_mc]
        a_t[number_of_time_steps] = []
        tss[number_of_time_steps] = lqr_monte.tss

        for t_k in time_grid_mc[:-1]:  # Iterates over each time step exclude the last time step
            random = torch.randn(batch_size, 1, 2) * np.sqrt(lqr_monte.tss) * sigma   # Generates random noise
            s_t = mc_S_0(lqr_monte, t_k)    # Compute optimal control
            alpha_no_xtn = -D_inv @ M.T @ s_t
            alpha_control = torch.matmul(alpha_no_xtn, space_tensor_mc.transpose(1, 2)).reshape(batch_size, 2)  
            xtn_coeff = H + (M @ alpha_no_xtn)      # Compute next state   
            Xt_step = lqr_monte.tss * torch.matmul(xtn_coeff, space_tensor_mc.transpose(1, 2)).transpose(1, 2)
            Xt_next_mc = space_tensor_mc + Xt_step + random
            space_tensor_mc = Xt_next_mc
            x_t[number_of_time_steps].append(Xt_next_mc)
            a_t[number_of_time_steps].append(alpha_control)

    return x_t, a_t, tss


def mc_batch_size(lqr_monte, number_of_time_steps, x_np):
    H = torch.tensor(lqr_monte.H)
    M = torch.tensor(lqr_monte.M)
    D_inv = torch.tensor(lqr_monte.D_inv)
    lqr_monte.change_time_steps(number_of_time_steps)
    x_t = {}
    a_t = {}
    tss = {}
    time_grid_mc = ee.get_time_grid(lqr_monte.t, lqr_monte.T, lqr_monte.tss)

    sample_sizes = [10, 50, 100, 500, 1000, 5000, 10000, 50000]
    batch_sizes = [10, 50, 100, 500, 1000, 5000, 10000, 50000, 100000]

     # Initialize dictionaries with empty lists
    for size in sample_sizes + batch_sizes:
        x_t[size] = []
        a_t[size] = []
        tss[size] = lqr_monte.tss

    for batch_size in batch_sizes:
        # Initialize starting state tensor
        space_tensor_mc = torch.tensor(np.tile(x_np, batch_size).reshape(batch_size, 1, 2))  
        x_t[batch_size].append(space_tensor_mc)

        for t_k in time_grid_mc:
            random = torch.randn(batch_size, 1, 2) * np.sqrt(lqr_monte.tss) * sigma
            s_t = mc_S_0(lqr_monte, t_k)
            alpha_no_x_tn = -D_inv @ M.T @ s_t
            alpha_control = torch.matmul(alpha_no_x_tn, space_tensor_mc.transpose(1, 2)).reshape(batch_size, 2)
            xtn_coeff = H + (M @ alpha_no_x_tn)
            Xt_step = lqr_monte.tss * torch.matmul(xtn_coeff, space_tensor_mc.transpose(1, 2)).transpose(1, 2)
            Xt_next_mc = space_tensor_mc + Xt_step + random
            a_t[batch_size].append(alpha_control)
            if t_k != time_grid_mc[-1]:
                x_t[batch_size].append(Xt_next_mc)

    return x_t, a_t, tss


def compute_expected_J(x_t, a_t, tss, lqr_mc, is_n_variable):
    J = {}
    mc_J_mean = {}
    theoretical_v = {}
    theoretical_alpha = {}
    # Set the number of time steps to 5000 for theoretical calculations
    lqr_mc.change_time_steps(5000)

    for key in x_t:
        J[key] =0
        delta_tau = tss[key]

        x_s_prev, a_s_prev = x_t[key][0], a_t[key][0]
        batch_shape = x_s_prev.shape[0]
        for x_s, a_s in zip(x_t[key][1:], a_t[key][1:]):
            x_C_x_prev = torch.matmul(torch.matmul(x_s_prev, torch.tensor(C)),x_s_prev.transpose(1, 2)).reshape(batch_shape,1)
            a_D_a_prev= (torch.matmul(a_s_prev, torch.tensor(D)) * a_s_prev).sum(axis=1).reshape(batch_shape,1)

            x_C_x = torch.matmul(torch.matmul(x_s, torch.tensor(C)),x_s.transpose(1, 2)).reshape(batch_shape,1)
            a_D_a= (torch.matmul(a_s, torch.tensor(D)) * a_s).sum(axis=1).reshape(batch_shape,1)

            J[key] += 0.5*(x_C_x + a_D_a + x_C_x_prev + a_D_a_prev) * delta_tau
            x_s_prev, a_s_prev = x_s.detach().clone(), a_s.detach().clone()

        x_T = x_t[key][-1]
        J[key] += torch.matmul(torch.matmul(x_T, torch.tensor(R)),x_T.transpose(1, 2)).reshape(batch_shape,1)

        mc_J_mean[key] = torch.mean(J[key]).item()

        # Compute theoretical values if requested
        if is_n_variable:
            lqr_mc.change_time_steps(key)
        t_tensor = torch.tensor(lqr_mc.t).reshape(1, 1)
        s_tensor = x_t[key][0]
        theoretical_v[key] = lqr_mc.control_problem_value(t_tensor, s_tensor, sigma).item()
        theoretical_alpha[key] = lqr_mc.markov_control(t_tensor, s_tensor)

    err=[]
    for key in mc_J_mean:
        err.append(np.abs(mc_J_mean[key] - theoretical_v[key]))
    return mc_J_mean, theoretical_v, err


def simulate_once(sim):
    print('Running Simulation for:', sim)
    n_step_sim = {'mc_J_mean': [], 'theoretical_v': [], 'err': []}
    n_batch_sim = {'mc_J_mean': [], 'theoretical_v': [], 'err': []}
    x_t_n_steps, a_t_n_steps, tau_dict_n_steps = mc_time_steps(lqr_monte, batch_size=100000, x_np=sim)
    mc_J_mean, theoretical_v, err = compute_expected_J(x_t_n_steps, a_t_n_steps, tau_dict_n_steps, lqr_monte, True)
    n_step_sim['mc_J_mean'].append(mc_J_mean)
    n_step_sim['theoretical_v'].append(theoretical_v)
    n_step_sim['err'].append(err)

    x_t_batch, a_t_batch, tau_dict_batch = mc_batch_size(lqr_monte, number_of_time_steps=5000, x_np=sim)
    mc_J_mean, theoretical_v, err = compute_expected_J(x_t_batch, a_t_batch, tau_dict_batch, lqr_monte, False)
    n_batch_sim['mc_J_mean'].append(mc_J_mean)
    n_batch_sim['theoretical_v'].append(theoretical_v)
    n_batch_sim['err'].append(err)

    return n_step_sim['err'], [key for key in n_step_sim['theoretical_v']], n_batch_sim['err'], [key for key in n_batch_sim['theoretical_v']]

err_1_list = []
err_2_list = []
key_1_list = []
key_2_list = []
sim_x_values = [np.array((3.0,2.0)),np.array((0.0,1.2)),np.array((1.8,2.2)),np.array((0.9,-2.0)),np.array((-0.5,-1.5)),np.array((3.0,3.0)),np.array((-3.0,-3.0)),np.array((1.0,1.0))]
for sim in sim_x_values:
    err_1, key_1, err_2, key_2 = simulate_once(sim)
    err_1_list.append(err_1)
    err_2_list.append(err_2)
    key_1_list.append(key_1)
    key_2_list.append(key_2)


for simulation,error,sim_x in zip(key_1_list,err_1_list,sim_x_values):
    plt.loglog([k for k in simulation[0]], [x for x in error[0]], label='x({},{})'.format(sim_x[0],sim_x[1]))
plt.ylabel('Log L-1 Error')
plt.xlabel('Log number of steps')
plt.legend()
plt.title('Plot of L-1 Error with respect to Log N for fixed batch size of 100000')
plt.show()

for simulation,error,sim_x in zip(key_2_list,err_2_list,sim_x_values):
    plt.loglog([k for k in simulation[0]], [x for x in error[0]],label='x({},{})'.format(sim_x[0],sim_x[1]))
plt.ylabel('Log L-1 Error')
plt.xlabel('Log Batch Size')
plt.title('Plot of L-1 Error with respect to Log Batch Size for fixed N steps = 5000')
plt.legend()
plt.show()