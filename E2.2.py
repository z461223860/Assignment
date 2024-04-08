import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import E1 as ee
import E2 as eee


# Define LQR parameters
dim = 2
H = np.identity(dim)
M = np.identity(dim)
C = 0.1 * np.identity(dim)
D = 0.1 * np.identity(dim)
R = np.identity(dim)
sigma = np.array([0.05, 0.05])
t_0 = 0
T = 1
batch_size = 256
max_num_epochs = 1000
loss_improvement_limit = 10
sizes = [3, 100, 100, 2]

# Initialize LQR object
lqr = ee.LQR(H, M, C, D, R, t_0, T)
lqr.update_t_0_and_T(t_0=t_0, T=T, n=100)

# Instantiate the model
net_ffn = eee.FFN(sizes = sizes)

# Loss function and optimizer
loss_function = nn.MSELoss()
optimizer = torch.optim.Adam(eee.FFN(sizes = sizes).parameters(), lr=0.001)

# Training loop
losses = []
no_improvement_count = 0
epoch_counter = 0
train = True
print('Starting Training Epoch 0 / {}'.format(max_num_epochs))

t_tensor = torch.tensor(np.random.uniform(low=0, high=T, size=batch_size)).unsqueeze(1)
x_tensor = torch.tensor(np.random.uniform(low=(-3, -3), high=(3, 3), size=(batch_size,1, 2)))
x_reshaped = x_tensor.reshape(-1, 2)
input_tensor = torch.cat((t_tensor, x_reshaped), dim=1)

while train:
    epoch_counter +=1
    if epoch_counter%50 == 0:
        print('Epoch {} / {}'.format(epoch_counter,max_num_epochs))
    y = lqr.markov_control(t_tensor, x_tensor)
    y_pred = net_ffn.forward(input_tensor).to(torch.float64)

    loss = loss_function(y_pred, y)
    losses.append(loss.item())
    net_ffn.zero_grad()
    loss.backward()
    optimizer.step()

    if len(losses) >= 2:
        if losses[-1] > losses[-2]:
           loss_no_improvement_counter += 1
           if loss_no_improvement_counter >= loss_improvement_limit:
            train = False
        else:
           loss_no_improvement_counter = 0
    else:
        loss_no_improvement_counter = 0

    if epoch_counter > max_num_epochs:
        train = False

plt.plot(losses[1:])
plt.xlabel('Iteration')
plt.ylabel('MSE Value')
plt.title('MSE Loss w.r.t iteration number')
plt.show() 