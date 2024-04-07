import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import E1 as ee


class DGM_Layer(nn.Module):
    def __init__(self, dim_x, dim_S, activation='Tanh'):
        super(DGM_Layer, self).__init__()
        self.activation = self._get_activation(activation)
        self.gate_Z = self.layer(dim_x+dim_S, dim_S)
        self.gate_G = self.layer(dim_x+dim_S, dim_S)
        self.gate_R = self.layer(dim_x+dim_S, dim_S)
        self.gate_H = self.layer(dim_x+dim_S, dim_S)


    def forward(self, x, S):
        x_S = torch.cat([x,S],1)
        Z = self.gate_Z(x_S)
        G = self.gate_G(x_S)
        R = self.gate_R(x_S)
        input_gate_H = torch.cat([x, S*R],1)
        H = self.gate_H(input_gate_H)
        output = ((1-G))*H + Z*S
        return output


    def layer(self, n_in, n_out):
        layer = nn.Sequential(nn.Linear(n_in, n_out),self.activation)
        return layer
    

    def _get_activation(self, activation):
        if activation.lower() == 'relu':
            return nn.ReLU()
        elif activation.lower() == 'tanh':
            return nn.Tanh()
        elif activation.lower() == 'sigmoid':
            return nn.Sigmoid()
        elif activation.lower() == 'logsigmoid':
            return nn.LogSigmoid()
        else:
            raise ValueError("Unknown activation function: {}".format(activation))


class Net_DGM(nn.Module):
    def __init__(self, dim_x, dim_S, activation='Tanh'):
        super(Net_DGM, self).__init__()
        self.dim = dim_x
        if activation == 'ReLU':
            self.activation = nn.ReLU()
        elif activation == 'Tanh':
            self.activation = nn.Tanh()
        elif activation == 'Sigmoid':
            self.activation = nn.Sigmoid()
        elif activation == 'LogSigmoid':
            self.activation = nn.LogSigmoid()
        else:
            raise ValueError("Unknown activation function {}".format(activation))
        self.input_layer = nn.Sequential(nn.Linear(dim_x+1, dim_S), self.activation).to(torch.float64)
        self.DGM1 = DGM_Layer(dim_x=dim_x+1, dim_S=dim_S, activation=activation).to(torch.float64)
        self.DGM2 = DGM_Layer(dim_x=dim_x+1, dim_S=dim_S, activation=activation).to(torch.float64)
        self.DGM3 = DGM_Layer(dim_x=dim_x+1, dim_S=dim_S, activation=activation).to(torch.float64)
        self.output_layer = nn.Linear(dim_S, 1).to(torch.float64)


    def forward(self,t,x):
        tx = torch.cat([t,x], 1)
        S1 = self.input_layer(tx)
        S2 = self.DGM1(tx,S1)
        S3 = self.DGM2(tx,S2)
        S4 = self.DGM3(tx,S3)
        output = self.output_layer(S4)
        return output


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

# Initialize LQR object
lqr = ee.LQR(H, M, C, D, R, t_0, T)
lqr.update_t_0_and_T(t_0=t_0, T=T, n=100)

# Instantiate the model
net_dgm = Net_DGM(dim_x=2, dim_S=100)

# Loss function and optimizer
loss_function = nn.MSELoss()
optimizer = torch.optim.Adam(net_dgm.parameters(), lr=0.001)

# Training loop
losses = []
no_improvement_count = 0
epoch_counter = 0
train = True
print('Starting Training Epoch 0 / {}'.format(max_num_epochs))

while train:
    epoch_counter +=1
    if epoch_counter%50 == 0:
        print('Epoch {} / {}'.format(epoch_counter,max_num_epochs))
    t_tensor = torch.tensor(np.random.uniform(low=0, high=T, size=batch_size)).unsqueeze(1)
    x_tensor = torch.tensor(np.random.uniform(low=(-3, -3), high=(3, 3), size=(batch_size,1, 2)))
    x_reshaped = x_tensor.reshape(-1, 2)

    y = lqr.control_problem_value(t_tensor, x_tensor, sigma).unsqueeze(1)
    y_pred = net_dgm.forward(t_tensor, x_reshaped).to(torch.float64)

    loss = loss_function(y_pred, y)
    losses.append(loss.item())
    net_dgm.zero_grad()
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

# Plot losses
plt.plot(losses)
plt.xlabel('Epoch')
plt.ylabel('MSE Loss')
plt.title('Training Loss')
plt.show()
