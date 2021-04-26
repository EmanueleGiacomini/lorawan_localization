"""density_estimator.py
"""

import pickle
import matplotlib.patches as patches
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from matplotlib import pyplot as plt


class RegressorModel(torch.nn.Module):
    def __init__(self):
        super(RegressorModel, self).__init__()

        self.fc1 = torch.nn.Linear(2, 20)
        self.fc2 = torch.nn.Linear(20, 20)
        self.fc3 = torch.nn.Linear(20, 20)
        self.fc4 = torch.nn.Linear(20, 1)

    def forward(self, x):
        out = torch.nn.ReLU()(self.fc1(x))
        out = torch.nn.Sigmoid()(self.fc2(out))
        out = torch.nn.ReLU()(self.fc3(out))
        out = self.fc4(out)
        return out

class RegressorModelV2(torch.nn.Module):
    def __init__(self, n_gateways: int, layer_size=10, num_layers=5):
        super(RegressorModelV2, self).__init__()

        self.linears = nn.ModuleList([nn.Linear(2, layer_size)])
        self.linears.extend([nn.Linear(layer_size, layer_size) for i in range(num_layers-1)])
        self.linears.extend([nn.Linear(layer_size, n_gateways)])

    def forward(self, x):
        out = nn.Sigmoid()(self.linears[0](x))
        for i in range(1, len(self.linears)-1):
            out = nn.ReLU()(self.linears[i](out))
        return self.linears[-1](out)

fname = 'cross00'

def regressor_model_v0_train(positions, g0_values, g1_values, g2_values, num_epochs):
    # TODO
    
    return

def regressor_model_v2_train(positions, g0_values, g1_values, g2_values, num_epochs):
    # Model hyperparameters
    n_gateways = 3
    layer_size = 20
    num_layers = 20
    # Training parameters
    lr = 5e-4
    model = RegressorModelV2(n_gateways, layer_size, num_layers)
    optimizer = torch.optim.RMSprop(model.parameters(), lr)
    loss_fn = nn.MSELoss()

    # Data preprocessing
    x = torch.from_numpy(positions)
    y0 = torch.from_numpy(g0_values)
    y1 = torch.from_numpy(g1_values)
    y2 = torch.from_numpy(g2_values)
    y = torch.stack((y0, y1, y2), axis=1)

    for epoch in range(num_epochs):
        y_pred = model(x.float())
        loss = loss_fn(y_pred, y.float())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if epoch % 100 == 0:
            loss_val = loss.item()
            print(f'[{epoch}] loss: {loss_val:.3f}')
    torch.save(model, './logs/model0.pth')
    return model


if __name__ == '__main__':
    np.random.seed(1)
    torch.manual_seed(1)

    fig, ax = plt.subplots(1,1, figsize=(10, 8))
    # Load obstacles to plot
    object_lst = pickle.load(open('./maps/'+fname+'_object.p', 'rb'))
    rect_idx_lst = pickle.load(open('./maps/'+fname+'_rectidx.p', 'rb'))
    
    # Read data
    raw_data_df = pd.read_csv('./data/cross00_sample01.csv')
    positions = raw_data_df[['x', 'y']].to_numpy()
    g0_values = raw_data_df['t0'].to_numpy()
    g1_values = raw_data_df['t1'].to_numpy()
    g2_values = raw_data_df['t2'].to_numpy()

    num_epochs = 70000
    #model = regressor_model_v2_train(positions, g0_values, g1_values, g2_values, num_epochs)
    model = torch.load('./logs/model0.pth')

    # Plot regression
    x_lin = np.linspace(0, 100, 100).astype(np.float32)
    y_lin = np.linspace(0, 80, 100).astype(np.float32)
    x_mesh, y_mesh = np.meshgrid(x_lin, y_lin)
    mesh = np.stack((x_mesh, y_mesh), axis=2)
    mesh = torch.from_numpy(np.reshape(mesh, (-1, 2)))
    T = model(mesh)
    T = torch.reshape(T, (x_mesh.shape[0], x_mesh.shape[1], 3))
    T_sum = torch.sum(T, axis=2)

    plt.cla()
    ax.pcolormesh(x_mesh, y_mesh, T_sum.detach().numpy(), shading='auto', cmap='viridis')
    plt.show()
    exit(0)
