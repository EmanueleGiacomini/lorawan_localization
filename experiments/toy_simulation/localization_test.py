"""localization_test.py
"""

import numpy as np
import torch
from matplotlib import pyplot as plt
import pandas as pd
from localizer import LocalizerILS
from density_estimator import RegressorModelV2


if __name__ == '__main__':
    model = torch.load('./logs/model0.pth')
    # Read test data
    raw_data_df = pd.read_csv('./data/cross00_validation00.csv')
    x_gt = torch.from_numpy(raw_data_df[['x', 'y']].to_numpy())
    measurements = torch.from_numpy(raw_data_df[['t0', 't1', 't2']].to_numpy())

    n_measurements, n_gateways = measurements.shape

    print(f'Experiment contains {n_measurements} measures of {n_gateways} gateways')

    localizer = LocalizerILS(n_measurements, n_gateways, model)
    # Random estimate
    #initial_guess = np.random.random((n_measurements, 2))
    #initial_guess[:, 0] *= 100
    #initial_guess[:, 1] *= 80
    #initial_guess = initial_guess.flatten()
    #estimate, chi_lst = localizer.compute(measurements, initial_guess=initial_guess, n_iter=200)
    # Good estimate
    initial_guess = x_gt.flatten().detach().numpy()
    estimate, chi_lst = localizer.compute(measurements, initial_guess=initial_guess, n_iter=200)

    # Plot section
    fig, axs = plt.subplots(1, 2)
    # Ground-Truth
    axs[0].scatter(x_gt[:, 0], x_gt[:, 1])
    axs[0].plot(x_gt[:, 0], x_gt[:, 1], 'r')
    # Computed Estimate
    estimate_mat = estimate.reshape(n_measurements, 2)
    axs[0].set_xlim([0, 100])
    axs[0].set_ylim([0, 80])
    axs[0].scatter(estimate_mat[:, 0], estimate_mat[:, 1])
    axs[0].plot(estimate_mat[:, 0], estimate_mat[:, 1], 'b')

    axs[1].plot(chi_lst)
    
    plt.show()
    exit(0)
