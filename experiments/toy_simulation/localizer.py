"""localizer.py
"""

import numpy as np
from scipy.misc import derivative
import torch

def partial_derivative(func, var=0, point=[]):
    args = point[:]
    def wraps(x):
        args[var] = x
        return func(torch.from_numpy(args).float())
    return derivative(wraps, point[var], dx=1e-5)


class LocalizerILS:
    """ Iterative Least Squares Localizer
    Once measurements and initial estimate is set, the localizer attempts to produce a 
    position estimate for each sample.
    """
    def __init__(self, n, n_gateways, f):
        """
        Input:
        n [int] : number of samples
        n_gateways [int] : number of gateways
        """
        self.n = n
        self.n_gateways = n_gateways
        self.f = f
        self.estimate = np.zeros(2*n)
        self.Omega = np.ones((self.n_gateways, self.n_gateways), dtype=np.float)
        
    def compute(self, z_vect, initial_guess=None, n_iter=10):
        """ Apply least squares optimization
        Input:
        z [(n, n_gateways) float] : measurement matrix
        initial_guess [(2 * n) float] : initial guess for state estimate
        """

        def compute_error_and_jacobian(i, autograd=False):
            state = self.estimate[2*i:2*i+2]
            state_torch = torch.from_numpy(state).float()
            state_torch.requires_grad=True
            z = z_vect[i:i+(self.n_gateways)]
            z_hat = self.f(state_torch)
            e = (z_hat - z)
            # Build i-th Jacobian row
            J_i = np.zeros((self.n_gateways, 2))
            # Partial derivatives for x and y
            if autograd is False:
                df_x = partial_derivative(self.f, 0, state).detach().numpy()
                df_y = partial_derivative(self.f, 1, state).detach().numpy()
                J_i[:, 0] = df_x
                J_i[:, 1] = df_y
            else:
                # Using torch autograd for partial derivatives
                # Compute backward pass for every gateway
                for j in range(self.n_gateways):
                    grad = np.zeros(self.n_gateways)
                    grad[j] = 1
                    grad = torch.from_numpy(grad)
                    e.backward(gradient=grad, retain_graph=True)
                    J_i[j, :] = state_torch.grad.detach().numpy()
            """
            for j in range(self.n_gateways):
                # Compute partial derivative for j-th gateway
                df_x = partial_derivative(self.f, 0, state).detach().numpy()
                df_y = partial_derivative(self.f, 1, state).detach().numpy()
                J_i[j, i] = df_x
                J_i[j, i+1] = df_y
            """
            # Compute error
            e = e.detach().numpy()
            return e, J_i

        def buildLinearSystemRSSI(z_vect, damping=1e-4):
            # Initialize system variables
            H = np.zeros((2*self.n, 2*self.n))
            b = np.zeros(2*self.n)
            chi_tot = 0.0
            for i in range(self.n):
                e_i, J_i = compute_error_and_jacobian(i, autograd=True)
                # Compute chi
                chi_tot += np.matmul(e_i.T, e_i)
                # Update H matrix and b vector
                H[2*i:2*i+2, 2*i:2*i+2] += np.matmul(J_i.T, np.matmul(self.Omega, J_i))
                b[2*i:2*i+2] += np.matmul(J_i.T, np.matmul(self.Omega, e_i))
            H = H + np.eye(2*self.n)*damping
            return H, b, chi_tot
            
        
        if initial_guess is None:
            self.initial_guess = np.zeros(2*self.n)
        else:
            self.initial_guess = initial_guess

        chi_lst = []

        self.estimate = self.initial_guess

        z_vect = z_vect.flatten()
        
        for iteration in range(n_iter):
            H, b, chi_it = buildLinearSystemRSSI(z_vect)
            Dx = - np.matmul(np.linalg.inv(H), b)
            
            self.estimate += Dx
            chi_lst.append(chi_it)

            # Print chi
            if iteration % 10 == 0:
                print(f'iteration {iteration}: chi={chi_it:.3f}')
                #print('H:')
                #print(H)
                #print('b:')
                #print(b)
            
        return self.estimate, chi_lst
