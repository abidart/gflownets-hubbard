import tensorflow as tf
import numpy as np
from scipy.optimize import fsolve
from visualize import draw_lattice

class Reward:

    def __init__(self, init_state):
        self.state = init_state
        self.w = self.state.shape[1]
        self.h = self.state.shape[2]

        # hyperparameters of systems, don't know what to set them to 
        self.d_tau = 1
        self.U = 2
        self.t = 4

        # gamma is a complex value, so used optimizer given other inputs
        self.gamma = self.get_gamma()

        # maybe call reward function in initializer

    def update_state(self, new_state):
        self.state = new_state
        self.w = self.state.shape[1]
        self.h = self.state.shape[2]

    def get_gamma(self):
        def equation(gamma):
            return np.tanh(gamma)**2 - np.tanh(self.d_tau * self.U / 4)
        
        gamma_guess = 0.5
        gamma_solution, = fsolve(equation, gamma_guess)

        return gamma_solution
    
    def potential_reward(self):
        """
        Computes the potential component of the HS trotterization, 
        to be used as the reward function.

        Returns:
        - all_site_product: iterates over each site in the system and
        multiplies the product component and sum component based on 
        electrons in site, multiplying each site value to get the final
        potential value of state.
        """
        all_site_product = 1

        for i in range(self.w):
            for j in range(self.h):
                n_down = self.state[0, i, j]
                n_up = self.state[1, i, j]

                product = np.exp(-self.d_tau * self.U * (n_up + n_down) / 2)

                if n_up == n_down:
                    total_spin_sum = 1
                else:
                    up_spin_sum = np.exp(2 * self.gamma * (n_up - n_down))
                    down_spin_sum = np.exp(-2 * self.gamma * (n_up - n_down))
                    total_spin_sum = 0.5 * (up_spin_sum + down_spin_sum)

                site_potential = product * total_spin_sum

                all_site_product *= site_potential

        return all_site_product
    
state_stupid = np.array([[[1., 1., 1.],
                          [1., 1., 1.],
                          [1., 1., 1.]],

                          [[1., 1., 1.],
                           [1., 1., 1.],
                           [1., 1., 1.]]])

calculator = Reward(init_state=state_stupid)

print("If all sites are doubly occupied: " + str(calculator.potential_reward()))

state_half_stupid = np.array([[[1., 1., 1.],
                               [1., 1., 1.],
                               [1., 1., 1.]],

                              [[0., 0., 0.],
                               [0., 0., 0.],
                               [0., 0., 0.]]])

valid_interesting_state = np.array([[[1., 1., 1.],
                               [0., 0., 0.],
                               [1., 1., 1.]],

                              [[1., 0., 1.],
                               [0., 1., 0.],
                               [1., 1., 1.]]])

calculator.update_state(valid_interesting_state)

print("If all sites have down electrons: " + str(calculator.potential_reward()))

draw_lattice(valid_interesting_state)
    

    
