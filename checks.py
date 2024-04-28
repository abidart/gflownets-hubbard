import numpy as np


def check_valid_lattice(lattice):
    assert np.all((lattice == 0) | (lattice == 1)), "All sites should 0 or 1"
    assert np.sum(lattice[0]) == np.sum(lattice[1]), "The number of up and down spins is different"
