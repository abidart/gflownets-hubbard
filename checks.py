import numpy as np


def assert_rewardable_lattice(lattice):
    assert np.all((lattice == 0) | (lattice == 1)), "All sites should 0 or 1"
    assert np.sum(lattice[0]) == np.sum(
        lattice[1]
    ), "The number of up and down spins is different"


def assert_drawable_lattice(lattice):
    assert np.all((lattice == 0) | (lattice == 1)), "All sites should 0 or 1"


def check_valid_state(lattice) -> bool:
    correct_sites = np.all((lattice == 0) | (lattice == 1))
    balanced_states = np.sum(lattice[0]) == np.sum(lattice[1])
    return True if correct_sites and balanced_states else False
