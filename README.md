# GFlowNets for sampling paths of the Hubbard model

This project uses [Poetry](https://python-poetry.org/) for package management [Black](https://github.com/psf/black) 
for formatting.

## Running the project
- Create an environment using Poetry or your package management tool of choice. The dependencies are listed in the 
`pyproject.toml` file in the root directory.
- Configure the variables in `gflownet_hubbard.py` and run the file as a script. 

## Acknowledgements
This code is based on mbi2gs's [glownet_tf2](https://github.com/mbi2gs/gflownet_tf2) codebase, Emmanuel Bengio's
[GFlowNet tutorial](https://colab.research.google.com/drive/1fUMwgu2OhYpQagpzU5mhe9_Esib3Q2VR) notebook, and 
the paper [“Generative Flow Networks for Discrete Probabilistic Modeling”](https://arxiv.org/pdf/2202.01361.pdf) 
by Zhang et. al. 