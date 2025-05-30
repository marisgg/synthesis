# rfPG: Robust Finite-Memory Policy Gradients for Hidden-Model POMDPs

This repository contains the code and docker set-up to reproduce the experiments of the IJCAI 2025 paper: "rfPG: Robust Finite-Memory Policy Gradients for Hidden-Model POMDPs".

## Experiments

The lineplot and heatmap (enumeration) experiments take ~2 hours each. Each call to run on the union takes 1 hour each, thus 2 for both approaches. That makes 6 hours of experiments for 6 environments, totalling 36 hours if no parallelization is used. 

Output will collect in the output folder specified in `config.py`, default is `./output/IJCAI`. Since we make use of a volume, the output will also appear in this repository when executed the experiments through the Docker environment.

Adapt `config.py` to set various global hyperparameters and experiments to run. Its constants are used in `entrypoint.py`.

To generate the plots, execute the Jupyter notebook `plots.ipynb`. Alternatively, convert it to Python code and execute as a script.

## Installation

### Docker with precompiled image (recommended and tested)

Download the docker image `rfpg.tar` from our [Zenodo repository](https://doi.org/10.5281/zenodo.15479643) and load it with, e.g., `docker load -i rfpg.tar`. Follow similar instructions for using `podman`. The image should appear as `localhost/rfpg:ijcai`. Adapt the instructions below accordingly if you load the image under a different name.

Then, execute the following *in the root directory of this repository*:

```shell
docker run  -v "$(pwd):/opt/payntdev" --name YOURCONTAINERNAMEHERE localhost/rfpg:ijcai python3 entrypoint.py
```

#### Docker with pull 

Use helper script to build on top of the Paynt image we created. Quick and easy, but it *may break in the future*. Run the following *in the root directory of this repository*:
```shell
docker run -dit -v "$(pwd):/opt/payntdev" --name YOURCONTAINERNAMEHERE randriu/paynt:latest bash -c 'cd /opt/payntdev && bash setup.bash && python3 entrypoint.py'
```

