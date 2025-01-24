# PAYNT

## Installation

## Docker (recommended and tested)

Quick and easy. Install Docker, then, run:
```
docker run -dit -v "$(pwd):/opt/payntdev" --name IJCAI randriu/paynt:latest bash -c 'cd /opt/payntdev && bash setup.bash && python3 entrypoint.py'
```

Output will collect in the output folder specified in `config.py`, default is `./output/IJCAI`.

To generate the plots, execute the Jupyter notebook `plots.ipynb`. Alternatively, convert it to Python code and execute as a script.

The lineplot and heatmap (enumeration) experiments take ~2 hours each. Each call to run on the union takes 1 hour each, thus 2 for both approaches. That makes 6 hours of experiments for 6 environments, totalling 36 hours if no parallelization is used. 

### Manual (untested)

Install Paynt:

```shell
git clone https://github.com/randriu/synthesis.git synthesis
cd synthesis
```

PAYNT requires [Storm](https://github.com/moves-rwth/storm) and [Stormpy](https://github.com/moves-rwth/stormpy). If you have Stormpy installed (e.g. within a Python environment), PAYNT and its dependencies can be installed by

```shell
sudo apt install -y graphviz
source ${VIRTUAL_ENV}/bin/activate
pip3 install click z3-solver psutil graphviz
cd payntbind
python3 setup.py develop
cd ..
python3 paynt.py --help
```

If you do not have Stormpy installed, you can run the installation script `install.sh` to install Storm, Stormpy and other required dependencies (designed for Ubuntu-like systems with `apt`). Complete compilation might take up to an hour. The Python environment will be available in `prerequisistes/venv`:

```shell
./install.sh
source prerequisistes/venv/bin/activate
python3 paynt.py --help
```

Run experiments with setup from `config.py`:
```
python3 entrypoint.py
```

