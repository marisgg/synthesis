import copy
import functools
import itertools
import math
import os
import random
import stormpy
import payntbind
from pomdp_families import POMDPFamiliesSynthesis
from pomdp_families import Method
import paynt.quotient.fsc

import traceback

import multiprocessing

from config import *


# env = f"{BASE_SKETCH_DIR}/rocks-4-1"
# env = f"{BASE_SKETCH_DIR}/obstacles-demo"
env = f"{BASE_SKETCH_DIR}/avoid-8-2-easy"
gd = POMDPFamiliesSynthesis(env, use_softmax=True, steps=1, learning_rate=0.01, use_momentum=True, seed=0)
fsc, value = gd.run_gradient_descent_on_family(1000, num_nodes=1, timeout=600, memory_model=None, random_selection=False)