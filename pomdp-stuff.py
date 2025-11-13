import paynt
import payntbind
import stormpy

import paynt.parser.sketch
from paynt.quotient.pomdp import PomdpQuotient

from stormpy.storage._storage import SparseMdp
from collections import defaultdict

import os
import json
import sys

import click
import cProfile
import pstats

import logging
import time
logger = logging.getLogger(__name__)

def setup_logger(log_path = None):
    ''' Setup routine for logging. '''

    root = logging.getLogger()
    root.setLevel(logging.DEBUG)
    # root.setLevel(logging.INFO)

    # formatter = logging.Formatter('%(asctime)s %(threadName)s - %(name)s - %(levelname)s - %(message)s')
    formatter = logging.Formatter('%(asctime)s - %(filename)s:%(lineno)d - %(message)s')

    handlers = []
    if log_path is not None:
        fh = logging.FileHandler(log_path)
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(formatter)
        handlers.append(fh)
    sh = logging.StreamHandler(sys.stdout)
    handlers.append(sh)
    sh.setLevel(logging.DEBUG)
    sh.setFormatter(formatter)
    for h in handlers:
        root.addHandler(h)
    return handlers


@click.command()
@click.argument('project', type=click.Path(exists=True))
@click.option("--sketch", default="sketch.templ", show_default=True,
    help="name of the sketch file in the project")
@click.option("--props", default="sketch.props", show_default=True,
    help="name of the properties file in the project")
def main(project, sketch, props):

    model_file = os.path.join(project, sketch)
    props_file = os.path.join(project, props)
    quotient = paynt.parser.sketch.Sketch.load_sketch(model_file, props_file)

    assert quotient.pomdp is not None, "POMDP on the input expected"

    pomdp = quotient.pomdp

    print(f"Input POMDP has {pomdp.nr_states} states, {pomdp.nr_observations} observations, {pomdp.nr_choices} choices and {pomdp.nr_transitions} transitions.")

    components = stormpy.SparseModelComponents(
        transition_matrix = pomdp.transition_matrix,
        state_labeling = pomdp.labeling,
        reward_models = pomdp.reward_models,
        rate_transitions = False)
    components.choice_labeling = pomdp.choice_labeling

    obs_fun = pomdp.observations

    approx_model = pomdp

    underlying_mdp = stormpy.storage.SparseMdp(components)

    for _ in range(5):
        belief_support_unfolder = unfold(approx_model)
        unfolded_mdp : SparseMdp = belief_support_unfolder.belief_support_mdp()

        counts = defaultdict(lambda : 0)

        self_loop_belsup_states = identify_loops(unfolded_mdp)

        for belsupstate in self_loop_belsup_states:
            belsup = belief_support_unfolder.get_belief_support_of_state(belsupstate)
            for s in belsup:
                counts[s] += 1
            obs = set([pomdp.observations[s] for s in belsup])
            assert len(obs) == 1
            [obs] = obs

        most_occurring_state = max(counts, key=counts.get)

        # for state in counts.keys():
            # obs_fun = add_observability(obs_fun, state)

        obs_fun = add_observability(obs_fun, most_occurring_state)

        components.observability_classes = obs_fun

        approx_model =  stormpy.storage.SparsePomdp(components)

        approx_model = stormpy.pomdp.make_canonic(approx_model)

        print(quotient.get_property().property.raw_formula)

        print("UBbel(100k):", check_pomdp(quotient, approx_model, 100000, interactive=False))
        print("UBbel(overapp):", check_pomdp(quotient, approx_model, 100000, overapp=True))
    print("POMDP:", check_pomdp(quotient, pomdp, 10000, overapp=True))
    result = stormpy.check_model_sparse(pomdp, quotient.get_property().property.raw_formula, force_fully_observable=True)
    print("UBMDP:", result)

def identify_loops(belief_support_mdp):
    self_loop_belsup_states = set()

    for state in belief_support_mdp.states:
        for action in state.actions:
            for succ in action.transitions:
                if state.id == succ.column:
                    self_loop_belsup_states.add(state.id)

    print(len(self_loop_belsup_states), "out of", belief_support_mdp.nr_states)

    self_loop_belsup_states = sorted(list(self_loop_belsup_states))

    return self_loop_belsup_states

def unfold(pomdp):
    belief_support_unfolder = payntbind.synthesis.BeliefSupportUnfolder(pomdp)
    belief_support_unfolder.unfold_belief_support_mdp()
    return belief_support_unfolder

def add_observability(obs_fun, state):
    new_obs = max(obs_fun) + 1

    obs_fun[state] = new_obs

    return obs_fun

def get_interactive_options():
    options = stormpy.pomdp.BeliefExplorationModelCheckerOptionsDouble(False, True)
    options.use_state_elimination_cutoff = False
    options.size_threshold_init = 0
    options.skip_heuristic_schedulers = False
    options.interactive_unfolding = True
    options.gap_threshold_init = 0
    options.refine = False
    options.cut_zero_gap = False
    options.use_clipping = False
    # if self.storm_options == "clip2":
    #     options.use_clipping = True
    #     options.clipping_grid_res = 2
    # elif self.storm_options == "clip4":
    #     options.use_clipping = True
    #     options.clipping_grid_res = 4
    return options


def get_overapp_options(belief_states=20000000):
    options = stormpy.pomdp.BeliefExplorationModelCheckerOptionsDouble(True, False)
    options.use_state_elimination_cutoff = False
    options.size_threshold_init = belief_states
    options.use_clipping = False
    return options



def check_pomdp(quotient, model, num_beliefs = 100, interactive = False, overapp = False):
    if interactive:
        options = get_interactive_options()
    elif overapp:
        options = get_overapp_options(num_beliefs)
    else:
        options = stormpy.pomdp.BeliefExplorationModelCheckerOptionsDouble(False, True)
        # options = stormpy.pomdp.BeliefExplorationModelCheckerOptionsDouble(False, True)
        # options.use_state_elimination_cutoff = True
        # options.size_threshold_init = 100_000
        # options.use_clipping = True
        # options.clipping_grid_res = 2
        # options.gap_threshold_init = 0.1
        options.use_state_elimination_cutoff = False
        options.size_threshold_init = num_beliefs
        options.use_clipping = False
    belmc = stormpy.pomdp.BeliefExplorationModelCheckerDouble(model, options)
    result = belmc.check(quotient.get_property().property.raw_formula, [])
    print(belmc.has_converged())
    # print(dir(belmc))
    # print(dir(result))
    return (result.lower_bound, result.upper_bound)

if __name__ == "__main__":
    # setup_logger()
    main()