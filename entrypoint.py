import os
import stormpy
import payntbind
from pomdp_families import POMDPFamiliesSynthesis
from pomdp_families import Method

import pickle

BASE_OUTPUT_DIR = "./outputs/second"

BASE_SKETCH_DIR = 'models/pomdp/sketches'

# Can try various models.
OBSTACLES_EIGHTH_THREE = f"{BASE_SKETCH_DIR}/obstacles-8-3"
OBSTACLES_TEN_TWO = f"{BASE_SKETCH_DIR}/obstacles-10-2"
AVOID = f"{BASE_SKETCH_DIR}/avoid"
DPM = f"{BASE_SKETCH_DIR}/dpm"
ROVER = f"{BASE_SKETCH_DIR}/rover"
ACO = f"{BASE_SKETCH_DIR}/aco"

ENVS = [OBSTACLES_TEN_TWO, OBSTACLES_EIGHTH_THREE, DPM, AVOID]

def run_family_experiment(num_nodes = 2):
    for project_path in ENVS:

        dr = f"{BASE_OUTPUT_DIR}/{project_path.split('/')[-1]}/"
        os.makedirs(dr, exist_ok=True)

        gd = POMDPFamiliesSynthesis(project_path, use_softmax=True, steps=1, learning_rate=0.01, use_momentum=False)
        gd.run_gradient_descent_on_family(150, num_nodes)

        results = {}
        results['gd-no-momentum'] = {
            'family_trace' : gd.family_trace,
            'gd_trace' : gd.gd_trace,
            'current_values' : gd.current_values
        }

        gd = POMDPFamiliesSynthesis(project_path, use_softmax=True, steps=1, learning_rate=0.001, use_momentum=True)
        gd.run_gradient_descent_on_family(150, num_nodes, random_selection=True)
        
        results['gd-random'] = {
            'family_trace' : gd.family_trace,
            'gd_trace' : gd.gd_trace,
            'current_values' : gd.current_values
        }

        gd = POMDPFamiliesSynthesis(project_path, use_softmax=True, steps=1, learning_rate=0.001, use_momentum=True)
        gd.run_gradient_descent_on_family(150, num_nodes)

        results['gd-normal'] = {
            'family_trace' : gd.family_trace,
            'gd_trace' : gd.gd_trace,
            'current_values' : gd.current_values
        }

        with open(f"{dr}/gd-experiment.pickle", 'wb') as handle:
            pickle.dump(results, handle)

def run_family(project_path, num_nodes = 2, memory_model = None):
    gd = POMDPFamiliesSynthesis(project_path, use_softmax=False, steps=10)
    gd.run_gradient_descent_on_family(1000, num_nodes, memory_model=memory_model)

def run_family_softmax(project_path, num_nodes = 2, memory_model = None):
    gd = POMDPFamiliesSynthesis(project_path, use_softmax=True, steps=1, learning_rate=0.01)
    gd.run_gradient_descent_on_family(1000, num_nodes, memory_model=memory_model)

def run_subfamily(project_path, subfamily_size = 10, timeout = 60, num_nodes = 2):
    gd = POMDPFamiliesSynthesis(project_path, use_softmax=True, steps=1, learning_rate=0.01)
    subfamily_assigments = gd.create_random_subfamily(subfamily_size)

    for method in [Method.SAYNT, Method.GRADIENT]:

        subfamily_other_results = gd.experiment_on_subfamily(subfamily_assigments, num_nodes, method, num_gd_iterations=1000, timeout=timeout, evaluate_on_whole_family=True)

        dr = f"{BASE_OUTPUT_DIR}/{project_path.split('/')[-1]}/{subfamily_size}/"
        os.makedirs(dr, exist_ok=True)
        with open(f"{dr}/{method.name.lower()}.pickle", 'wb') as handle:
            pickle.dump(subfamily_other_results, handle)

    best_gd_fsc, subfamily_gd_best_value = gd.run_gradient_descent_on_family(1000, num_nodes, subfamily_assigments, timeout=subfamily_size*timeout)
    print(subfamily_gd_best_value)

    dtmc_sketch = gd.get_dtmc_sketch(best_gd_fsc)

    our_evaluations = gd.get_values_on_subfamily(dtmc_sketch, subfamily_assigments)

    _, family_value = gd.paynt_call(dtmc_sketch)

    our_results = {
        'ours' : our_evaluations,
        'whole_family' : family_value,
        'fsc' : best_gd_fsc
    }

    print("OURS:", our_evaluations, 'family value:', family_value)
    with open(f"{dr}/ours.pickle", 'wb') as handle:
        pickle.dump(our_results, handle)

def run_union(project_path, method):
    num_assignments = 10
    gd = POMDPFamiliesSynthesis(project_path, use_softmax=False, steps=10)
    assignments = gd.create_random_subfamily(num_assignments)
    # assignments = [pomdp_sketch.family.pick_random() for _ in range(num_assignments)]
    # [print(a) for a in assignments]
    pomdps = []
    # make sure that all POMDPs have the same action mapping so we can store only one copy
    observation_action_to_true_action = None
    for assignment in assignments:
        pomdp,obs_action_to_true_action = gd.assignment_to_pomdp(gd.pomdp_sketch,assignment,restore_absorbing_states=False)
        pomdps.append(pomdp)
        if observation_action_to_true_action is None:
            observation_action_to_true_action = obs_action_to_true_action
        else:
            assert observation_action_to_true_action == obs_action_to_true_action
    # [print(pomdp) for pomdp in pomdps]

    union_pomdp = payntbind.synthesis.createModelUnion(pomdps)

    nodes = 2
    
    if method == Method.SAYNT:
        fsc = gd.solve_pomdp_saynt(union_pomdp, gd.pomdp_sketch.specification, nodes, timeout=15)
        print(fsc.memory_model)
        exit()
    
    elif method == Method.GRADIENT:
        value, resolution, action_function_params, memory_function_params, *_ = gd.gradient_descent_on_single_pomdp(union_pomdp, 150, 2, timeout=10, parameter_resolution={}, resolution={}, action_function_params={}, memory_function_params={})
        fsc = gd.parameters_to_paynt_fsc(action_function_params, memory_function_params, resolution, 2, gd.nO, gd.pomdp_sketch.observation_to_actions)

    # get rid of the fresh observation
    initial_node = fsc.update_function[0][-1]
    for n in range(nodes):
        fsc.action_function[n] = fsc.action_function[n][:-1]
        assert len(fsc.action_function[n]) == gd.nO
        fsc.update_function[n] = fsc.update_function[n][:-1]
        assert len(fsc.update_function[n]) == gd.nO

    fsc.num_observations -= 1

    # ensure that 0 is the initial node
    fsc_update_fixed = list(range(nodes))
    if initial_node != 0:
        fsc_update_fixed[0] = initial_node
        fsc_update_fixed[initial_node] = 0
        tmp = fsc.action_function[0]; fsc.action_function[0] = fsc.action_function[initial_node]; fsc.action_function[initial_node] = tmp
        tmp = fsc.update_function[0]; fsc.update_function[0] = fsc.update_function[initial_node]; fsc.update_function[initial_node] = tmp
    for n in range(nodes):
        for o in range(fsc.num_observations):
            fsc.update_function[n][o] = fsc_update_fixed[fsc.update_function[n][o]]

    # ensure that FSC uses the same ordering of action labels as the POMDP sketch (required by fsc.check())
    for n in range(nodes):
        for o in range(fsc.num_observations):
            action_label = fsc.action_labels[fsc.action_function[n][o]]
            fsc.action_function[n][o] = gd.pomdp_sketch.action_labels.index(action_label)
    fsc.action_labels = gd.pomdp_sketch.action_labels.copy()

    # fix possibly dummy actions
    for n in range(nodes):
        for o in range(fsc.num_observations):
            fsc_action_label = fsc.action_labels[fsc.action_function[n][o]]
            true_action_label = observation_action_to_true_action[o][fsc_action_label]
            fsc.action_function[n][o] = fsc.action_labels.index(true_action_label)

    fsc.check(gd.pomdp_sketch.observation_to_actions)
    # return
    # TODO make FSC stochastic ?
    
    print(fsc)
    
    fsc = gd.deterministic_fsc_to_stochastic_fsc(fsc)
    
    print(fsc)

    print(gd.pomdp_sketch.observation_to_actions)

    print(gd.get_values_on_subfamily(gd.get_dtmc_sketch(fsc), assignments))

# run_family(OBSTACLES_TEN_TWO, 4)
# run_family_softmax(ACO)
# run_family_softmax(OBSTACLES_TEN_TWO, 4, memory_model=[1, 3, 1, 1, 1, 1, 4, 1, 1, 1, 1, 4, 4, 4])
# run_family_softmax(OBSTACLES_TEN_TWO, 4, memory_model=[2, 3, 2, 2, 2, 2, 4, 2, 2, 2, 2, 4, 4, 4])
# run_family_softmax(OBSTACLES_TEN_TWO, 2, memory_model=[1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2])
run_family_softmax(OBSTACLES_TEN_TWO, 2, memory_model=[1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
# run_family_softmax(OBSTACLES_TEN_TWO, 2)
# run_family(OBSTACLES_TEN_TWO, 2)
# run_family_softmax(OBSTACLES_TEN_TWO, 4)
# run_family_experiment()
# run_subfamily(ROVER, timeout=30)
# for env, timeout in zip([DPM, AVOID], [10, 60]):
    # run_subfamily(env, timeout=timeout, subfamily_size=5)
# run_subfamily(num_nodes=3, timeout=60)

# run_union(OBSTACLES_TEN_TWO, Method.SAYNT)