import copy
import os
import random
import stormpy
import payntbind
from pomdp_families import POMDPFamiliesSynthesis
from pomdp_families import Method

from multiprocessing import Pool

import pickle

BASE_OUTPUT_DIR = "./output-parallel-subfamily"

BASE_SKETCH_DIR = 'models/pomdp/sketches'



# WIP models, might get deleted:
ACO = f"{BASE_SKETCH_DIR}/aco"
INTERCEPT = f"{BASE_SKETCH_DIR}/intercept"

# Can try various models.
OBSTACLES_EIGHTH_THREE = f"{BASE_SKETCH_DIR}/obstacles-8-3"
OBSTACLES_TEN_TWO = f"{BASE_SKETCH_DIR}/obstacles-10-2"
AVOID = f"{BASE_SKETCH_DIR}/avoid"
DPM = f"{BASE_SKETCH_DIR}/dpm"
ROVER = f"{BASE_SKETCH_DIR}/rover"
NETWORK = f"{BASE_SKETCH_DIR}/network"

ENVS = [OBSTACLES_TEN_TWO, OBSTACLES_EIGHTH_THREE, DPM, AVOID, ROVER, NETWORK]

def run_family_experiment(project_path, num_nodes = 2, memory_model=None, timeout=None, max_iter=1000):

    dr = f"{BASE_OUTPUT_DIR}/{project_path.split('/')[-1]}/"
    os.makedirs(dr, exist_ok=True)

    gd = POMDPFamiliesSynthesis(project_path, use_softmax=True, steps=1, learning_rate=0.01, use_momentum=False)
    gd.run_gradient_descent_on_family(max_iter, num_nodes, timeout=timeout, memory_model=memory_model)

    results = {}
    results['gd-no-momentum'] = {
        'family_trace' : gd.family_trace,
        'gd_trace' : gd.gd_trace,
        'current_values' : gd.current_values
    }

    gd = POMDPFamiliesSynthesis(project_path, use_softmax=True, steps=1, learning_rate=0.01, use_momentum=True)
    gd.run_gradient_descent_on_family(max_iter, num_nodes, timeout=timeout, memory_model=memory_model, random_selection=True)
    
    results['gd-random'] = {
        'family_trace' : gd.family_trace,
        'gd_trace' : gd.gd_trace,
        'current_values' : gd.current_values
    }

    gd = POMDPFamiliesSynthesis(project_path, use_softmax=True, steps=1, learning_rate=0.01, use_momentum=True)
    gd.run_gradient_descent_on_family(max_iter, num_nodes, timeout=timeout, memory_model=memory_model)

    results['gd-normal'] = {
        'family_trace' : gd.family_trace,
        'gd_trace' : gd.gd_trace,
        'current_values' : gd.current_values
    }

    with open(f"{dr}/gd-experiment.pickle", 'wb') as handle:
        pickle.dump(results, handle)

def determine_memory_model_stratified(project_path, num_nodes = 2, seed=11, num_samples=5):
    gd = POMDPFamiliesSynthesis(project_path, use_softmax=True, steps=1, learning_rate=0.01, seed=seed)
    assignments = gd.stratified_subfamily_sampling(num_samples, seed=seed)
    mem = gd.determine_memory_model_from_assignments(assignments, max_num_nodes=num_nodes)
    print(mem)
    return mem

def run_family(project_path, num_nodes = 2, memory_model = None, dynamic_memory=False):
    gd = POMDPFamiliesSynthesis(project_path, use_softmax=False, steps=10, dynamic_memory=dynamic_memory)
    gd.run_gradient_descent_on_family(1000, num_nodes, memory_model=memory_model)

def run_family_softmax(project_path, num_nodes = 2, memory_model = None, dynamic_memory=False, seed=11):
    gd = POMDPFamiliesSynthesis(project_path, use_softmax=True, steps=1, learning_rate=0.01, dynamic_memory=dynamic_memory, seed=seed)
    gd.run_gradient_descent_on_family(1000, num_nodes, memory_model=memory_model)

def run_subfamily(project_path, subfamily_size = 10, timeout = 60, num_nodes = 2, memory_model = None, baselines = [Method.GRADIENT, Method.SAYNT], seed=11, stratified=True, determine_memory_model=True):
    gd = POMDPFamiliesSynthesis(project_path, use_softmax=True, steps=1, learning_rate=0.01, seed=seed)

    if stratified:
        subfamily_assigments = gd.stratified_subfamily_sampling(subfamily_size, seed=seed)
    else:
        subfamily_assigments = gd.create_random_subfamily(subfamily_size)
    
    if determine_memory_model:
        memory_model = gd.determine_memory_model_from_assignments(subfamily_assigments,max_num_nodes=num_nodes)
        num_nodes = max(memory_model)

    dr = f"{BASE_OUTPUT_DIR}/{project_path.split('/')[-1]}/{subfamily_size}/"
    os.makedirs(dr, exist_ok=True)

    for method in baselines:

        subfamily_other_results = gd.experiment_on_subfamily(subfamily_assigments, num_nodes, method, memory_model=memory_model, num_iters=1000, timeout=timeout, evaluate_on_whole_family=True)

        with open(f"{dr}/{method.name.lower()}.pickle", 'wb') as handle:
            pickle.dump(subfamily_other_results, handle)

    best_gd_fsc, subfamily_gd_best_value = gd.run_gradient_descent_on_family(1000, num_nodes, subfamily_assigments, timeout=subfamily_size*timeout, memory_model=memory_model)
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
    with open(f"{dr}/ours-sparse.pickle", 'wb') as handle:
        pickle.dump(our_results, handle)
    
    return memory_model

def run_union(project_path, method):
    num_assignments = 10
    gd = POMDPFamiliesSynthesis(project_path, use_softmax=False, steps=10)
    assignments = gd.create_random_subfamily(num_assignments)
    # assignments = [pomdp_sketch.family.pick_random() for _ in range(num_assignments)]
    # [print(a) for a in assignments]
    pomdps = []
    pomdp_maps = []
    nodes = 2
    for assignment in assignments:
        pomdp,true_action_map = gd.assignment_to_pomdp(gd.pomdp_sketch,assignment,restore_absorbing_states=False)
        pomdps.append(pomdp)
        pomdp_maps.append(true_action_map)

    # make sure that all POMDPs have the same action mapping so we can store only one copy
    observation_action_to_true_action = []
    for obs in range(gd.pomdp_sketch.num_observations):
        obs_map = None
        for pomdp_map in pomdp_maps:
            obs_map_next = pomdp_map[obs]
            if obs_map_next is None:
                continue
            if obs_map is None:
                obs_map = obs_map_next
                continue
            assert obs_map == obs_map_next, "\n".join(['', str(obs_map), "=/=", str(obs_map_next)])
        observation_action_to_true_action.append(obs_map)

    # create and solve the union
    union_pomdp = payntbind.synthesis.createModelUnion(pomdps)
    if method == Method.SAYNT:
        fsc = gd.solve_pomdp_saynt(union_pomdp, gd.pomdp_sketch.specification, nodes, timeout=5)
    elif method == Method.GRADIENT:
        value, resolution, action_function_params, memory_function_params, *_ = gd.gradient_descent_on_single_pomdp(union_pomdp, 150, nodes, timeout=10, parameter_resolution={}, resolution={}, action_function_params={}, memory_function_params={})
        fsc = gd.parameters_to_paynt_fsc(action_function_params, memory_function_params, resolution, nodes, gd.nO, gd.pomdp_sketch.observation_to_actions)

    print("OLD FSC:", [fsc.action_labels[act] if act is not None else None for act in fsc.action_function[0]])

    # the last observation is the fresh one for the fresh initial state
    # get the initial memory node set from this fres initial state
    assert fsc.num_observations == gd.pomdp_sketch.num_observations+1
    initial_node = fsc.update_function[0][-1]

    # get rid of the fresh observation
    for node in range(fsc.num_nodes):
        fsc.action_function[node] = fsc.action_function[node][:-1]
        fsc.update_function[node] = fsc.update_function[node][:-1]
    fsc.num_observations -= 1

    # fix possibly dummy actions
    for node in range(fsc.num_nodes):
        for obs in range(fsc.num_observations):
            action = fsc.action_function[node][obs]
            if action is None:
                continue
            action_label = fsc.action_labels[action]
            true_action_label = observation_action_to_true_action[obs][action_label]
            fsc.action_function[node][obs] = fsc.action_labels.index(true_action_label)

    # fill actions for unreachable observations
    for node in range(fsc.num_nodes):
        for obs in range(fsc.num_observations):
            if fsc.action_function[node][obs] is None:
                available_action = gd.pomdp_sketch.observation_to_actions[obs][0]
                available_action_label = gd.pomdp_sketch.action_labels[available_action]
                fsc.action_function[node][obs] = fsc.action_labels.index(available_action_label)

    # make 0 the initial node and reorder the actions
    node_order = list(range(fsc.num_nodes))
    node_order[0] = initial_node; node_order[initial_node] = 0
    fsc.reorder_nodes(node_order)
    fsc.reorder_actions(gd.pomdp_sketch.action_labels)
    fsc.check(gd.pomdp_sketch.observation_to_actions)
    
    print("NEW FSC:", [fsc.action_labels[act] for act in fsc.action_function[0]])
    fsc.make_stochastic()
    
    print(gd.get_values_on_subfamily(gd.get_dtmc_sketch(fsc), assignments)) # TODO, returns unexpected values. FSC might be incorrectly morphed back to family?
    exit()
    

# run_family_softmax(AVOID, 4)
# run_family_softmax(ACO)
# run_family_softmax(AVOID, num_nodes=2, dynamic_memory=False)
# run_family_softmax(OBSTACLES_TEN_TWO, num_nodes=2, memory_model=[1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1], seed=0)
# run_subfamily(OBSTACLES_TEN_TWO, subfamily_size=5)
# run_family_softmax(ROVER, num_nodes=3, memory_model=[random.randint(1,3) for _ in range(20)])
# run_subfamily(ROVER, 5, 60, )
# run_subfamily(OBSTACLES_TEN_TWO, num_nodes=4, memory_model=[1, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4], baselines=[], timeout=20)
# run_subfamily(OBSTACLES_TEN_TWO, num_nodes=2, memory_model=None, baselines=[], timeout=20)
# run_family_softmax(OBSTACLES_TEN_TWO, 4, memory_model=[1, 1, 1, 1, 1, 1, 4, 1, 1, 1, 1, 1, 1, 4])
# run_family_softmax(OBSTACLES_TEN_TWO, 4, memory_model=[4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4])
# run_family_softmax(OBSTACLES_TEN_TWO, 4, memory_model=[2, 3, 2, 2, 2, 2, 4, 2, 2, 2, 2, 4, 4, 4])
# run_family_softmax(OBSTACLES_TEN_TWO, 2, memory_model=[1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2])
# run_family_softmax(OBSTACLES_TEN_TWO, 2, memory_model=[1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
# run_family_softmax(OBSTACLES_TEN_TWO, 2)
# run_family(OBSTACLES_TEN_TWO, 2)
# run_family_softmax(OBSTACLES_TEN_TWO, 4)
# run_family_experiment()
# run_subfamily(ROVER, timeout=30)
# for env, timeout in zip([DPM, AVOID, OBSTACLES_TEN_TWO, ROVER, NETWORK], [10, 60]):
    # run_subfamily(env, timeout=timeout, subfamily_size=5)
# run_subfamily(num_nodes=3, timeout=60)

# SAYNT ERROR ACTIONS == [] (ROVER/NETWORK):
# run_family_softmax(ROVER, num_nodes=3)
# determine_memory_model(NETWORK)

# UNION ERROR UNEXPECTED VALUES (ANY BENCHMARK):
# RA: fixed
# run_union(OBSTACLES_TEN_TWO, Method.SAYNT)
# run_union(ROVER, Method.SAYNT)

# run_union(ROVER, Method.SAYNT)

# run_family_softmax(OBSTACLES_EIGHTH_THREE, num_nodes=2, memory_model=[1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,2,2,2,2,2,2,2,2,2,2,1])

def run(env):
    memory_model = None
    try:
        memory_model = run_subfamily(env, timeout=60, subfamily_size=5, num_nodes=5, determine_memory_model=True, stratified=True)
    except Exception as e:
        print("SUBFAMILY EXPERIMENT FAILED FOR", env)
        print(e)
    try:
        if memory_model is None:
            memory_model = determine_memory_model_stratified(env, 5)
        # run_family_softmax(env, num_nodes=max(memory_model), memory_model=memory_model, dynamic_memory=False, seed=11)
        run_family_experiment(env, max(memory_model), memory_model, max_iter=1000, timeout=600)
    except Exception as e:
        print("FULL FAMILY GRADIENT DESCENT EXPERIMENT FAILED FOR", env)
        print(e)

with Pool(len(ENVS)) as p:
    p.map(run, ENVS)

# for env in ENVS:
    # run(env)
