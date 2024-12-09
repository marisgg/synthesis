import copy
import os
import random
import stormpy
import payntbind
from pomdp_families import POMDPFamiliesSynthesis
from pomdp_families import Method

import pickle

BASE_OUTPUT_DIR = "./output-new"

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

def determine_memory_model(project_path, num_nodes = 2, memory_model = None, dynamic_memory=False, seed=11):
    gd = POMDPFamiliesSynthesis(project_path, use_softmax=True, steps=1, learning_rate=0.01, dynamic_memory=dynamic_memory, seed=seed)
    assignments = gd.stratified_subfamily_sampling(num_samples=5, seed=seed)
    mem = gd.determine_memory_model_from_assignments(gd, assignments, max_num_nodes=num_nodes, seed=seed)
    print(mem)
    exit()

def run_family(project_path, num_nodes = 2, memory_model = None, dynamic_memory=False):
    gd = POMDPFamiliesSynthesis(project_path, use_softmax=False, steps=10, dynamic_memory=dynamic_memory)
    gd.run_gradient_descent_on_family(1000, num_nodes, memory_model=memory_model)

def run_family_softmax(project_path, num_nodes = 2, memory_model = None, dynamic_memory=False, seed=11):
    gd = POMDPFamiliesSynthesis(project_path, use_softmax=True, steps=1, learning_rate=0.01, dynamic_memory=dynamic_memory, seed=seed)
    gd.run_gradient_descent_on_family(1000, num_nodes, memory_model=memory_model)

def run_subfamily(project_path, subfamily_size = 10, timeout = 60, num_nodes = 2, memory_model = None, baselines = [Method.GRADIENT], seed=11, stratified=True, determine_memory_model=True):
    gd = POMDPFamiliesSynthesis(project_path, use_softmax=True, steps=1, learning_rate=0.01, seed=seed)

    if stratified:
        subfamily_assigments = gd.stratified_subfamily_sampling(subfamily_size, seed=seed)
    else:
        subfamily_assigments = gd.create_random_subfamily(subfamily_size)
    
    if determine_memory_model:
        memory_model = gd.determine_memory_model_from_assignments(subfamily_assigments,max_num_nodes=num_nodes)

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

def run_union(project_path, method):
    num_assignments = 10
    gd = POMDPFamiliesSynthesis(project_path, use_softmax=False, steps=10)
    assignments = gd.create_random_subfamily(num_assignments)
    # assignments = [pomdp_sketch.family.pick_random() for _ in range(num_assignments)]
    # [print(a) for a in assignments]
    pomdps = []
    # make sure that all POMDPs have the same action mapping so we can store only one copy
    observation_action_to_true_action = None
    nodes = 2
    for assignment in assignments:
        pomdp,obs_action_to_true_action = gd.assignment_to_pomdp(gd.pomdp_sketch,assignment,restore_absorbing_states=False)
        pomdps.append(pomdp)
        if observation_action_to_true_action is None:
            observation_action_to_true_action = obs_action_to_true_action
            continue
        # fill in labels for missing observations
        for obs,true_action in enumerate(obs_action_to_true_action):
            if true_action is None:
                obs_action_to_true_action[obs] = observation_action_to_true_action[obs].copy()
                # observation_action_to_true_action[obs] = obs_action_to_true_action[obs].copy()
        assert observation_action_to_true_action == obs_action_to_true_action, "\n".join(['', str(observation_action_to_true_action), "=/=", str(obs_action_to_true_action)])

    union_pomdp = payntbind.synthesis.createModelUnion(pomdps)
    
    if method == Method.SAYNT:
        fsc = gd.solve_pomdp_saynt(union_pomdp, gd.pomdp_sketch.specification, nodes, timeout=3)
        nodes = fsc.num_nodes
    
    elif method == Method.GRADIENT:
        value, resolution, action_function_params, memory_function_params, *_ = gd.gradient_descent_on_single_pomdp(union_pomdp, 150, nodes, timeout=10, parameter_resolution={}, resolution={}, action_function_params={}, memory_function_params={})
        fsc = gd.parameters_to_paynt_fsc(action_function_params, memory_function_params, resolution, nodes, gd.nO, gd.pomdp_sketch.observation_to_actions)

   
    initial_node = fsc.update_function[0][-1]
    
    print("\n\n\nOBSERVATION LABELS:", fsc.observation_labels, '\n\n\n')
    
    print([fsc.action_labels[act] for act in fsc.action_function[0]])
    
    new_fsc = copy.deepcopy(fsc)
    # get rid of the fresh observation 
    for n in range(nodes):
        new_fsc.action_function[n] = fsc.action_function[n][:-1]
        assert len(new_fsc.action_function[n]) == gd.nO
        new_fsc.update_function[n] = fsc.update_function[n][:-1]
        assert len(new_fsc.update_function[n]) == gd.nO

    new_fsc.num_observations -= 1
    
    print("NODES: ", nodes)

    # ensure that 0 is the initial node
    fsc_update_fixed = list(range(nodes))
    if initial_node != 0:
        assert False
        fsc_update_fixed[0] = initial_node
        fsc_update_fixed[initial_node] = 0
        tmp = copy.copy(fsc.action_function[0])
        new_fsc.action_function[0] = copy.copy(fsc.action_function[initial_node])
        new_fsc.action_function[initial_node] = tmp
        tmp = copy.copy(fsc.update_function[0])
        new_fsc.update_function[0] = copy.copy(fsc.update_function[initial_node])
        new_fsc.update_function[initial_node] = tmp
    for n in range(nodes):
        for o in range(new_fsc.num_observations):
            new_fsc.update_function[n][o] = fsc_update_fixed[new_fsc.update_function[n][o]]

    # ensure that FSC uses the same ordering of action labels as the POMDP sketch (required by fsc.check())
    for n in range(nodes):
        for o in range(new_fsc.num_observations):
            action_label = new_fsc.action_labels[new_fsc.action_function[n][o]]
            # action_label = observation_action_to_true_action[o][action_label]
            new_fsc.action_function[n][o] = gd.pomdp_sketch.action_labels.index(action_label)
    
    new_fsc.action_labels = gd.pomdp_sketch.action_labels.copy()

    # fix possibly dummy actions
    for n in range(nodes):
        for o in range(new_fsc.num_observations):
            fsc_action_label = new_fsc.action_labels[new_fsc.action_function[n][o]]
            true_action_label = observation_action_to_true_action[o][fsc_action_label]
            new_fsc.action_function[n][o] = gd.pomdp_sketch.action_labels.index(true_action_label)
            assert gd.pomdp_sketch.action_labels.index(true_action_label) == new_fsc.action_labels.index(true_action_label)
            new_fsc.action_function[n][o] = new_fsc.action_labels.index(true_action_label)

    new_fsc.check(gd.pomdp_sketch.observation_to_actions)
    # return
    # TODO make FSC stochastic ?
    
    print("OLF FSC:", [fsc.action_labels[act] for act in fsc.action_function[0]])
    print("NEW FSC:", [new_fsc.action_labels[act] for act in new_fsc.action_function[0]])
    # exit()
    
    # print(new_fsc)
    
    import paynt.quotient.fsc
    
    # paynt.quotient.fsc.FSC()
    
    new_fsc = gd.deterministic_fsc_to_stochastic_fsc(new_fsc)
    
    # print(new_fsc)
    
    print([new_fsc.action_labels[next(iter(act.keys()))] for act in new_fsc.action_function[0]])

    print(gd.pomdp_sketch.observation_to_actions)
    
    # print(gd.paynt_call_given_fsc(fsc))

    print(gd.get_values_on_subfamily(gd.get_dtmc_sketch(new_fsc), assignments)) # TODO, returns unexpected values. FSC might be incorrectly morphed back to family?
    

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

# UNION ASSERTIONERROR (ROVER):
# run_union(ROVER, Method.SAYNT)

# UNION ERROR UNEXPECTED VALUES (ANY BENCHMARK):
# run_union(OBSTACLES_TEN_TWO, Method.SAYNT)

# run_union(ROVER, Method.SAYNT)

for env in ENVS:
    run_subfamily(env, timeout=6, subfamily_size=5, num_nodes=4, determine_memory_model=False, stratified=True)
