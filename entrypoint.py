import os
import stormpy
import payntbind
from pomdp_families import POMDPFamiliesSynthesis
from pomdp_families import Method

import pickle

# Can try various models.
OBSTACLES_EIGHTH_THREE = "models/pomdp/sketches/obstacles-8-3"
OBSTACLES_TEN_TWO = "models/pomdp/sketches/obstacles-10-2"
AVOID = "models/pomdp/sketches/avoid"
DPM = "models/pomdp/sketches/dpm"
ROVER = "models/pomdp/sketches/rover"

ENVS = [OBSTACLES_TEN_TWO, DPM, AVOID]

def run_family_experiment(num_nodes = 2):
    for project_path in ENVS:

        dr = f"./outputs/{project_path.split('/')[-1]}/"
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

def run_family(project_path):
    gd = POMDPFamiliesSynthesis(project_path, use_softmax=False, steps=10)
    gd.run_gradient_descent_on_family(1000, 2)

def run_family_softmax(project_path):
    gd = POMDPFamiliesSynthesis(project_path, use_softmax=True, steps=1, learning_rate=0.001)
    gd.run_gradient_descent_on_family(1000, 2)

def run_subfamily(project_path, subfamily_size = 10, timeout = 60, num_nodes = 2):
    gd = POMDPFamiliesSynthesis(project_path, use_softmax=True, steps=1, learning_rate=0.01)
    subfamily_assigments = gd.create_random_subfamily(subfamily_size)

    for method in [Method.SAYNT, Method.GRADIENT]:

        subfamily_other_results = gd.experiment_on_subfamily(subfamily_assigments, num_nodes, method, num_gd_iterations=250, timeout=timeout, evaluate_on_whole_family=True)

        dr = f"./outputs/{project_path.split('/')[-1]}/{subfamily_size}/"
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
        'whole_family' : family_value
    }

    print("OURS:", our_evaluations, 'family value:', family_value)
    with open(f"{dr}/ours.pickle", 'wb') as handle:
        pickle.dump(our_results, handle)

def run_union(project_path):
    num_assignments = 10
    gd = POMDPFamiliesSynthesis(project_path, use_softmax=False, steps=10)
    assignments = gd.create_random_subfamily(10)
    # assignments = [pomdp_sketch.family.pick_random() for _ in range(num_assignments)]
    # [print(a) for a in assignments]
    pomdps = [gd.assignment_to_pomdp(gd.pomdp_sketch,assignment)[0] for assignment in assignments]
    print([pomdp for pomdp in pomdps])
    # fsc = gd.solve_pomdp_paynt(pomdps[0], gd.pomdp_sketch.specification, 2, timeout=5)
    # print(fsc)

    # exit()



    union_pomdp = payntbind.synthesis.createModelUnion(pomdps)
    print(union_pomdp)

    nodes = 2
    fsc = gd.solve_pomdp_paynt(union_pomdp, gd.pomdp_sketch.specification, nodes, timeout=5)
    initial_node = fsc.update_function[0][-1]
    print("initial node is:", initial_node)
    print(fsc)
    for n in range(nodes):
        fsc.action_function[n] = fsc.action_function[n][:-1]
        assert len(fsc.action_function[n]) == gd.nO
        fsc.update_function[n] = fsc.update_function[n][:-1]
        assert len(fsc.update_function[n]) == gd.nO

    if initial_node == 1:
        action_function = [None] * nodes
        update_function = [None] * nodes
        for n in range(nodes):
            action_function[n] = fsc.action_function[(n+1) % 2]
            assert len(action_function[n]) == gd.nO
            update_function[n] = [(m+1) % nodes for m in fsc.update_function[(n+1) % 2]] # TODO
            assert len(update_function[n]) == gd.nO
        fsc.action_function = action_function
        fsc.update_function = update_function
    fsc.num_observations -= 1
    print(fsc)
    print(gd.pomdp_sketch.observation_to_actions)

    print(gd.nO)

    print(gd.get_values_on_subfamily(gd.get_dtmc_sketch(fsc), assignments))

# run_family()
# run_family_softmax(OBSTACLES_TEN_TWO)
# run_family_experiment()
run_subfamily(ROVER, timeout=30)
# for env, timeout in zip(ENVS, [10, 10, 60]):
    # run_subfamily(env, timeout=timeout)
# run_subfamily(num_nodes=3, timeout=60)
