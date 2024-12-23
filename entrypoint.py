import copy
import functools
import os
import random
import stormpy
import payntbind
from pomdp_families import POMDPFamiliesSynthesis
from pomdp_families import Method

import argparse

from multiprocessing import Pool

from config import *

import pickle

def run_family_experiment_for_lineplot(project_path, num_nodes = 2, memory_model=None, timeout=None, max_iter=1000):

    dr = f"{BASE_OUTPUT_DIR}/{project_path.split('/')[-1]}/"
    os.makedirs(dr, exist_ok=True)
    results = {}
    
    # Don't need to necessarily run below:

    # gd = POMDPFamiliesSynthesis(project_path, use_softmax=True, steps=1, learning_rate=0.01, use_momentum=False)
    # gd.run_gradient_descent_on_family(max_iter, num_nodes, timeout=timeout, memory_model=memory_model)

    # results['gd-no-momentum'] = {
    #     'family_trace' : gd.family_trace,
    #     'gd_trace' : gd.gd_trace,
    #     'current_values' : gd.current_values
    # }

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
    
    results['memory_model'] = memory_model

    with open(f"{dr}/gd-experiment.pickle", 'wb') as handle:
        pickle.dump(results, handle)

def determine_memory_model_stratified(project_path, num_nodes = 2, seed=11, num_samples=5):
    gd = POMDPFamiliesSynthesis(project_path, use_softmax=True, steps=1, learning_rate=0.01, seed=seed)
    assignments, hole_combinations = gd.stratified_subfamily_sampling(num_samples, seed=seed)
    mem = gd.determine_memory_model_from_assignments(assignments, hole_combinations, max_num_nodes=num_nodes)
    # mem = gd.determine_memory_model_from_assignments_via_belief_exploration(assignments, max_num_nodes=num_nodes)
    print(mem)
    return mem

def run_family(project_path, num_nodes = 2, memory_model = None, dynamic_memory=False):
    gd = POMDPFamiliesSynthesis(project_path, use_softmax=False, steps=10, dynamic_memory=dynamic_memory)
    gd.run_gradient_descent_on_family(1000, num_nodes, memory_model=memory_model)

def run_family_softmax(project_path, num_nodes = 2, memory_model = None, dynamic_memory=False, seed=11, max_iters=1000):
    gd = POMDPFamiliesSynthesis(project_path, use_softmax=True, steps=1, learning_rate=0.01, dynamic_memory=dynamic_memory, seed=seed)
    gd.run_gradient_descent_on_family(max_iters, num_nodes, memory_model=memory_model)

def run_subfamily_for_heatmap(project_path, subfamily_size = 10, timeout = 60, num_nodes = 2, memory_model = None, baselines = [Method.GRADIENT, Method.SAYNT], seed=11, stratified=True, determine_memory_model=True):
    gd = POMDPFamiliesSynthesis(project_path, use_softmax=True, steps=1, learning_rate=0.01, seed=seed)
    subfamily_size = min(subfamily_size, gd.pomdp_sketch.family.size)
    if stratified:
        subfamily_assigments, hole_combinations = gd.stratified_subfamily_sampling(subfamily_size, seed=seed)
    else:
        subfamily_assigments, hole_combinations = gd.create_random_subfamily(subfamily_size)
    
    if determine_memory_model:
        memory_model = gd.determine_memory_model_from_assignments(subfamily_assigments, hole_combinations, max_num_nodes=num_nodes)
        num_nodes = max(memory_model)

    dr = f"{BASE_OUTPUT_DIR}/{project_path.split('/')[-1]}/{subfamily_size}/"
    os.makedirs(dr, exist_ok=True)

    for method in baselines:

        subfamily_other_results = gd.experiment_on_subfamily(subfamily_assigments, hole_combinations, num_nodes, method, memory_model=memory_model, num_iters=1000, timeout=timeout, evaluate_on_whole_family=True)

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

def run_union(project_path, method=Method.SAYNT, timeout=10, num_assignments=5, nodes=2, stratified=True, seed=11):
    gd = POMDPFamiliesSynthesis(project_path, use_softmax=False, steps=10)
    if stratified:
        assignments, hole_combinations = gd.stratified_subfamily_sampling(5, seed=seed)
    else:
        assignments, hole_combinations = gd.create_random_subfamily(num_assignments)

    pomdps = []
    pomdp_maps = []
    for assignment in assignments:
        pomdp,true_action_map = gd.assignment_to_pomdp(gd.pomdp_sketch,assignment,restore_absorbing_states=False)
        pomdps.append(pomdp)
        pomdp_maps.append(true_action_map)
        
    dr = f"{BASE_OUTPUT_DIR}/{project_path.split('/')[-1]}/union/"
    os.makedirs(dr, exist_ok=True)

    # make sure that all POMDPs have the same action mapping so we can store only one copy
    observation_action_to_true_action = []
    for obs in range(gd.pomdp_sketch.num_observations):
        obs_map = None
        for index,pomdp_map in enumerate(pomdp_maps):
            if obs >= pomdps[index].nr_observations:
                # (one of the) last observation is unreachable in this POMDP
                continue
            obs_map_next = pomdp_map[obs]
            if obs_map_next is None:
                # middle observation is unreachable in this POMDP
                continue
            if obs_map is None:
                obs_map = obs_map_next
                continue
            assert obs_map == obs_map_next, "\n".join(['', str(obs_map), "=/=", str(obs_map_next)])
        observation_action_to_true_action.append(obs_map)

    # create and solve the union
    union_pomdp = payntbind.synthesis.createModelUnion(pomdps)
    if method == Method.SAYNT:
        # Don't need Saynt hotfix here, running on a single POMDP.
        fsc = gd.solve_pomdp_saynt(union_pomdp, gd.pomdp_sketch.specification, nodes, timeout=timeout)
    elif method == Method.GRADIENT:
        value, resolution, action_function_params, memory_function_params, *_ = gd.gradient_descent_on_single_pomdp(union_pomdp, 150, nodes, timeout=timeout, parameter_resolution={}, resolution={}, action_function_params={}, memory_function_params={})
        fsc = gd.parameters_to_paynt_fsc(action_function_params, memory_function_params, resolution, nodes, gd.nO, gd.pomdp_sketch.observation_to_actions)

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

    fsc.make_stochastic()

    dtmc_sketch = gd.get_dtmc_sketch(fsc)

    assignment_values = gd.get_values_on_subfamily(dtmc_sketch, assignments)

    _, family_value = gd.paynt_call(dtmc_sketch)

    results = {
        'subfamily' : assignment_values,
        'whole_family' : family_value,
        'fsc' : fsc
    }

    print(project_path.split('/')[-1], "Assignments:", assignment_values, 'family value:', family_value)
    with open(f"{dr}/union.pickle", 'wb') as handle:
        pickle.dump(results, handle)

def run_env_heatmap(env, timeout=600):
    subfamily_size = 5
    subfamily_timeout = timeout // subfamily_size
    max_num_nodes = 5
    memory_model = None
    try:
        run_subfamily_for_heatmap(env, timeout=subfamily_timeout, subfamily_size=subfamily_size, num_nodes=max_num_nodes, determine_memory_model=True, stratified=True)
    except Exception as e:
        print("SUBFAMILY EXPERIMENT FAILED FOR", env)
        print(e)
        print(traceback.format_exc())

import traceback

def run_env_lineplot(env, timeout=600):
    max_num_nodes = 5
    try:
        memory_model = determine_memory_model_stratified(env, num_nodes=max_num_nodes, num_samples=4)
        # run_family_softmax(env, num_nodes=max(memory_model), memory_model=memory_model, dynamic_memory=False, seed=11)
        run_family_experiment_for_lineplot(env, max(memory_model), memory_model, max_iter=1000, timeout=timeout)
        # run_family_experiment(env, 5, memory_model=None, max_iter=1000, timeout=timeout)
    except Exception as e:
        print("FULL FAMILY GRADIENT DESCENT EXPERIMENT FAILED FOR", env)
        print(e)
        print(traceback.format_exc())

def run_env_all(env, timeout=600):
    run_env_lineplot(env, timeout)
    run_env_heatmap(env, timeout)
    try:
        run_union(env, timeout=timeout)
    except Exception as e:
        print("UNION EXPERIMENT FAILED FOR", env)
        print(e)

def run_union_all():
    for env in ENVS:
        run_union(env)

def run_union_all_parallel():
    with Pool(min(len(ENVS), MAX_THREADS)) as p:
        p.map(run_union, ENVS)

def run():
    for env in ENVS:
        run_env_all(env)

def run_parallel():
    with Pool(min(len(ENVS), MAX_THREADS)) as p:
        p.map(run_env_all, ENVS)

# run()
# run_env_all(ILLUSTRATIVE, timeout=10)
