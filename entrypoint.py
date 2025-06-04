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

import pickle

def run_family_experiment_for_lineplot(project_path, num_nodes = 2, memory_model=None, timeout=None, max_iter=1000, seed=11):
    """
    Run experiment to collect the performance of rfPG (used throughout the experiments) and the random GA baselines. 
    """

    dr = f"{BASE_OUTPUT_DIR}/{project_path.split('/')[-1]}/seed{seed}/"
    os.makedirs(dr, exist_ok=True)
    results = {}

    if os.path.isfile(f"{dr}/gd-experiment.pickle"):
        if OVERWRITE_EXISTING_RESULTS:
            print("WHOLE FAMILY EXPERIMENT EXISTS FOR", project_path, seed, Method.GRADIENT.name.lower(), ". OVERWRITING!")
        else:
            print("WHOLE FAMILY EXPERIMENT EXISTS FOR", project_path, seed, Method.GRADIENT.name.lower(), ". SKIPPING!")
            return

    def store_results(gd : POMDPFamiliesSynthesis, seed : int, fsc : paynt.quotient.fsc.FSC, value : float):
        return {
            'family_trace' : gd.family_trace,
            'gd_trace' : gd.gd_trace,
            'current_values' : gd.current_values,
            'current_values_plot_times' : gd.current_values_plot_times,
            'plot_times' : gd.plot_times,
            'runtimes' : {
                'time_spent_grad' : gd.gd_time,
                'time_spent_grad_pmc_construction' : gd.pmc_construction_time,
                'time_spent_eval' : gd.paynt_time
            },
            'seed' : seed,
            'best_worst_value' : value,
            'fsc' : fsc
        }

    if memory_model is not None:
        timeout = (timeout - SAYNT_MEMORY_MODEL_SUBFAM_SIZE * SAYNT_MEMORY_MODEL_TIMEOUT)

    # Don't need to run below:

    # gd = POMDPFamiliesSynthesis(project_path, use_softmax=True, steps=1, learning_rate=0.01, use_momentum=False)
    # gd.run_gradient_descent_on_family(max_iter, num_nodes, timeout=timeout, memory_model=memory_model)

    # results['gd-no-momentum'] = store_results(gd, seed)

    gd = POMDPFamiliesSynthesis(project_path, use_softmax=True, steps=1, learning_rate=0.01, use_momentum=True, seed=seed)
    fsc, value = gd.run_gradient_descent_on_family(max_iter, num_nodes, timeout=timeout, memory_model=memory_model, random_selection=True)

    results['gd-random'] = store_results(gd, seed, fsc, value)

    with open(f"{dr}/gd-experiment.pickle", 'wb') as handle:
        pickle.dump(results, handle)

    gd = POMDPFamiliesSynthesis(project_path, use_softmax=True, steps=1, learning_rate=0.01, use_momentum=True, seed=seed)
    fsc, value = gd.run_gradient_descent_on_family(max_iter, num_nodes, timeout=timeout, memory_model=memory_model)

    results['gd-normal'] = store_results(gd, seed, fsc, value)

    results['memory_model'] = memory_model

    with open(f"{dr}/gd-experiment.pickle", 'wb') as handle:
        pickle.dump(results, handle)

def determine_memory_model_stratified(project_path, num_nodes = 2, seed=11, num_samples=5):
    gd = POMDPFamiliesSynthesis(project_path, use_softmax=True, steps=1, learning_rate=0.01, seed=seed)
    assignments, hole_combinations = gd.stratified_subfamily_sampling(num_samples, seed=seed)
    mem = gd.determine_memory_model_from_assignments(assignments, hole_combinations, max_num_nodes=num_nodes, timeout=SAYNT_MEMORY_MODEL_TIMEOUT)
    print(mem)
    return mem

def run_family(project_path, num_nodes = 2, memory_model = None, dynamic_memory=False):
    gd = POMDPFamiliesSynthesis(project_path, use_softmax=False, steps=10, dynamic_memory=dynamic_memory)
    gd.run_gradient_descent_on_family(1000, num_nodes, memory_model=memory_model)

def run_family_softmax(project_path, num_nodes = 2, memory_model = None, dynamic_memory=False, seed=11, max_iters=1000, **kwargs):
    gd = POMDPFamiliesSynthesis(project_path, use_softmax=True, steps=1, learning_rate=0.01, dynamic_memory=dynamic_memory, seed=seed)
    gd.run_gradient_descent_on_family(max_iters, num_nodes, memory_model=memory_model, **kwargs)    

def run_subfamily_for_heatmap(project_path, subfamily_size = 10, timeout = 60, num_nodes = 2, memory_model = None, baselines = [Method.GRADIENT, Method.SAYNT], seed=11, stratified=True, determine_memory_model=True, force_policy_deterministic=False):
    """
    Run experiments on the sampled subfamilies (baselines and rfPG-S).
    """
    dr = f"{BASE_OUTPUT_DIR}/{project_path.split('/')[-1]}/subfamsize{subfamily_size}/seed{seed}"
    os.makedirs(dr, exist_ok=True)

    gd = POMDPFamiliesSynthesis(project_path, use_softmax=True, steps=1, learning_rate=0.01, seed=seed)
    subfamily_size = min(subfamily_size, gd.pomdp_sketch.family.size)
    if stratified:
        subfamily_assigments, hole_combinations = gd.stratified_subfamily_sampling(subfamily_size, seed=seed)
    else:
        subfamily_assigments, hole_combinations = gd.create_random_subfamily(subfamily_size)

    if timeout:
        num_iters = int(1e30)
    else:
        num_iters = 1000

    if determine_memory_model:
        memory_model_subfamily_assignments, memory_model_hole_combinations = gd.stratified_subfamily_sampling(SAYNT_MEMORY_MODEL_SUBFAM_SIZE, seed=seed) if stratified else gd.create_random_subfamily(SAYNT_MEMORY_MODEL_SUBFAM_SIZE)
        memory_model = gd.determine_memory_model_from_assignments(memory_model_subfamily_assignments, memory_model_hole_combinations, max_num_nodes=num_nodes, timeout=SAYNT_MEMORY_MODEL_TIMEOUT)
        num_nodes = int(max(memory_model))
        gd_timeout = (timeout - SAYNT_MEMORY_MODEL_SUBFAM_SIZE * SAYNT_MEMORY_MODEL_TIMEOUT)
    else:
        gd_timeout = timeout

    standard_timeout = timeout

    for method in baselines:

        if os.path.isfile(f"{dr}/subfam-{method.name.lower()}.pickle"):
            if OVERWRITE_EXISTING_RESULTS:
                print("SUBFAMILY EXPERIMENT EXISTS FOR", project_path, seed, method.name.lower(), ". OVERWRITING!")
            else:
                print("SUBFAMILY EXPERIMENT EXISTS FOR", project_path, seed, method.name.lower(), ". SKIPPING!")
                continue

        sub_exp_timeout = (gd_timeout if method.value == Method.GRADIENT.value else standard_timeout) // subfamily_size

        subfamily_other_results = gd.experiment_on_subfamily(subfamily_assigments, hole_combinations, num_nodes, method, memory_model=memory_model, num_iters=num_iters, timeout=sub_exp_timeout, evaluate_on_whole_family=True, force_policy_deterministic=force_policy_deterministic)

        with open(f"{dr}/subfam-{method.name.lower()}.pickle", 'wb') as handle:
            pickle.dump(subfamily_other_results, handle)

    best_gd_fsc, subfamily_gd_best_value = gd.run_gradient_descent_on_family(num_iters, num_nodes, subfamily_assigments, timeout=gd_timeout, memory_model=memory_model)
    print(subfamily_gd_best_value)
    
    if force_policy_deterministic:
        best_gd_fsc.force_dirac_via_maxprob()

    dtmc_sketch = gd.get_dtmc_sketch(best_gd_fsc)

    our_evaluations = gd.get_values_on_subfamily(dtmc_sketch, subfamily_assigments)

    _, family_value = gd.paynt_call(dtmc_sketch)

    our_results = {
        'ours' : our_evaluations,
        'whole_family' : family_value,
        'fsc' : best_gd_fsc,
        'seed' : seed,
        'hole_combinations' : hole_combinations,
        'memory_model' : memory_model
    }

    print("OURS:", our_evaluations, 'family value:', family_value)
    with open(f"{dr}/subfam-ours.pickle", 'wb') as handle:
        pickle.dump(our_results, handle)
    
    return memory_model

def run_union(project_path, method=Method.SAYNT, timeout=10, num_assignments=5, num_nodes=2, stratified=True, seed=11, determine_memory_model = True):
    """
    Run experiments on the union POMDP (baselines).
    """
    dr = f"{BASE_OUTPUT_DIR}/{project_path.split('/')[-1]}/union/seed{seed}"
    os.makedirs(dr, exist_ok=True)
    
    if os.path.isfile(f"{dr}/union-{method.name.lower()}.pickle"):
        if OVERWRITE_EXISTING_RESULTS:
            print("UNION EXPERIMENT EXISTS FOR", project_path, seed, method.name.lower(), ". OVERWRITING!")
        else:
            print("UNION EXPERIMENT EXISTS FOR", project_path, seed, method.name.lower(), ". SKIPPING!")
            return
    
    gd : POMDPFamiliesSynthesis = POMDPFamiliesSynthesis(project_path, use_softmax=True, learning_rate=0.1, steps=1, use_momentum=True, union=True, seed=seed)

    if stratified:
        assignments, hole_combinations = gd.stratified_subfamily_sampling(num_assignments, seed=seed)
    else:
        assignments, hole_combinations = gd.create_random_subfamily(num_assignments)

    pomdps = []
    pomdp_maps = []
    for assignment in assignments:
        pomdp = gd.pomdp_sketch.build_pomdp(assignment).model
        # pomdp.
        # pomdp,true_action_map = gd.assignment_to_pomdp(gd.pomdp_sketch,assignment,restore_absorbing_states=False)
        pomdps.append(pomdp)
        # pomdp_maps.append(true_action_map)
        # print(true_action_map)

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
        fsc = gd.solve_pomdp_saynt(union_pomdp, gd.pomdp_sketch.specification.copy(), num_nodes, timeout=timeout)
    elif method == Method.PAYNT:
        fsc = gd.solve_pomdp_paynt(union_pomdp, gd.pomdp_sketch.specification.copy(), num_nodes, timeout=timeout)
    elif method == Method.GRADIENT:
        if determine_memory_model:
            memory_model_subfamily_assignments, memory_model_hole_combinations = gd.stratified_subfamily_sampling(SAYNT_MEMORY_MODEL_SUBFAM_SIZE, seed=seed) if stratified else gd.create_random_subfamily(SAYNT_MEMORY_MODEL_SUBFAM_SIZE)
            memory_model = gd.determine_memory_model_from_assignments(memory_model_subfamily_assignments, memory_model_hole_combinations, max_num_nodes=num_nodes, timeout=SAYNT_MEMORY_MODEL_TIMEOUT)
            num_nodes = int(max(memory_model))
            timeout = (timeout - SAYNT_MEMORY_MODEL_SUBFAM_SIZE * SAYNT_MEMORY_MODEL_TIMEOUT)
        else:
            memory_model = None
        value, resolution, action_function_params, memory_function_params, *_ = gd.gradient_descent_on_single_pomdp(union_pomdp, int(1e30) if timeout else 1000, num_nodes, timeout=timeout, parameter_resolution={}, resolution={}, action_function_params={}, memory_function_params={}, memory_model=memory_model)
        fsc = gd.parameters_to_paynt_fsc(action_function_params, memory_function_params, resolution, num_nodes, gd.nO, gd.pomdp_sketch.observation_to_actions)
    else:
        raise ValueError(f"Method unknown: {method}")
    
    if fsc is None:
        # Run failed, store arbitrary values.
        undefined = (math.inf if gd.minimizing else -math.inf)
        assignment_values = {j : undefined for j in range(len(hole_combinations))}
        family_value = undefined
    else:
        # the last observation is the fresh one for the fresh initial state
        # get the initial memory node set from this fresh initial state
        assert fsc.num_observations == gd.pomdp_sketch.num_observations+1
        if fsc.is_deterministic:
            initial_node = fsc.update_function[0][-1]
        else:
            possible_initial_nodes = list(fsc.update_function[0][-1].keys())
            assert len(possible_initial_nodes) == 1
            initial_node = possible_initial_nodes[0]
            assert initial_node == 0

        # get rid of the fresh observation
        for node in range(fsc.num_nodes):
            fsc.action_function[node] = fsc.action_function[node][:-1]
            fsc.update_function[node] = fsc.update_function[node][:-1]
        fsc.num_observations -= 1

        if fsc.is_deterministic:
            # fix possibly dummy actions
            # for node in range(fsc.num_nodes):
            #     for obs in range(fsc.num_observations):
            #         action = fsc.action_function[node][obs]
            #         if action is None:
            #             continue
            #         action_label = fsc.action_labels[action]
            #         true_action_label = observation_action_to_true_action[obs][action_label]
            #         fsc.action_function[node][obs] = fsc.action_labels.index(true_action_label)

            # fill actions for unreachable observations
            for node in range(fsc.num_nodes):
                for obs in range(fsc.num_observations):
                    if fsc.action_function[node][obs] is None:
                        available_action = gd.pomdp_sketch.observation_to_actions[obs][0]
                        available_action_label = gd.pomdp_sketch.action_labels[available_action]
                        fsc.action_function[node][obs] = fsc.action_labels.index(available_action_label)

            # make 0 the initial node and reorder the actions
            node_order = list(range(fsc.num_nodes))
            node_order[0] = initial_node
            node_order[initial_node] = 0
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
        'fsc' : fsc,
        'seed' : seed,
        'hole_combinations' : hole_combinations
    }

    print(project_path.split('/')[-1], "Assignments:", assignment_values, 'family value:', family_value)
    with open(f"{dr}/union-{method.name.lower()}.pickle", 'wb') as handle:
        pickle.dump(results, handle)


def run_lineplot_experiment(env, seed=SEED, num_samples=SAYNT_MEMORY_MODEL_SUBFAM_SIZE, num_nodes=MAX_NUM_NODES, timeout=TIMEOUT):
    try:
        memory_model = determine_memory_model_stratified(env, num_nodes=num_nodes, num_samples=num_samples, seed=seed)
        run_family_experiment_for_lineplot(env, max(memory_model), memory_model, max_iter=MAX_ITER, timeout=timeout, seed=seed)
    except Exception as e:
        print("FULL FAMILY GRADIENT DESCENT EXPERIMENT FAILED FOR", env, seed)
        print(e)
        print(traceback.format_exc())

def run_heatmap_experiment(env, seed=SEED, num_samples=SUBFAMILY_SIZE, num_nodes=MAX_NUM_NODES, timeout=TIMEOUT):
    try:
        run_subfamily_for_heatmap(env, timeout=timeout, subfamily_size=num_samples, baselines=[Method.SAYNT, Method.GRADIENT], num_nodes=num_nodes, determine_memory_model=True, stratified=True, seed=seed)
        # run_subfamily_for_heatmap(env, timeout=timeout, subfamily_size=num_samples, baselines=[Method.GRADIENT], num_nodes=num_nodes, determine_memory_model=True, stratified=True, seed=seed)
    except Exception as e:
        print("SUBFAMILY EXPERIMENT FAILED FOR", env, seed)
        print(e)
        print(traceback.format_exc())

def run_union_experiment(env, seed=SEED, num_samples=SUBFAMILY_SIZE, num_nodes=MAX_NUM_NODES, timeout=TIMEOUT, method=Method.SAYNT):
    try:
        run_union(env, method=method, timeout=timeout, num_assignments=num_samples, stratified=True, seed=seed, num_nodes=num_nodes)
    except Exception as e:
        print("UNION EXPERIMENT FAILED FOR", env, seed, method)
        print(e)
        print(traceback.format_exc())

def run_env_all(env):
    run_lineplot_experiment(env)
    run_heatmap_experiment(env)
    run_union_experiment(env)

def run_union_all():
    for env in ENVS:
        run_union(env)

def run_union_all_parallel():
    with multiprocessing.Pool(min(len(ENVS), MAX_THREADS)) as p:
        p.map(run_union, ENVS)

def run():
    for env in ENVS:
        run_env_all(env)

def run_parallel():
    with multiprocessing.Pool(min(len(ENVS), MAX_THREADS)) as p:
        p.map(run_env_all, ENVS)

def run_extreme(f, env, seed):
    f(env, seed=seed)

def run_specific():
    threads = 2
    assert threads <= multiprocessing.cpu_count()
    assert threads <= MAX_THREADS
    tasks = itertools.product(
        [
            run_lineplot_experiment,
            # run_heatmap_experiment,
            # functools.partial(run_union_experiment, method=Method.SAYNT),
            # functools.partial(run_union_experiment, method=Method.GRADIENT)
        ],
        [DPM, AVOID],
        [11]
    )

    with multiprocessing.Pool(threads) as p:
        p.starmap(run_extreme, tasks)

def run_parallel_extreme_large():
    assert MAX_THREADS <= multiprocessing.cpu_count()
    tasks = itertools.product(
        [
            run_lineplot_experiment,
            run_heatmap_experiment,
            functools.partial(run_union_experiment, method=Method.SAYNT),
            functools.partial(run_union_experiment, method=Method.GRADIENT)
        ],
        ENVS,
        range(2,12)
    )

    with multiprocessing.Pool(MAX_THREADS) as p:
        p.starmap(run_extreme, tasks)


def run_parallel_extreme_large_separate_calls():
    assert MAX_THREADS <= multiprocessing.cpu_count()
    tasks = itertools.product(
        [
            # run_lineplot_experiment,
            run_heatmap_experiment,
            # functools.partial(run_union_experiment, method=Method.SAYNT),
            # functools.partial(run_union_experiment, method=Method.GRADIENT)
        ],
        ENVS,
        range(2,12)
    )

    with multiprocessing.Pool(MAX_THREADS) as p:
        p.starmap(run_extreme, tasks)

    tasks = itertools.product(
        [
            # run_lineplot_experiment,
            # run_heatmap_experiment,
            functools.partial(run_union_experiment, method=Method.SAYNT),
            functools.partial(run_union_experiment, method=Method.GRADIENT)
        ],
        ENVS,
        range(2,12)
    )

    with multiprocessing.Pool(MAX_THREADS) as p:
        p.starmap(run_extreme, tasks)

# run_subfamily_for_heatmap(ILLUSTRATIVE, timeout=EXAMPLE_TIMEOUT, subfamily_size=EXAMPLE_SUBFAMILY_SIZE, baselines=[Method.GRADIENT], num_nodes=2, determine_memory_model=False, stratified=True, seed=EXAMPLE_SEED, force_policy_deterministic=False)
run_parallel_extreme_large()

