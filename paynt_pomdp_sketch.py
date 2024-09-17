# using Paynt for POMDP sketches

import paynt.cli
import paynt.parser.sketch
import paynt.quotient.pomdp_family
import paynt.quotient.fsc
import paynt.synthesizer.synthesizer_onebyone
import paynt.synthesizer.synthesizer_ar

import stormpy
import stormpy.pomdp
import stormpy.pars
import pycarl
import stormpy.info

if stormpy.info.storm_ratfunc_use_cln():
    import pycarl.cln as pc
else:
    import pycarl.gmp as pc

import numpy as np

import payntbind

import os
import gc
import random
import cProfile, pstats

def load_sketch(project_path):
    project_path = os.path.abspath(project_path)
    sketch_path = os.path.join(project_path, "sketch.templ")
    properties_path = os.path.join(project_path, "sketch.props")    
    pomdp_sketch = paynt.parser.sketch.Sketch.load_sketch(sketch_path, properties_path)
    return pomdp_sketch

def assignment_to_pomdp(pomdp_sketch, assignment):
    pomdp = pomdp_sketch.build_pomdp(assignment).model
    updated = payntbind.synthesis.restoreActionsInAbsorbingStates(pomdp)
    if updated is not None: pomdp = updated
    action_labels,_ = payntbind.synthesis.extractActionLabels(pomdp)
    num_actions = len(action_labels)
    pomdp,choice_to_true_action = payntbind.synthesis.enableAllActions(pomdp)
    observation_action_to_true_action = [None]* pomdp.nr_observations
    for state in range(pomdp.nr_states):
        obs = pomdp.observations[state]
        if observation_action_to_true_action[obs] is not None:
            continue
        observation_action_to_true_action[obs] = [None] * num_actions
        choice_0 = pomdp.transition_matrix.get_row_group_start(state)
        for action in range(num_actions):
            choice = choice_0+action
            true_action = choice_to_true_action[choice]
            observation_action_to_true_action[obs][action] = true_action
    return pomdp,observation_action_to_true_action

def solve_pomdp_paynt(pomdp_quotient, specification, k, timeout=1):
    pomdp_quotient.set_imperfect_memory_size(k)
    print(dir(pomdp_quotient), pomdp_quotient.action_labels_at_observation[0])#, pomdp_quotient.action_labels)
    synthesizer = paynt.synthesizer.synthesizer_ar.SynthesizerAR(pomdp_quotient)
    assignment = synthesizer.synthesize(timeout=timeout)
    assert assignment is not None
    fsc = pomdp_quotient.assignment_to_fsc(assignment)
    return fsc

def parameters_to_paynt_fsc(action_function_params, memory_function_params, resolution, num_nodes, num_obs, observation_to_actions):
    fsc = paynt.quotient.fsc.FSC(num_nodes, num_obs)
    
    for (n,o,a), var in action_function_params.items():
        if var in resolution:
            prob = float(resolution[var])
        else:
            if isinstance(var, pycarl.cln.cln.Rational):
                prob = float(var)
            else:
                prob = float(var.evaluate(resolution))
        
        if prob == 0:
            continue

        # print(n,o,a,prob)

        if fsc.action_function[n][o] is None:
            fsc.action_function[n][o] = {a : prob}
        else:
            fsc.action_function[n][o].update({a : prob})

    for (n,o,m), var in memory_function_params.items():
        if var in resolution:
            prob = float(resolution[var])
        else:
            if isinstance(var, pycarl.cln.cln.Rational):
                prob = float(var)
            else:
                prob = float(var.evaluate(resolution))

        # print(n,o,m,prob)
        if prob == 0:
            continue

        if fsc.update_function[n][o] is None:
            fsc.update_function[n][o] = {m : prob}
        else:
            fsc.update_function[n][o].update({m : prob})
    
    fsc.action_function = [[d if d is not None else {a : 1/len(observation_to_actions[o]) for a in observation_to_actions[o]} for o, d in enumerate(o_to_act)] for o_to_act in fsc.action_function]
    fsc.update_function = [[d if d is not None else {m : 1/num_nodes for m in range(num_nodes)} for d in o_to_mem ] for o_to_mem in fsc.update_function]
    
    return fsc

def solve_pomdp_saynt(pomdp, specification, k, timeout=1):
    pomdp_quotient = paynt.quotient.pomdp.PomdpQuotient(pomdp, specification)
    timeout = timeout
    paynt_iter_timeout = 5
    storm_iter_timeout = 2
    iterative_storm = (timeout,paynt_iter_timeout,storm_iter_timeout)
    storm_control = paynt.quotient.storm_pomdp_control.StormPOMDPControl()
    storm_control.set_options(
        storm_options="cutoff", get_storm_result=None, iterative_storm=iterative_storm, use_storm_cutoffs=False,
        unfold_strategy_storm="storm", prune_storm=False, export_fsc_storm=None, export_fsc_paynt=None
    )
    synthesizer = paynt.synthesizer.synthesizer.Synthesizer.choose_synthesizer(
        pomdp_quotient, method="ar", fsc_synthesis=True, storm_control=storm_control
    )
    synthesizer.run(optimum_threshold=None)
    assignment = synthesizer.storm_control.latest_paynt_result
    assert assignment is not None
    fsc = pomdp_quotient.assignment_to_fsc(assignment)
    return fsc

def random_fsc(pomdp_sketch, num_nodes):
    num_obs = pomdp_sketch.num_observations
    fsc = paynt.quotient.fsc.FSC(num_nodes, num_obs)
    # action function if of type NxZ -> Distr(Act)
    for n in range(num_nodes):
        for z in range(num_obs):
            actions = pomdp_sketch.observation_to_actions[z]
            fsc.action_function[n][z] = { action:1/len(actions) for action in actions }
    # memory update function is of type NxZ -> Distr(N) and is posterior-aware
    # note: this is currently inconsistent with definitions in paynt.quotient.fsc.FSC, but let's see how this works
    for n in range(num_nodes):
        for z in range(num_obs):
            actions = pomdp_sketch.observation_to_actions[z]
            fsc.update_function[n][z] = { n_new:1/num_nodes for n_new in range(num_nodes) }
    return fsc


def construct_pmc(pomdp, pomdp_sketch, reward_model_name, num_nodes, initial_probability, action_function_params = {}, memory_function_params = {}, distinct_parameters_for_final_probability = False, sanity_checks = True):
    pycarl.clear_pools()
    
    builder : stormpy.storage.storage.ParametricSparseMatrixBuilder = stormpy.storage.ParametricSparseMatrixBuilder()
    
    counter = 0
    
    seen = set()
    
    pmc_transitions = {}
    
    rewards = {}
    
    # print(pomdp.reward_models)
    reward_model = pomdp.reward_models[reward_model_name]
    assert not reward_model.has_state_rewards
    assert reward_model.has_state_action_rewards
    state_action_rewards = reward_model.state_action_rewards
    
    rewards = {}
    
    denom = pc.FactorizedPolynomial(pc.Rational(1))
    
    ndi = pomdp.nondeterministic_choice_indices
    
    labels = pomdp.labeling
    
    states = set()
    labeling = {l : [] for l in labels.get_labels()}
    
    target_label = pomdp_sketch.get_property().get_target_label()
    
    for state in pomdp.states:
        s = state.id
        o = pomdp.observations[s]
        for action in pomdp.states[s].actions:
            a = action.id
            quotient_action = pomdp_sketch.observation_to_actions[o][a]
            choice = ndi[s]+a
            reward = state_action_rewards[choice]
            for n in range(num_nodes):
                states.add(s * num_nodes + n)
                if s * num_nodes + n not in pmc_transitions:
                    pmc_transitions[s * num_nodes + n] = {}
                for label in state.labels:
                    labeling[label].append(s * num_nodes + n)

                for transition in action.transitions:
                    t = transition.column
                    t_prob = transition.value()
                    for m in range(num_nodes):
                        states.add(t * num_nodes + m)
                        act_tup = (n, o, quotient_action)
                        mem_tup = (n, o, m)
                        
                        assert (s * num_nodes + n, t * num_nodes + m, a) not in seen, (s * num_nodes + n, t * num_nodes + m, seen)
                        seen.add((s * num_nodes + n, t * num_nodes + m, a))
                        
                        if act_tup in action_function_params:
                            act_param = action_function_params[act_tup]
                        else:
                            action_ids = [b.id for b in pomdp.states[s].actions]
                            quotient_actions = pomdp_sketch.observation_to_actions[o]
                            if not distinct_parameters_for_final_probability and a == max(action_ids):
                                act_param = pc.Rational(1)
                                for a_ in quotient_actions:
                                    if quotient_action != a_:
                                        assert (n, o, a_) in action_function_params
                                        act_param -= action_function_params[n, o, a_]
                            else:
                                p_a_name = f"p{counter}_n{n}_o{o}_a{quotient_action}"
                                assert pycarl.variable_with_name(p_a_name).is_no_variable, (p_a_name, action_function_params)
                                act_param = pycarl.Variable(p_a_name)
                                counter += 1
                                
                            action_function_params[act_tup] = act_param
                                

                        if mem_tup in memory_function_params:
                            mem_param = memory_function_params[mem_tup]
                        else:
                            if not distinct_parameters_for_final_probability and m == num_nodes-1:
                                mem_param = pc.Rational(1)
                                for m_ in range(num_nodes):
                                    if m != m_:
                                        assert (n, o, m_) in memory_function_params
                                        mem_param -= memory_function_params[n, o, m_]
                                memory_function_params[mem_tup] = mem_param
                            else:
                                p_o_name = f"p{counter}_n{n}_o{o}_m{m}"
                                assert pycarl.variable_with_name(p_o_name).is_no_variable, (p_o_name, action_function_params)
                                mem_param = pycarl.Variable(p_o_name)
                                memory_function_params[mem_tup] = mem_param
                                counter += 1

                        action_poly = pc.Polynomial(act_param)
                        mem_poly = pc.Polynomial(mem_param)
                        action_mem_poly = action_poly * mem_poly * pc.Rational(float(t_prob))

                        if (t * num_nodes + m) in pmc_transitions[s * num_nodes + n]:
                            pmc_transitions[(s * num_nodes + n)][(t * num_nodes + m)] += action_mem_poly
                        else:
                            pmc_transitions[(s * num_nodes + n)][(t * num_nodes + m)] = action_mem_poly

                if (s * num_nodes + n) in rewards:
                    rewards[s * num_nodes + n] += pc.Polynomial(action_function_params[(n, o, quotient_action)]) * pc.Rational(float(reward))
                else:
                    rewards[s * num_nodes + n] = pc.Polynomial(action_function_params[(n, o, quotient_action)]) * pc.Rational(float(reward))
    
    resolution = {expr : pc.Rational(initial_probability) for key, expr in action_function_params.items() if isinstance(expr, pycarl.Variable)}
    resolution.update({
        expr : pc.Rational(initial_probability) for key, expr in memory_function_params.items() if isinstance(expr, pycarl.Variable)
    })

    for s, next_states in sorted(pmc_transitions.items(), key = lambda x : x[0]):
        # BELOW EATS MEMORY
        # assert all([float(pc.FactorizedRationalFunction(pc.FactorizedPolynomial(probability_function, pycarl.cln.cln._FactorizationCache()), denom).evaluate(resolution)) > 0 and float(pc.FactorizedRationalFunction(pc.FactorizedPolynomial(probability_function, pycarl.cln.cln._FactorizationCache()), denom).evaluate(resolution)) <= 1 for probability_function in pmc_transitions[s].values()])
        # assert np.isclose(sum([float(pc.FactorizedRationalFunction(pc.FactorizedPolynomial(probability_function, pycarl.cln.cln._FactorizationCache()), denom).evaluate(resolution)) for probability_function in pmc_transitions[s].values()]), 1)
        rewards[s] = pc.FactorizedRationalFunction(pc.FactorizedPolynomial(rewards[s], pycarl.cln.cln._FactorizationCache()), denom)
        for t, probability_function in sorted(next_states.items(), key = lambda x : x[0]):
            parametric_transition = pc.FactorizedRationalFunction(pc.FactorizedPolynomial(probability_function, pycarl.cln.cln._FactorizationCache()), denom)
            evaluation = float(parametric_transition.evaluate(resolution))
            builder.add_next_value(s, t, parametric_transition)
            assert parametric_transition.evaluate(resolution) > 0 and parametric_transition.evaluate(resolution) <= 1
            assert probability_function.evaluate(resolution) > 0 and probability_function.evaluate(resolution) <= 1

    del pmc_transitions
    
    gc.collect()
    
    print("Building pDTMC transition matrix:")
    p_matrix = builder.build()
    print("Done.")

    for s in states:
        row = p_matrix.get_row(s)
        for entry in row:
            assert entry.value().evaluate(resolution) > 0, ()
    
    def print_params(params_dict : dict) -> None:
        for key, var in sorted(params_dict.items()):
            print(key, var)
    
    labelling = stormpy.storage.StateLabeling(len(states))
    
    for label, states in labeling.items():
        labelling.add_label(label)
        for s in states:
            labelling.add_label_to_state(label, s)


    pmc_reward_model = stormpy.storage.SparseParametricRewardModel(optional_state_reward_vector=[r for r in rewards.values()])
    
    del rewards
    
    components = stormpy.storage.SparseParametricModelComponents(p_matrix, labelling, reward_models={reward_model_name : pmc_reward_model})
    
    del p_matrix
    
    pmc = stormpy.storage.SparseParametricDtmc(components)
    
    return pmc, action_function_params, memory_function_params, resolution


def main():
    profiling = True
    if profiling:
        profiler = cProfile.Profile()
        profiler.enable()

    random.seed(11)

    # enable PAYNT logging
    paynt.cli.setup_logger()

    # load sketch
    # project_path="models/pomdp/sketches/obstacles-10-2"
    project_path="models/pomdp/sketches/avoid"
    pomdp_sketch = load_sketch(project_path)
    
    nO = pomdp_sketch.num_observations
    
    reward_model_name = pomdp_sketch.get_property().get_reward_name()

    # construct POMDP from the given hole assignment
    hole_assignment = pomdp_sketch.family.pick_any()
    # pomdp,observation_action_to_true_action = assignment_to_pomdp(pomdp_sketch,hole_assignment)
    
    pomdp_class = pomdp_sketch.build_pomdp(hole_assignment)
    pomdp = pomdp_class.model
    
    print("|O| =", pomdp_sketch.num_observations, "|A| =", pomdp_sketch.num_actions)
    
    formula = pomdp_sketch.get_property().property.raw_formula
    
    Rmin = pomdp_sketch.get_property().minimizing
    
    print(Rmin)
    
    print(formula, pomdp_sketch.get_property().formula)
    
    
    num_nodes = 2
    
    results = {}
    
    for i, hole_combination in enumerate(pomdp_sketch.family.all_combinations()):
        # print(hole_combination)
        assignment = pomdp_sketch.family.construct_assignment(hole_combination)
        pomdp = pomdp_sketch.build_pomdp(assignment).model
        pomdp_quotient = paynt.quotient.pomdp.PomdpQuotient(pomdp, pomdp_sketch.specification)
        # current_pomdp = 
        fsc = solve_pomdp_paynt(pomdp_quotient, pomdp_sketch.specification, num_nodes, timeout=2)
        print(fsc.action_function, pomdp_sketch.observation_to_actions, pomdp_quotient.action_labels_at_observation, pomdp_sketch.action_labels)

        for n in range(num_nodes):
            for o in range(len(fsc.action_function[n])):
                if len(pomdp_quotient.action_labels_at_observation[o]) == 1: continue
                a = fsc.action_function[n][o]
                strings = pomdp_quotient.action_labels_at_observation[o]
                print(a, strings, pomdp_sketch.action_labels, pomdp_quotient.actions_at_observation[o])
                print(dir(pomdp_quotient))
                exit()
                
                
        # print([pomdp_sketch.action_labels.index(action_string) for action_string in ])
        fsc.action_function = [[pomdp_sketch.observation_to_actions[o][fsc.action_function[n][o]] for o in range(len(fsc.action_function[n]))] for n in range(num_nodes)]
        dtmc_sketch = pomdp_sketch.build_dtmc_sketch(fsc, negate_specification=True)
        one_by_one = paynt.synthesizer.synthesizer_onebyone.SynthesizerOneByOne(dtmc_sketch)
        evaluation = one_by_one.evaluate_all(dtmc_sketch, formula, keep_value_only=True)
        print(evaluation)
        results[hole_assignment] = (fsc, evaluation)
        
    print(results)
    
    task = stormpy.ParametricCheckTask(pomdp_sketch.get_property().formula, only_initial_states=False)
    
    # print("form:", formula, type(formula), formula.comparison_type, formula.threshold)
    
    storm_pmc_construction = False
    
    max_actions_per_observation = max([len(pomdp_sketch.observation_to_actions[o]) for o in range(nO)])

    # fsc = solve_pomdp_paynt(pomdp, pomdp_sketch.specification, num_nodes)
    
    initial_probability = 1 / max(max_actions_per_observation, num_nodes)
    initial_probability = 0.1
    
    print(initial_probability)
    # exit()
    
    use_softmax = False
    
    if storm_pmc_construction:
        
        pomdp = pomdp_sketch.build_pomdp(hole_assignment).model
        
        # pomdp = stormpy.pomdp.make_simple(pomdp, keep_state_valuations=True)
        # initial_probability = 0.5
        
        memory_builder = stormpy.pomdp.PomdpMemoryBuilder()
        memory = memory_builder.build(stormpy.pomdp.PomdpMemoryPattern.full, num_nodes)
        pomdp = stormpy.pomdp.unfold_memory(pomdp, memory, add_memory_labels=True, keep_state_valuations=True)
        pmc : stormpy.storage.storage.SparseParametricDtmc = stormpy.pomdp.apply_unknown_fsc(pomdp, stormpy.pomdp.PomdpFscApplicationMode.simple_linear)
        parameters : set = pmc.collect_all_parameters()
        print(f"There are currently {len(parameters)} parameters in the pMC!")
        
        resolution = {
            p : pc.Rational(initial_probability) for p in parameters
        }
        
    else:
        
        pmc, action_function_params, memory_function_params, resolution = construct_pmc(pomdp, pomdp_sketch, reward_model_name, num_nodes, initial_probability, distinct_parameters_for_final_probability=use_softmax)
        
        action_function_params_no_const = {index : var for index, var in action_function_params.items() if isinstance(var, pycarl.Variable)}
        memory_function_params_no_const = {index : var for index, var in memory_function_params.items() if isinstance(var, pycarl.Variable)}
        for p in pmc.collect_all_parameters():
            assert p in action_function_params_no_const.values() or p in memory_function_params_no_const.values()

    # memory_builder = stormpy.pomdp.PomdpMemoryBuilder()
    # memory = memory_builder.build(stormpy.pomdp.PomdpMemoryPattern.selective_counter, 3)
    # pomdp = stormpy.pomdp.unfold_memory(pomdp, memory, add_memory_labels=True, keep_state_valuations=True)
    # pmc : stormpy.storage.storage.SparseParametricDtmc = stormpy.pomdp.apply_unknown_fsc(pomdp, stormpy.pomdp.PomdpFscApplicationMode.standard)
    
    print(pmc)
    instantiator = stormpy.pars.PDtmcInstantiator(pmc)
    parameters : set = pmc.collect_all_parameters()
    
    if not use_softmax:
        for s in pmc.states:
            for action in s.actions:
                summation = 0
                for transition in action.transitions:
                    valuation = transition.value().evaluate(resolution)
                    assert valuation > 0 and valuation <= 1, (valuation, str(transition.value()), float(valuation))
                    summation += valuation
                    # exit()
                assert np.isclose(float(summation), 1)
    
    print(f"There are currently {len(parameters)} parameters in the pMC!")
    env = stormpy.Environment()
    instantiated_model = instantiator.instantiate(resolution)
    result = stormpy.model_checking(instantiated_model, formula)
    current_dtmc_value = result.as_explicit_quantitative().at(0)
    print("RESULT:", result, current_dtmc_value, type(result))
    
    checker = payntbind.synthesis.SparseDerivativeInstantiationModelCheckerFamily(pmc) 
    checker.specifyFormula(env, task)
    
    wrapper = payntbind.synthesis.GradientDescentInstantiationSearcherFamily(pmc)
    # synth_task = payntbind.synthesis.FeasibilitySynthesisTask(formula)
    # synth_task.set_bound(formula.comparison_type, formula.threshold_expr)
    # wrapper.setup(env, synth_task)
    # wrapper.gradientDescent()
    
    def sign(x):
        if np.isclose(x, 0):
            return 0
        else:
            return 1 if x > 0 else -1
        
    num_iters = 1000
    
    for i in range(num_iters):  
        
        if i % 10 == 0:
            if i > 0:
                fsc = parameters_to_paynt_fsc(action_function_params, memory_function_params, resolution, num_nodes, nO, pomdp_sketch.observation_to_actions)
                # print(fsc)
                dtmc_sketch =  pomdp_sketch.build_dtmc_sketch(fsc, negate_specification=True)
                synthesizer = paynt.synthesizer.synthesizer_ar.SynthesizerAR(dtmc_sketch)
                hole_assignment = synthesizer.run()
                # hole_assignment = pomdp_sketch.family.pick_random()
            else:
                hole_assignment = pomdp_sketch.family.pick_any()
            # print(hole_assignment)
            pomdp_class = pomdp_sketch.build_pomdp(hole_assignment)
            pomdp = pomdp_class.model
            pmc, action_function_params, memory_function_params, _ = construct_pmc(pomdp, pomdp_sketch, reward_model_name, num_nodes, initial_probability, action_function_params=action_function_params, memory_function_params=memory_function_params)
            # print(action_function_params)
            instantiator = stormpy.pars.PDtmcInstantiator(pmc)
            parameters = pmc.collect_all_parameters()
            action_function_params_no_const = {index : var for index, var in action_function_params.items() if isinstance(var, pycarl.Variable)}
            memory_function_params_no_const = {index : var for index, var in memory_function_params.items() if isinstance(var, pycarl.Variable)}
            for p in pmc.collect_all_parameters():
                assert p in action_function_params_no_const.values() or p in memory_function_params_no_const.values()
        
        instantiated_model = instantiator.instantiate(resolution)
        result = stormpy.model_checking(instantiated_model, formula)
        current_dtmc_value = result.as_explicit_quantitative().at(0)
        print(f"I{i}. POMDP={hole_assignment}. RESULT:", current_dtmc_value)

        new_resolution = {}
        
        lr = 0.001

        for p in parameters:
            try:
                res = checker.check(env, resolution, p)
            except Exception as e:
                fsc = parameters_to_paynt_fsc(action_function_params, memory_function_params, resolution, num_nodes, nO, pomdp_sketch.observation_to_actions)
                print(fsc.action_function)
                print(fsc.update_function)
                raise e
            if Rmin:
                update = float(resolution[p]) - lr * sign(res.at(0))
            else:
                update = float(resolution[p]) + lr * sign(res.at(0))
            corrected = pc.Rational(min(max(update, 0), 1))
            # print(update, float(corrected))
            assert corrected >= 0 and corrected <= 1
            new_resolution[p] = corrected
            
            assert new_resolution[p] >= 0

        resolution = new_resolution
        
    fsc = parameters_to_paynt_fsc(action_function_params, memory_function_params, resolution, num_nodes, nO, pomdp_sketch.observation_to_actions)
    
    dtmc_sketch =  pomdp_sketch.build_dtmc_sketch(fsc, negate_specification=True)
    
    print(fsc.action_function)
    print(fsc.update_function)
    
    print(synthesizer.run())


    if profiling:
        profiler.disable()
        stats = profiler.create_stats()
        pstats.Stats(profiler).sort_stats('tottime').print_stats(10)


main()