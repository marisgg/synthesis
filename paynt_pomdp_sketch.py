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


def construct_pmc(pomdp, pomdp_sketch, reward_model_name, num_nodes, initial_probability):
    pycarl.clear_pools()
    
    builder : stormpy.storage.storage.ParametricSparseMatrixBuilder = stormpy.storage.ParametricSparseMatrixBuilder()
    
    counter = 0
    
    action_function_params = {}
    memory_function_params = {}
    
    seen = set()
    
    pmc_transitions = {}
    
    rewards = {}
    
    # print(pomdp.reward_models)
    reward_model = pomdp.reward_models[reward_model_name]
    assert not reward_model.has_state_rewards
    state_action_rewards = reward_model.state_action_rewards
    # print(state_action_rewards)
    state_action_rewards_vector = np.array([x for x in state_action_rewards])
    
    rewards = {}
    
    denom = pc.FactorizedPolynomial(pc.Rational(1))
    
    ndi = pomdp.nondeterministic_choice_indices
    
    labels = pomdp.labeling
    
    states = set()
    labeling = {l : [] for l in labels.get_labels()}
    
    target_label = pomdp_sketch.get_property().get_target_label()
    
    # print("Memory function parameters:", num_nodes * pomdp.nr_observations * num_nodes)
    
    print(pomdp_sketch.action_labels)
    
    print(pomdp_sketch.choice_to_action)
    print(pomdp_sketch.observation_to_actions)
    # exit()
    
    for state in pomdp.states:
        s = state.id
        o = pomdp.observations[s]
        for action in pomdp.states[s].actions:
            a = action.id
            quotient_action = pomdp_sketch.observation_to_actions[o][a]
            choice = ndi[s]+a
            reward = state_action_rewards[choice]
            for transition in action.transitions:
                t = transition.column
                t_prob = transition.value()
                # print(t_prob, type(t_prob))
                for n in range(num_nodes):
                    # if len(state.labels) > 0:
                    states.add(s * num_nodes + n)
                    if s * num_nodes + n not in pmc_transitions:
                        pmc_transitions[s * num_nodes + n] = {}
                    for label in state.labels:
                        labeling[label].append(s * num_nodes + n)
                    # if target_label in state.labels:
                        # goal_states.append(s * num_nodes + n)
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
                            # if len(action_ids) > 1:
                            if a == max(action_ids):
                                # n, o, a = None
                                act_param = pc.Rational(1)
                                for a_ in quotient_actions:
                                    if quotient_action != a_:
                                        assert (n, o, a_) in action_function_params
                                        act_param -= action_function_params[n, o, a_]
                                # print("Var:", var, a, a_, [b for b in pomdp.states[s].actions][-1].id)
                            else:
                                p_a_name = f"p{counter}_n{n}_o{o}_a{quotient_action}"
                                assert pycarl.variable_with_name(p_a_name).is_no_variable, (p_a_name, action_function_params)
                                act_param = pycarl.Variable(p_a_name)
                                counter += 1
                            # else:
                                # act_
                                
                            action_function_params[act_tup] = act_param
                                

                        if mem_tup in memory_function_params:
                            mem_param = memory_function_params[mem_tup]
                        else:
                            if m == num_nodes-1:
                                # n, o, a = None
                                mem_param = pc.Rational(1)
                                for m_ in range(num_nodes):
                                    if m != m_:
                                        assert (n, o, m_) in memory_function_params
                                        mem_param -= memory_function_params[n, o, m_]
                                # print("Var:", mem_param, m, n, o, m_)
                                # exit()
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

                    if s * num_nodes + n in rewards:
                        rewards[s * num_nodes + n] += pc.Polynomial(action_function_params[(n, o, quotient_action)]) * pc.Rational(float(reward))
                    else:
                        rewards[s * num_nodes + n] = pc.Polynomial(action_function_params[(n, o, quotient_action)]) * pc.Rational(float(reward))
    
    resolution = {expr : pc.Rational(initial_probability) for key, expr in action_function_params.items() if isinstance(expr, pycarl.Variable)}
    resolution.update({
        expr : pc.Rational(initial_probability) for key, expr in memory_function_params.items() if isinstance(expr, pycarl.Variable)
    })
    
    # print(resolution)
    
    # print("----")
    for s, next_states in sorted(pmc_transitions.items(), key = lambda x : x[0]):
        for t, probability_function in sorted(next_states.items(), key = lambda x : x[0]):
            parametric_transition = pc.FactorizedRationalFunction(pc.FactorizedPolynomial(probability_function, pycarl.cln.cln._FactorizationCache()), denom)
            evaluation = float(parametric_transition.evaluate(resolution))
            builder.add_next_value(s, t, parametric_transition)
            # if s == 87 and t == 15:
                # print(float(parametric_transition.evaluate(resolution)))
                # print(float(probability_function.evaluate(resolution)))
            assert parametric_transition.evaluate(resolution) > 0 and parametric_transition.evaluate(resolution) <= 1
            assert probability_function.evaluate(resolution) > 0 and probability_function.evaluate(resolution) <= 1
            # print(s, t, evaluation)
            # print(help(probability_function))
            # exit()
    # print("----")
    
    # del pmc_transitions
    
    print("Building pDTMC transition matrix:")
    p_matrix = builder.build()
    print("Done.")
    
    # print(pmc_transitions)
    # exit()
    
    for s in states:
        assert all([float(pc.FactorizedRationalFunction(pc.FactorizedPolynomial(probability_function, pycarl.cln.cln._FactorizationCache()), denom).evaluate(resolution)) > 0 and float(pc.FactorizedRationalFunction(pc.FactorizedPolynomial(probability_function, pycarl.cln.cln._FactorizationCache()), denom).evaluate(resolution)) <= 1 for probability_function in pmc_transitions[s].values()])
        assert np.isclose(sum([float(pc.FactorizedRationalFunction(pc.FactorizedPolynomial(probability_function, pycarl.cln.cln._FactorizationCache()), denom).evaluate(resolution)) for probability_function in pmc_transitions[s].values()]), 1)
        # print(f"sum_t(t | s={s})", )
        rewards[s] = pc.FactorizedRationalFunction(pc.FactorizedPolynomial(rewards[s], pycarl.cln.cln._FactorizationCache()), denom)

    for s in states:
        row = p_matrix.get_row(s)
        for entry in row:
            # print(entry.column, str(entry.value()), entry.value().evaluate(resolution) > 0)
            assert entry.value().evaluate(resolution) > 0, ()

    # print(dir(p_matrix))
    # exit()
    
    # print(len(action_function_params), len(memory_function_params), counter)
    
    def print_params(params_dict : dict) -> None:
        for key, var in sorted(params_dict.items()):
            print(key, var)
    
    # print("actfun dict:")
    # print_params(action_function_params)
    # print("memfun dict:")
    # print_params(memory_function_params)
    # exit()
    
    # print("MC:", pmc_transitions)
    
    labelling = stormpy.storage.StateLabeling(len(states))
    
    for label, states in labeling.items():
        labelling.add_label(label)
        for s in states:
            labelling.add_label_to_state(label, s)

    # exit()
    
    pmc_reward_model = stormpy.storage.SparseParametricRewardModel(optional_state_reward_vector=[r for r in rewards.values()])
    
    components = stormpy.storage.SparseParametricModelComponents(p_matrix, labelling, reward_models={reward_model_name : pmc_reward_model})
    
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
    project_path="models/pomdp/sketches/obstacles-8-3"
    # project_path="models/pomdp/sketches/test"
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
    
    print(formula, pomdp_sketch.get_property().formula)
    
    task = stormpy.ParametricCheckTask(pomdp_sketch.get_property().formula, only_initial_states=False)
    
    # print("form:", formula, type(formula), formula.comparison_type, formula.threshold)
    
    storm_pmc_construction = False
    
    num_nodes = 2
    
    initial_probability = 0.1
    
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
        
        pmc, action_function_params, memory_function_params, resolution = construct_pmc(pomdp, pomdp_sketch, reward_model_name, num_nodes, initial_probability)
        
        action_function_params_no_const = {index : var for index, var in action_function_params.items() if isinstance(var, pycarl.Variable)}
        memory_function_params_no_const = {index : var for index, var in memory_function_params.items() if isinstance(var, pycarl.Variable)}
        for p in pmc.collect_all_parameters():
            assert p in action_function_params_no_const.values() or p in memory_function_params_no_const.values()
    
    # exit()
    
    
    # memory_builder = stormpy.pomdp.PomdpMemoryBuilder()
    # memory = memory_builder.build(stormpy.pomdp.PomdpMemoryPattern.selective_counter, 3)
    # pomdp = stormpy.pomdp.unfold_memory(pomdp, memory, add_memory_labels=True, keep_state_valuations=True)
    # pmc : stormpy.storage.storage.SparseParametricDtmc = stormpy.pomdp.apply_unknown_fsc(pomdp, stormpy.pomdp.PomdpFscApplicationMode.standard)
    
    print(pmc)
    parameters : set = pmc.collect_all_parameters()
    # print(parameters)
    
    for s in pmc.states:
        for action in s.actions:
            summation = 0
            for transition in action.transitions:
                valuation = transition.value().evaluate(resolution)
                assert valuation > 0 and valuation <= 1, (valuation, str(transition.value()), float(valuation))
                summation += valuation
                # exit()
            assert np.isclose(float(summation), 1)
    
    # print(dir(pmc))
    # exit()

    # assert len(parameters) == 
    print(f"There are currently {len(parameters)} parameters in the pMC!")
    # print(point)
    env = stormpy.Environment()
    # print(pmc.transition_matrix.get_row(0))
    # print(type(list(parameters)[0]))
    
    # exit()
    
    # print(help(pmc))
    
    # pmc.get_states_with_parameter(parameter=list(parameters)[0])
    # pmc.labels_state(state=0)
    
    # print(pmc.labels_state(state=0), pmc.labels_state(state=1), pmc.labels_state(state=2))
    # print(pmc.labels_state(state=2), pmc.labels_state(state=3), pmc.labels_state(state=4))
    # print(pmc.get_states_with_parameter(parameter=list(parameters)[0]))
    
    # for p in parameters:
        # states = pmc.get_states_with_parameter(parameter=p)
        
        # print(states)
    
    # assert len(pmc.collect_reward_parameters()) == 0, pmc.collect_reward_parameters()
    

    instantiator = stormpy.pars.PDtmcInstantiator(pmc)
    instantiated_model = instantiator.instantiate(resolution)
    result = stormpy.model_checking(instantiated_model, formula)
    print("RESULT:", result, result.as_explicit_quantitative().at(0), type(result))
    
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
        
    num_iters = 100
    
    for i in range(num_iters):  
        
        
        instantiated_model = instantiator.instantiate(resolution)
        result = stormpy.model_checking(instantiated_model, formula)
        print(f"I{i}. RESULT:", result, result.as_explicit_quantitative().at(0), type(result))

        new_resolution = {}

        for p in parameters:
            # print(env, point, list(parameters)[12])
            res = checker.check(env, resolution, p)
            # assert float(resolution[p]) + 0.001 * res.at(894) > 0 and float(resolution[p]) + 0.001 * res.at(894) <= 1, (float(resolution[p]), 0.001 * res.at(894))
            # new_resolution[p] = pc.Rational(min(max(float(resolution[p]) + 0.001 * res.at(894), 0), 1))
            new_resolution[p] = pc.Rational(min(max(float(resolution[p]) - 0.05 * sign(res.at(0)), 0), 1))
            
            # print(res.at(0), res.at(894), np.size(res.get_values()))
            # exit()
        # print(res.at(0), res.get_values(), dir(res))
        resolution = new_resolution
    
    # print(resolution)
    
    fsc = paynt.quotient.fsc.FSC(num_nodes, nO)
    
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

    memory_function = {}

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
    
    dtmc_sketch =  pomdp_sketch.build_dtmc_sketch(fsc, negate_specification=True)
    
    print(fsc.action_function)
    print(fsc.update_function)

    # solve
    synthesizer = paynt.synthesizer.synthesizer_ar.SynthesizerAR(dtmc_sketch)
    print(synthesizer.run())
    
    # print()
    
    # exit()


    
    exit()

    # construct an arbitrary 3-FSC
    fsc = random_fsc(pomdp_sketch,3)
    
    # unfold this FSC into the quotient POMDP to obtain the quotient MDP (DTMC sketch)
    # negate specification since we are interested in violating/worst assignments
    dtmc_sketch =  pomdp_sketch.build_dtmc_sketch(fsc, negate_specification=True)

    # solve
    synthesizer = paynt.synthesizer.synthesizer_ar.SynthesizerAR(dtmc_sketch)
    synthesizer.run()


    if profiling:
        profiler.disable()
        stats = profiler.create_stats()
        pstats.Stats(profiler).sort_stats('tottime').print_stats(10)


main()