# using Paynt for POMDP sketches

import paynt.cli
import paynt.parser.sketch
import paynt.quotient.pomdp_family
import paynt.quotient.fsc
import paynt.synthesizer.synthesizer_onebyone
import paynt.synthesizer.synthesizer_ar

import stormpy
import stormpy.pomdp
import pycarl
import stormpy.info

# import payntbind.synthesis.SparseDerivativeInstantiationModelCheckerFamily

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
    action_labels,_ = payntbind.synthesis.extractActionLabels(pomdp);
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
    project_path="models/pomdp/sketches/test"
    pomdp_sketch = load_sketch(project_path)
    
    reward_model_name = pomdp_sketch.get_property().get_reward_name()

    # construct POMDP from the given hole assignment
    hole_assignment = pomdp_sketch.family.pick_any()
    # pomdp,observation_action_to_true_action = assignment_to_pomdp(pomdp_sketch,hole_assignment)
    
    pomdp = pomdp_sketch.build_pomdp(hole_assignment).model
    
    print("|O| =", pomdp_sketch.num_observations, "|A| =", pomdp_sketch.num_actions)
    
    formula = pomdp_sketch.get_property().property.raw_formula
    
    print("form:", formula, type(formula), formula.comparison_type, formula.threshold)
    
    task = stormpy.ParametricCheckTask(pomdp_sketch.get_property().formula, only_initial_states=False)
    
    synth_task = payntbind.synthesis.FeasibilitySynthesisTask(formula)
    
    synth_task.set_bound(formula.comparison_type, formula.threshold_expr)
    
    # for i in range(0, 100):
    #     # construct POMDP from the given hole assignment
    #     hole_assignment = pomdp_sketch.family.pick_random()
    #     # pomdp,observation_action_to_true_action = assignment_to_pomdp(pomdp_sketch,hole_assignment)
        
    #     pomdp = pomdp_sketch.build_pomdp(hole_assignment).model
    #     memory_builder = stormpy.pomdp.PomdpMemoryBuilder()
    #     memory = memory_builder.build(stormpy.pomdp.PomdpMemoryPattern.selective_counter, 3)
    #     pomdp = stormpy.pomdp.unfold_memory(pomdp, memory, add_memory_labels=True, keep_state_valuations=True)
    #     pmc : stormpy.storage.storage.SparseParametricDtmc = stormpy.pomdp.apply_unknown_fsc(pomdp, stormpy.pomdp.PomdpFscApplicationMode.simple_linear)
    #     parameters : set = pmc.collect_all_parameters()
    #     print(f"There are currently {len(parameters)} parameters in the pMC!")
    
    # exit()
    
    num_nodes = 3
    
    pycarl.clear_pools()

    if stormpy.info.storm_ratfunc_use_cln():
        import pycarl.cln as pc
    else:
        import pycarl.gmp as pc
    
    builder : stormpy.storage.storage.ParametricSparseMatrixBuilder = stormpy.storage.ParametricSparseMatrixBuilder()
    
    counter = 0
    
    action_function_params = {}
    memory_function_params = {}
    
    seen = set()
    
    pmc_transitions = {}
    
    rewards = {}
    
    import numpy as np
    
    
    print(pomdp.reward_models)
    reward_model = pomdp.reward_models[reward_model_name]
    assert not reward_model.has_state_rewards
    state_action_rewards = reward_model.state_action_rewards
    print(state_action_rewards)
    state_action_rewards_vector = np.array([x for x in state_action_rewards])
    
    rewards = {}
    
    denom = pc.FactorizedPolynomial(pc.Rational(1))
    
    ndi = pomdp.nondeterministic_choice_indices
    
    labels = pomdp.labeling
    
    states = set()
    labeling = {l : [] for l in labels.get_labels()}
    
    target_label = pomdp_sketch.get_property().get_target_label()
    
    print("Memory function parameters:", num_nodes * pomdp.nr_observations * num_nodes)
    
    for state in pomdp.states:
        s = state.id
        o = pomdp.observations[s]
        for action in pomdp.states[s].actions:
            a = action.id
            choice = ndi[s]+a
            reward = state_action_rewards[choice]
            for transition in action.transitions:
                t = transition.column
                t_prob = transition.value()
                # print(t_prob, type(t_prob))
                for n in range(num_nodes):
                    # if len(state.labels) > 0:
                    states.add(s * num_nodes + n)
                    pmc_transitions[s * num_nodes + n] = {}
                    for label in state.labels:
                        labeling[label].append(s * num_nodes + n)
                    # if target_label in state.labels:
                        # goal_states.append(s * num_nodes + n)
                    for m in range(num_nodes):
                        states.add(t * num_nodes + m)
                        act_tup = (n, o, a)
                        mem_tup = (n, o, m)
                        
                        assert (s * num_nodes + n, t * num_nodes + m, a) not in seen, (s * num_nodes + n, t * num_nodes + m, seen)
                        seen.add((s * num_nodes + n, t * num_nodes + m, a))
                        
                        if act_tup in action_function_params:
                            act_param = action_function_params[act_tup]
                        else:
                            action_ids = [b.id for b in pomdp.states[s].actions]
                            # if len(action_ids) > 1:
                            if a == max(action_ids):
                                # n, o, a = None
                                act_param = pc.Rational(1)
                                for a_ in action_ids:
                                    if a != a_:
                                        assert (n, o, a_) in action_function_params
                                        act_param -= action_function_params[n, o, a_]
                                # print("Var:", var, a, a_, [b for b in pomdp.states[s].actions][-1].id)
                            else:
                                p_a_name = f"p{counter}_n{n}_o{o}_a{a}"
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
                        # num = pc.FactorizedPolynomial(action_mem_poly, pycarl.cln.cln._FactorizationCache())

                        # assert s * num_nodes + n < t * num_nodes + m
                        if (t * num_nodes + m) in pmc_transitions[s * num_nodes + n]:
                            pmc_transitions[(s * num_nodes + n)][(t * num_nodes + m)] += action_mem_poly
                        else:
                            pmc_transitions[(s * num_nodes + n)][(t * num_nodes + m)] = action_mem_poly

                    if s * num_nodes + n in rewards:
                        rewards[s * num_nodes + n] += pc.Polynomial(action_function_params[(n, o, a)]) * pc.Rational(float(reward))
                    else:
                        rewards[s * num_nodes + n] = pc.Polynomial(action_function_params[(n, o, a)]) * pc.Rational(float(reward))
    
    resolution = {expr : pc.Rational(0.4) for key, expr in action_function_params.items() if isinstance(expr, pycarl.Variable)}
    resolution.update({
        expr : pc.Rational(0.4) for key, expr in memory_function_params.items() if isinstance(expr, pycarl.Variable)
    })
    
    print(resolution)
    
    print("----")
    for s, next_states in sorted(pmc_transitions.items(), key = lambda x : x[0]):
        for t, probability_function in next_states.items():
            # print(s, t, probability_function)
            builder.add_next_value(s, t, pc.FactorizedRationalFunction(pc.FactorizedPolynomial(probability_function, pycarl.cln.cln._FactorizationCache()), denom))
            print(s, t, float(probability_function.evaluate(resolution)))
            # print(help(probability_function))
            # exit()
    print("----")
    
    # print(pmc_transitions)
    # exit()
    
    for s in states:
        print(f"sum_t(t | s={s})", sum([float(probability_function.evaluate(resolution)) for probability_function in pmc_transitions[s].values()]))
        rewards[s] = pc.FactorizedRationalFunction(pc.FactorizedPolynomial(rewards[s], pycarl.cln.cln._FactorizationCache()), denom)
                    
    p_matrix = builder.build()
    
    print(len(action_function_params), len(memory_function_params), counter)
    
    print("actfun dict:", action_function_params)
    print("memfun dict:", memory_function_params)
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
    
    # memory_builder = stormpy.pomdp.PomdpMemoryBuilder()
    # memory = memory_builder.build(stormpy.pomdp.PomdpMemoryPattern.selective_counter, 3)
    # pomdp = stormpy.pomdp.unfold_memory(pomdp, memory, add_memory_labels=True, keep_state_valuations=True)
    # pmc : stormpy.storage.storage.SparseParametricDtmc = stormpy.pomdp.apply_unknown_fsc(pomdp, stormpy.pomdp.PomdpFscApplicationMode.standard)
    
    print(pmc)
    parameters : set = pmc.collect_all_parameters()
    print(f"There are currently {len(parameters)} parameters in the pMC!")
    print(len(action_function_params) + len(memory_function_params))
    point = {p: stormpy.RationalRF(1/2 + 1e-6)  for p in parameters}
    env = stormpy.Environment()
    print(pmc.collect_all_parameters())
    # print(pmc.transition_matrix.get_row(0))
    print(type(list(parameters)[0]))
    
    # exit()
    
    # print(help(pmc))
    
    # pmc.get_states_with_parameter(parameter=list(parameters)[0])
    # pmc.labels_state(state=0)
    
    print(pmc.labels_state(state=0), pmc.labels_state(state=1), pmc.labels_state(state=2))
    print(pmc.labels_state(state=2), pmc.labels_state(state=3), pmc.labels_state(state=4))
    print(pmc.get_states_with_parameter(parameter=list(parameters)[0]))
    
    # for p in parameters:
        # states = pmc.get_states_with_parameter(parameter=p)
        
        # print(states)
    
    # assert len(pmc.collect_reward_parameters()) == 0, pmc.collect_reward_parameters()
    
    wrapper = payntbind.synthesis.GradientDescentInstantiationSearcherFamily(pmc)
    wrapper.setup(env, synth_task)
    wrapper.gradientDescent()

    checker = payntbind.synthesis.SparseDerivativeInstantiationModelCheckerFamily(pmc)
    checker.specifyFormula(env, task)
    # print(env, point, list(parameters)[12])
    res = checker.check(env, point, list(point.keys())[0])
    print(res)
    print(res.at(0), res.get_values(), dir(res))
    
    exit()

    wrapper = payntbind.synthesis.GradientDescentInstantiationSearcherFamily(pmc)
    wrapper.setup(env, synth_task)
    wrapper.gradientDescent()
    
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