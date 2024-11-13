# using Paynt for POMDP sketches
import copy
import math
import operator
import time
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

from enum import Enum

class Method(Enum):
    GRADIENT = 1
    SAYNT = 2
    PAYNT = 3

def sign(x):
    if np.isclose(x, 0):
        return 0
    else:
        return 1 if x > 0 else -1
    
def stablesoftmax(x, temperature = 1) -> np.ndarray:
    """Compute the softmax of vector x in a numerically stable way."""
    assert len(x.shape) <= 1, "Not implemented for arrays of dimension greater than 1."
    shiftx = x - np.max(x)
    exps = np.exp(shiftx / temperature)
    return exps / np.sum(exps)

class POMDPFamiliesSynthesis:

    def load_sketch(self, project_path):
        project_path = os.path.abspath(project_path)
        sketch_path = os.path.join(project_path, "sketch.templ")
        properties_path = os.path.join(project_path, "sketch.props")    
        pomdp_sketch = paynt.parser.sketch.Sketch.load_sketch(sketch_path, properties_path)
        return pomdp_sketch

    def assignment_to_pomdp(self, pomdp_sketch, assignment):
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

    def solve_pomdp_paynt(self, pomdp, specification, k, timeout=1):
        pomdp_quotient = paynt.quotient.pomdp.PomdpQuotient(pomdp, specification)
        pomdp_quotient.set_imperfect_memory_size(k)
        # print(dir(pomdp_quotient), pomdp_quotient.action_labels_at_observation[0])#, pomdp_quotient.action_labels)
        synthesizer = paynt.synthesizer.synthesizer_ar.SynthesizerAR(pomdp_quotient)
        assignment = synthesizer.synthesize(timeout=timeout)
        assert assignment is not None
        fsc = pomdp_quotient.assignment_to_fsc(assignment)
        return fsc

    def parameters_to_paynt_fsc(self, action_function_params, memory_function_params, resolution, num_nodes, num_obs, observation_to_actions):
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

    def solve_pomdp_saynt(self, pomdp, specification, k, timeout=10):
        pomdp_quotient = paynt.quotient.pomdp.PomdpQuotient(pomdp, specification)
        storm_control = paynt.quotient.storm_pomdp_control.StormPOMDPControl()
        paynt_iter_timeout = 5
        storm_iter_timeout = 2
        iterative_storm = (timeout, paynt_iter_timeout, storm_iter_timeout)
        storm_control.set_options(
            storm_options="cutoff", get_storm_result=None, iterative_storm=iterative_storm, use_storm_cutoffs=False,
            unfold_strategy_storm="storm", prune_storm=True, export_fsc_storm=None, export_fsc_paynt=None
        )
        synthesizer = paynt.synthesizer.synthesizer.Synthesizer.choose_synthesizer(
                pomdp_quotient, method="ar", fsc_synthesis=True, storm_control=storm_control
        )
        synthesizer.run(optimum_threshold=None)
        assert synthesizer.storm_control.latest_paynt_result_fsc is not None
        fsc = synthesizer.storm_control.latest_paynt_result_fsc
        return fsc, None # replacing None with synthesizer stores the classes in a list to not delete the objects, but to no avail.

    def random_fsc(self, pomdp_sketch, num_nodes):
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
    
    def create_random_subfamily(self, family_size : int):
        hole_combinations = random.choices(list(self.pomdp_sketch.family.all_combinations()), k = family_size)
        hole_assignments_to_test = [self.pomdp_sketch.family.construct_assignment(hole_combination) for hole_combination in hole_combinations]
        return hole_assignments_to_test

    def experiment_on_subfamily(self, hole_assignments_to_test : list, num_nodes : int, method : Method, timeout=15, evaluate_on_whole_family=False):
        results = {}

        nO = self.pomdp_sketch.num_observations

        dummy = []
        
        pomdps = []

        for i, assignment in enumerate(hole_assignments_to_test):
            
            if method.value == method.GRADIENT.value:
                value, resolution, action_function_params, memory_function_params, *_ = self.gradient_descent_on_single_pomdp(assignment, 1000, num_nodes, timeout=timeout, parameter_resolution={})
                print(i, assignment, value)
                fsc = self.parameters_to_paynt_fsc(action_function_params, memory_function_params, resolution, num_nodes, nO, self.pomdp_sketch.observation_to_actions)
            elif method.value > method.GRADIENT.value:
                pomdp = self.pomdp_sketch.build_pomdp(assignment)
                pomdps.append(pomdp) 
                # assert that observation classes are preserved
                for state in range(pomdp.model.nr_states):
                    quotient_state = pomdp.quotient_state_map[state]
                    assert pomdp.model.observations[state] == self.pomdp_sketch.state_to_observation[quotient_state]
                pomdp = pomdp.model
                specification = self.pomdp_sketch.specification.copy()
                if method.value == method.SAYNT.value:
                    fsc, objects = self.solve_pomdp_saynt(pomdp, specification, num_nodes, timeout=timeout) # GO OOM
                    dummy.append(objects)
                else:
                    fsc = self.solve_pomdp_paynt(pomdp, specification, num_nodes, timeout=timeout)

                for n in range(fsc.num_nodes):
                    for o in range(nO):
                        if o >= len(fsc.action_function[n]):
                            fsc.action_function[n].append({a : 1 / len(self.pomdp_sketch.observation_to_actions[o]) for a in self.pomdp_sketch.observation_to_actions[o]})
                            fsc.update_function[n].append({m : 1 / num_nodes for m in range(fsc.num_nodes)})
                        else:
                            a = fsc.action_function[n][o]
                            action_label = fsc.action_labels[a]
                            family_action = self.pomdp_sketch.action_labels.index(action_label)
                            assert family_action in self.pomdp_sketch.observation_to_actions[o]
                            fsc.action_function[n][o] = {family_action : 1.0}
                            fsc.update_function[n][o] = {fsc.update_function[n][o] : 1.0}

                assert all([len(fsc.action_function[n]) == nO for n in range(fsc.num_nodes)])
                fsc.is_deterministic = False
                fsc.num_observations = nO

            dtmc_sketch = self.pomdp_sketch.build_dtmc_sketch(fsc, negate_specification=True)
            one_by_one = paynt.synthesizer.synthesizer_onebyone.SynthesizerOneByOne(dtmc_sketch)
            evaluations = {}
            for j, family in enumerate(hole_assignments_to_test):
                evaluations[j] = one_by_one.evaluate(family, keep_value_only=True)

            if evaluate_on_whole_family:
                synthesizer = paynt.synthesizer.synthesizer_ar.SynthesizerAR(dtmc_sketch)
                synthesizer.synthesize(keep_optimum=True)
                results[i] = (str(assignment), fsc, evaluations, synthesizer.best_assignment_value)
                print(evaluations, synthesizer.best_assignment_value)
            else:
                results[i] = (str(assignment), fsc, evaluations)
                print(evaluations)
        

        if evaluate_on_whole_family:
            print("Whole family result:", [result[3] for result in results.values()])

        print("Subfamily Result:", [max(result[2].values()) for result in results.values()])
        
        return results

    # @profile
    def construct_pmc(self, pomdp, pomdp_sketch, reward_model_name, num_nodes, action_function_params = {}, memory_function_params = {}, resolution = {}, parameter_resolution = None, distinct_parameters_for_final_probability = False, sanity_checks = True):
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
                                    if self.use_softmax: parameter_resolution[act_param] = random.normalvariate()
                                    resolution[act_param] = pc.Rational(1 / len(pomdp_sketch.observation_to_actions[o]) + 1e-6)
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
                                    resolution[mem_param] = pc.Rational(1 / num_nodes + 1e-6)
                                    if self.use_softmax: parameter_resolution[mem_param] = random.normalvariate()
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
        
        # resolution = {expr : pc.Rational(initial_probability) for key, expr in action_function_params.items() if isinstance(expr, pycarl.Variable)}
        # resolution.update({
            # expr : pc.Rational(initial_probability) for key, expr in memory_function_params.items() if isinstance(expr, pycarl.Variable)
        # })
        
        cache = pycarl.cln.cln._FactorizationCache()

        for s, next_states in sorted(pmc_transitions.items(), key = lambda x : x[0]):
            # BELOW EATS MEMORY
            assert all([float(pc.FactorizedRationalFunction(pc.FactorizedPolynomial(probability_function, cache), denom).evaluate(resolution)) > 0 and float(pc.FactorizedRationalFunction(pc.FactorizedPolynomial(probability_function, cache), denom).evaluate(resolution)) <= 1 for probability_function in pmc_transitions[s].values()])
            assert np.isclose(sum([float(pc.FactorizedRationalFunction(pc.FactorizedPolynomial(probability_function, cache), denom).evaluate(resolution)) for probability_function in pmc_transitions[s].values()]), 1)
            rewards[s] = pc.FactorizedRationalFunction(pc.FactorizedPolynomial(rewards[s], cache), denom)
            for t, probability_function in sorted(next_states.items(), key = lambda x : x[0]):
                parametric_transition = pc.FactorizedRationalFunction(pc.FactorizedPolynomial(probability_function, cache), denom)
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
        
        return pmc, action_function_params, memory_function_params, resolution, parameter_resolution

    def resolution_to_softmax(self, action_function_params, memory_function_params, parameter_resolution, num_nodes):
        probabilistic_resolution = {}
        for n in range(num_nodes):
            for o in range(self.nO):
                action_params = []
                action_params = [action_function_params[n,o,a] for a in range(self.nA) if (n,o,a) in action_function_params]
                action_parameter_values = np.array([float(parameter_resolution[var]) for var in action_params if var in parameter_resolution])
                softmax_action_probs = stablesoftmax(action_parameter_values)
                softmax_action_jacobian = np.diag(softmax_action_probs) - np.inner(softmax_action_probs, softmax_action_probs)
                assert math.isclose(sum(softmax_action_probs), 1)
                for var, softmax_prob in zip(action_params, softmax_action_probs):
                    assert softmax_prob > 0 and softmax_prob <= 1, (softmax_action_probs, action_parameter_values)
                    probabilistic_resolution[var] = pycarl.cln.cln.Rational(softmax_prob)
                    
                node_params = [memory_function_params[n,o,m] for m in range(num_nodes) if (n,o,m) in memory_function_params]
                memory_parameter_values = np.array([float(parameter_resolution[var]) for var in node_params if var in parameter_resolution])
                softmax_memory_probs = stablesoftmax(np.array([float(parameter_resolution[var]) for var in node_params]))
                softmax_memory_jacobian = np.diag(softmax_memory_probs) - np.inner(softmax_memory_probs, softmax_memory_probs)
                assert math.isclose(sum(softmax_memory_probs), 1)
                for var, softmax_prob in zip(node_params, softmax_memory_probs):
                    assert softmax_prob > 0 and softmax_prob <= 1, (softmax_memory_probs, memory_parameter_values)
                    probabilistic_resolution[var] = pycarl.cln.cln.Rational(softmax_prob)
        
        return probabilistic_resolution

    def softmax_gradients(self, action_function_params, memory_function_params, parameter_resolution, num_nodes, gradients):
        softmax_grad = {}
        for n in range(num_nodes):
            for o in range(self.nO):
                action_params = []
                action_params = [action_function_params[n,o,a] for a in range(self.nA) if (n,o,a) in action_function_params]
                action_parameter_values = np.array([float(parameter_resolution[var]) for var in action_params if var in parameter_resolution])
                action_gradients = np.array([float(gradients[var]) for var in action_params if var in parameter_resolution])
                softmax_action_probs = stablesoftmax(action_parameter_values)
                summation_term = np.inner(softmax_action_probs, action_gradients).sum()
                # softmax_action_jacobian = np.diag(softmax_action_probs) - np.inner(softmax_action_probs, softmax_action_probs)
                assert math.isclose(softmax_action_probs.sum(), 1)
                for var, softmax_prob in zip(action_params, softmax_action_probs):
                    assert softmax_prob > 0 and softmax_prob <= 1, (softmax_action_probs, action_parameter_values)
                    softmax_grad[var] = softmax_prob * (gradients[var] - summation_term)
                    
                node_params = [memory_function_params[n,o,m] for m in range(num_nodes) if (n,o,m) in memory_function_params]
                memory_parameter_values = np.array([float(parameter_resolution[var]) for var in node_params if var in parameter_resolution])
                memory_gradients = np.array([float(gradients[var]) for var in node_params if var in parameter_resolution])
                softmax_memory_probs = stablesoftmax(np.array([float(parameter_resolution[var]) for var in node_params]))
                # softmax_memory_jacobian = np.diag(softmax_memory_probs) - np.inner(softmax_memory_probs, softmax_memory_probs)
                summation_term = np.inner(softmax_memory_probs, memory_gradients).sum()
                assert math.isclose(sum(softmax_memory_probs), 1)
                for var, softmax_prob in zip(node_params, softmax_memory_probs):
                    assert softmax_prob > 0 and softmax_prob <= 1, (softmax_memory_probs, memory_parameter_values)
                    softmax_grad[var] = softmax_prob * (gradients[var] - summation_term)
        
        return softmax_grad
    
    def clip_gradient(self, x, doclip=True):
        if doclip:
            return np.clip(x, -50, 50)
        else:
            return x

    def gradient_descent_on_single_pomdp(self, hole_assignment, num_iters : int, num_nodes : int, action_function_params = {}, memory_function_params = {}, resolution = {}, parameter_resolution = None, timeout = None):
        pomdp_class = self.pomdp_sketch.build_pomdp(hole_assignment)
        pomdp = pomdp_class.model
        pmc, action_function_params, memory_function_params, resolution, parameter_resolution = self.construct_pmc(pomdp, self.pomdp_sketch, self.reward_model_name, num_nodes, distinct_parameters_for_final_probability=self.use_softmax, parameter_resolution=parameter_resolution, resolution=resolution, action_function_params=action_function_params, memory_function_params=memory_function_params)
        
        if self.use_softmax:
            resolution = self.resolution_to_softmax(action_function_params, memory_function_params, parameter_resolution, num_nodes)
            self.sanity_check_pmc_at_instantiation(pmc, resolution)
            checker = payntbind.synthesis.SparseDerivativeInstantiationModelCheckerFamily(pmc)
            task = stormpy.ParametricCheckTask(self.pomdp_sketch.get_property().formula, only_initial_states=False)
            checker.specifyFormula(stormpy.Environment(), task)
            # exit()
        else:
            wrapper = payntbind.synthesis.GradientDescentInstantiationSearcherFamily(pmc, self.lr, self.mbs, self.gd_steps)
            wrapper.setup(self.env, self.synth_task)
            wrapper.resetDynamicValues()

        action_function_params_no_const = {index : var for index, var in action_function_params.items() if isinstance(var, pycarl.Variable)}
        memory_function_params_no_const = {index : var for index, var in memory_function_params.items() if isinstance(var, pycarl.Variable)}
        for p in pmc.collect_all_parameters():
            assert p in action_function_params_no_const.values() or p in memory_function_params_no_const.values()
        
        if timeout:
            tik = time.time()
        
        # for thing, other in resolution.items():
            # print(thing, other)

        # print(result)
        
        # print(pmc.collect_all_parameters(), resolution.values())
        
        # print("ALL PARAM:", set(pmc.collect_all_parameters()), "RESOLUTION:", set(list(resolution.keys())), sep='\n\n')
        
        # print("RESOLUTION:", resolution)
        
        assert set(pmc.collect_all_parameters()) == set(list(resolution.keys()))
        
        
        for i in range(num_iters):
            if self.use_softmax:
                new_resolution = {}
                new_parameter_resolution = {}
                grads = {}
                direction_operator = operator.sub if self.minimizing else operator.add
                for p in pmc.collect_all_parameters():
                    res = checker.check(self.env, resolution, p)
                    # if self.minimizing:
                        # update = float(parameter_resolution[p]) - self.lr * (res.at(0))
                    # else:
                        # update = float(parameter_resolution[p]) + self.lr * (res.at(0))
                    # for j in pmc.collect_all_parameters():
                        # resolution[p] * 
                            
                    # corrected = pc.Rational(min(max(update, 0), 1))
                    # print(update, float(corrected))
                    # assert corrected >= 0 and corrected <= 1
                    # grads[p] = np.clip((res.at(0)), -5, 5)
                    grads[p] = self.clip_gradient(res.at(0), doclip=False)
                # print(grads)
                
                # for i in pmc.collect_all_parameters(): 
                    # for j in pmc.collect_all_parameters(): # TODO: only have to differentiate with respect to the others variables that were inside the same softmax. Not all other variables.
                        # softmaxgrad = (parameter_resolution[i] * ((1 if i == j else 0) - parameter_resolution[j])) * grads[j]
                        # new_parameter_resolution[j] = direction_operator(float(parameter_resolution[j]), self.lr * softmaxgrad)
                softmax_grads = self.softmax_gradients(action_function_params, memory_function_params, parameter_resolution, num_nodes, grads)
                for p in pmc.collect_all_parameters():
                    new_parameter_resolution[p] = direction_operator(float(parameter_resolution[p]), self.lr * self.clip_gradient(softmax_grads[p], doclip=True))
                    # assert new_resolution[p] >= 0
                # print(new_parameter_resolution)
                # exit()
                new_resolution = self.resolution_to_softmax(action_function_params, memory_function_params, new_parameter_resolution, num_nodes)
                instantiator = stormpy.pars.PDtmcInstantiator(pmc)
                instantiated_model = instantiator.instantiate(new_resolution)
                result = stormpy.model_checking(instantiated_model, self.pomdp_sketch.get_property().property.raw_formula)
                current_value = result.at(0)
                parameter_resolution = new_parameter_resolution
                resolution = new_resolution
            else:
                try:
                    wrapper.resetDynamicValues()
                    self.sanity_check_pmc_at_instantiation(pmc, resolution)
                    current_value, resolution = wrapper.stochasticGradientDescent(resolution)
                except Exception as e:
                    print([float(x) for x in resolution.values()])
                    self.sanity_check_pmc_at_instantiation(pmc, resolution)
                    raise e
            print(i, current_value)
            if timeout and time.time() - tik > timeout:
                break
        
        return current_value, resolution, action_function_params, memory_function_params, parameter_resolution

    
    def run_gradient_descent_on_family(self, num_iters : int, num_nodes : int, assignments : list = None, timeout : int = None, random_selection : bool = False):
        current_value = None
        
        action_function_params = {}
        memory_function_params = {}
        
        best_family_value = 1e30 if self.minimizing else -1e30
        
        best_fsc = None
        
        if assignments is None:
            print("Running on 'entire' family of size:")
        else:
            print("Running on sub-family of size:", len(assignments))
        
        values = []
        resolution = {}
        
        parameter_resolution = {} if self.use_softmax else None
        
        if timeout: tik = time.time()
        
        for i in range(num_iters):
            
            artificial_upper_bound = None # if current_value is None else current_value * (0.9 if self.minimizing else 1.1)
            
            if i > 0:
                fsc = self.parameters_to_paynt_fsc(action_function_params, memory_function_params, resolution, num_nodes, self.nO, self.pomdp_sketch.observation_to_actions)                
                dtmc_sketch =  self.pomdp_sketch.build_dtmc_sketch(fsc, negate_specification=True)
                hole_assignment, paynt_value = self.paynt_call(dtmc_sketch, assignments=assignments, artificial_upper_bound=artificial_upper_bound, random_selection=random_selection)
                print("Paynt value:", paynt_value, "previous family best:", best_family_value)
                op = operator.lt if self.minimizing else operator.gt
                if op(paynt_value, best_family_value):
                    best_fsc = fsc
                    best_family_value = paynt_value
                if timeout and time.time() - tik > timeout: break
            else:
                hole_assignment = self.pomdp_sketch.family.pick_any()
            
            current_value, new_resolution, *_ = self.gradient_descent_on_single_pomdp(hole_assignment, 10 // self.gd_steps, num_nodes, action_function_params, memory_function_params, resolution, parameter_resolution)
            
            values.append(current_value)
            
            resolution = new_resolution
        
        return best_fsc, best_family_value
    
    def paynt_call(self, dtmc_sketch, assignments = None, artificial_upper_bound = None, random_selection = False) -> tuple[paynt.family.family.Family, float]:
        if assignments is None:
            synthesizer = paynt.synthesizer.synthesizer_ar.SynthesizerAR(dtmc_sketch)
            hole_assignment = synthesizer.synthesize(optimum_threshold=artificial_upper_bound, keep_optimum=True)
            paynt_value = synthesizer.best_assignment_value
            if random_selection: 
                hole_assignment = self.pomdp_sketch.family.pick_random()
        else:
            assert assignments is not None
            synthesizer = paynt.synthesizer.synthesizer_onebyone.SynthesizerOneByOne(dtmc_sketch)
            evaluations = np.zeros(len(assignments))
            for j, family in enumerate(assignments):
                evaluations[j] = synthesizer.evaluate(family, keep_value_only=True)[0]
            hole_assignment_idx = np.argmax(evaluations) if self.minimizing else np.argmin(evaluations)
            paynt_value = evaluations[hole_assignment_idx]
            if random_selection:
                hole_assignment = random.choice(assignments)
            else:
                hole_assignment = assignments[hole_assignment_idx]
        
        return hole_assignment, paynt_value
    
    def sanity_check_pmc_at_instantiation(self, pmc : stormpy.storage.SparseParametricDtmc, resolution : dict[pycarl.Variable, pc.Rational]):
        # if not self.use_softmax:
        if True:
            for s in pmc.states:
                for action in s.actions:
                    summation = 0
                    for transition in action.transitions:
                        valuation = float(transition.value().evaluate(resolution))
                        assert valuation > 0 and valuation <= 1, (valuation, str(transition.value()), float(valuation))
                        summation += valuation
                    assert np.isclose(summation, 1)
                    
            instantiator = stormpy.pars.PDtmcInstantiator(pmc)            
            instantiated_model = instantiator.instantiate(resolution)
            # print(instantiated_model)
                    
            print("pMC shoud be ok.")

    def __init__(self, project_path : str, seed : int = 11, use_softmax : bool = False, learning_rate = 0.001, minibatch_size = 256, steps = 10):
        random.seed(seed)
        np.random.seed(seed)

        # enable PAYNT logging
        paynt.cli.setup_logger()
        
        self.lr = learning_rate
        self.mbs = minibatch_size
        self.gd_steps = steps

        # load sketch
        self.pomdp_sketch = self.load_sketch(project_path)
        
        self.storm_control = None
        
        self.nO = self.pomdp_sketch.num_observations
        self.nA = self.pomdp_sketch.num_actions
        
        self.reward_model_name = self.pomdp_sketch.get_property().get_reward_name()
        
        print("|O| =", self.nO, "|A| =", self.nA)
        
        self.formula = self.pomdp_sketch.get_property().property.raw_formula
        
        self.minimizing = self.pomdp_sketch.get_property().minimizing
        
        print(self.formula, self.pomdp_sketch.get_property().formula)

        self.use_softmax = use_softmax

        self.env = stormpy.Environment()
        
        temp_form = self.formula.clone()
        manager = stormpy.storage.ExpressionManager()
        parsed = manager.create_integer(1 if self.minimizing else int(1e6))
        temp_form.set_bound(stormpy.ComparisonType.LEQ if self.minimizing else stormpy.ComparisonType.GEQ,  parsed)
        temp_form.remove_optimality_type()
        self.synth_task = payntbind.synthesis.FeasibilitySynthesisTask(temp_form)
        self.synth_task.set_bound(stormpy.ComparisonType.LEQ if self.minimizing else stormpy.ComparisonType.GEQ,  parsed)
