# using Paynt for POMDP sketches
import copy
import math
import operator
import time
from collections import defaultdict
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
    
    def __init__(self, project_path : str, seed : int = 11, use_softmax : bool = False, learning_rate = 0.001, minibatch_size = 256, steps = 10, use_momentum = True, dynamic_memory = False):
        random.seed(seed)
        np.random.seed(seed)

        self.gd_trace = []
        self.family_trace = []
        
        self.dynamic_memory = dynamic_memory
        
        self.clip_gradient_value = 5

        # enable PAYNT logging
        paynt.cli.setup_logger()

        self.lr = learning_rate
        self.mbs = minibatch_size
        self.gd_steps = steps

        self.use_momentum = use_momentum
        self.reset_momentum = False
        self.momentum = None
        self.beta = 0.9

        # load sketch
        self.pomdp_sketch : paynt.quotient.pomdp_family.PomdpFamilyQuotient = self.load_sketch(project_path)

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


    def load_sketch(self, project_path):
        project_path = os.path.abspath(project_path)
        sketch_path = os.path.join(project_path, "sketch.templ")
        properties_path = os.path.join(project_path, "sketch.props")    
        pomdp_sketch = paynt.parser.sketch.Sketch.load_sketch(sketch_path, properties_path)
        return pomdp_sketch

    def assignment_to_pomdp(self, pomdp_sketch, assignment, restore_absorbing_states=True):
        pomdp = pomdp_sketch.build_pomdp(assignment).model
        if restore_absorbing_states:
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
            observation_action_to_true_action[obs] = {}
            choice_0 = pomdp.transition_matrix.get_row_group_start(state)
            for action,action_label in enumerate(action_labels):
                choice = choice_0+action
                true_action = choice_to_true_action[choice]
                true_action_label = action_labels[true_action]
                observation_action_to_true_action[obs][action_label] = true_action_label
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

    def parameters_to_paynt_fsc(self, action_function_params, memory_function_params, resolution, num_nodes, num_obs, observation_to_actions, memory_model = None):
        
        if memory_model is None:
            memory_model = [num_nodes] * self.nO
        
        fsc = paynt.quotient.fsc.FSC(num_nodes, num_obs)

        for (n,o,a), var in action_function_params.items():
            if var in resolution:
                prob = float(resolution[var])
            else:
                if isinstance(var, pycarl.cln.cln.Rational):
                    prob = float(var)
                else:
                    print("Should this occur?")
                    prob = float(var.evaluate(resolution))

            if prob == 0:
                continue

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
                    print("Should this occur?")
                    prob = float(var.evaluate(resolution))

            if prob == 0:
                continue

            if fsc.update_function[n][o] is None:
                fsc.update_function[n][o] = {m : prob}
            else:
                fsc.update_function[n][o].update({m : prob})
        
        new_fsc = paynt.quotient.fsc.FSC(num_nodes, num_obs)

        for o in range(self.nO):
            for n in range(num_nodes):
                new_fsc.action_function[n][o] = fsc.action_function[n % memory_model[o]][o]
                new_fsc.update_function[n][o] = fsc.update_function[n % memory_model[o]][o]
                
        fsc = new_fsc

        fsc.action_function = [[d if d is not None else {a : 1/len(observation_to_actions[o]) for a in observation_to_actions[o]} for o, d in enumerate(o_to_act)] for o_to_act in fsc.action_function]
        fsc.update_function = [[d if d is not None else {m : 1/num_nodes for m in range(num_nodes)} for d in o_to_mem ] for o_to_mem in fsc.update_function]

        return fsc

    def solve_pomdp_saynt(self, pomdp, specification, k, timeout=10):
        pomdp_quotient = paynt.quotient.pomdp.PomdpQuotient(pomdp, specification)
        storm_control = paynt.quotient.storm_pomdp_control.StormPOMDPControl()
        paynt_iter_timeout = 3
        storm_iter_timeout = 3
        iterative_storm = (timeout, paynt_iter_timeout, storm_iter_timeout)
        storm_control.set_options(
            storm_options="cutoff", get_storm_result=None, iterative_storm=iterative_storm, use_storm_cutoffs=False,
            unfold_strategy_storm="storm", prune_storm=True, export_fsc_storm=None, export_fsc_paynt=None
        )
        synthesizer = paynt.synthesizer.synthesizer.Synthesizer.choose_synthesizer(
                pomdp_quotient, method="ar", fsc_synthesis=True, storm_control=storm_control
        )
        synthesizer.run(optimum_threshold=None)
        # assert synthesizer.storm_control.latest_paynt_result_fsc is not None
        fsc = synthesizer.storm_control.latest_paynt_result_fsc
        return fsc

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
    
    def create_subfamily(self, hole_combinations):
        return [self.pomdp_sketch.family.construct_assignment(hole_combination) for hole_combination in hole_combinations]
    
    def create_random_subfamily(self, family_size : int):
        hole_combinations = random.choices(list(self.pomdp_sketch.family.all_combinations()), k = family_size)
        return self.create_subfamily(hole_combinations)

    def stratified_subfamily_sampling(self, family_size : int):
        options = [self.pomdp_sketch.family.hole_options(hole) for hole in range(self.pomdp_sketch.family.num_holes)]
        lb = [min(xs) for xs in options]
        ub = [max(xs) + 1 for xs in options]
        from scipy.stats import qmc
        sampler = qmc.LatinHypercube(d=len(options))
        samples = sampler.random(n=5)
        hole_combination_samples = qmc.scale(samples, lb, ub).astype(int)
        return self.create_subfamily(hole_combination_samples)
    
    def deterministic_fsc_to_stochastic_fsc(self, fsc):
        for n in range(fsc.num_nodes):
            for o in range(self.nO):
                if o >= len(fsc.action_function[n]):
                    fsc.action_function[n].append({int(a) : 1 / len(self.pomdp_sketch.observation_to_actions[o]) for a in self.pomdp_sketch.observation_to_actions[o]})
                    fsc.update_function[n].append({int(m) : 1 / fsc.num_nodes for m in range(fsc.num_nodes)})
                else:
                    a = fsc.action_function[n][o]
                    action_label = fsc.action_labels[a]
                    family_action = self.pomdp_sketch.action_labels.index(action_label)
                    assert family_action in self.pomdp_sketch.observation_to_actions[o]
                    fsc.action_function[n][o] = {int(family_action) : 1.0}
                    fsc.update_function[n][o] = {int(fsc.update_function[n][o]) : 1.0}

        assert all([len(fsc.action_function[n]) == self.nO for n in range(fsc.num_nodes)])
        fsc.is_deterministic = False
        return fsc

    def experiment_on_subfamily(self, hole_assignments_to_test : list, num_nodes : int, method : Method, timeout=15, num_gd_iterations=1000, evaluate_on_whole_family=False):
        results = {}

        nO = self.pomdp_sketch.num_observations

        dummy = []
        
        pomdps = []

        for i, assignment in enumerate(hole_assignments_to_test):
            
            if method.value == method.GRADIENT.value:
                value, resolution, action_function_params, memory_function_params, *_ = self.gradient_descent_on_single_pomdp_from_hole_assignment(assignment, num_gd_iterations, num_nodes, timeout=timeout, parameter_resolution={}, resolution={}, action_function_params={}, memory_function_params={})
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
                    fsc = self.solve_pomdp_saynt(pomdp, specification, num_nodes, timeout=timeout)
                else:
                    fsc = self.solve_pomdp_paynt(pomdp, specification, num_nodes, timeout=timeout)

                fsc = self.deterministic_fsc_to_stochastic_fsc(fsc)
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
    def construct_pmc(self, pomdp, pomdp_sketch, reward_model_name, num_nodes, action_function_params = {}, memory_function_params = {}, resolution = {}, parameter_resolution = None, distinct_parameters_for_final_probability = False, sanity_checks = False, memory_model = None):
        # pycarl.clear_pools()
        
        builder : stormpy.storage.storage.ParametricSparseMatrixBuilder = stormpy.storage.ParametricSparseMatrixBuilder()
        
        print(memory_model)
        
        counter = 0
        
        seen = set()
        
        pmc_transitions = {}
        
        rewards = {}

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
                N = num_nodes
                for n in range(N):
                    sMC = s * N + n
                    states.add(sMC)
                    if sMC not in pmc_transitions:
                        pmc_transitions[sMC] = {}
                    for label in state.labels:
                        labeling[label].append(sMC)

                    for transition in action.transitions:
                        t = transition.column
                        t_prob = transition.value()
                        M = N
                        for m in range(M):
                            tMC = t * M + m
                            states.add(tMC)
                            act_tup = (n, o, quotient_action)
                            mem_tup = (n, o, m)

                            if sanity_checks: 
                                assert (sMC, tMC, a) not in seen, (sMC, tMC, seen)
                                seen.add((sMC, tMC, a))

                            if act_tup in action_function_params:
                                act_param = action_function_params[act_tup]
                            elif (memory_model[o] < num_nodes and (act_tup[0] % memory_model[o], act_tup[1], act_tup[2]) in action_function_params):
                                # Here, we duplicate existing parameters in case of a non-flat memory model.
                                act_param = action_function_params[(act_tup[0] % memory_model[o], act_tup[1], act_tup[2])]
                                action_function_params[act_tup] = act_param
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
                                    if self.use_softmax: parameter_resolution[act_param] = random.gauss(mu=0, sigma=1)
                                    resolution[act_param] = pc.Rational(1 / len(pomdp_sketch.observation_to_actions[o]))
                                    counter += 1

                                action_function_params[act_tup] = act_param

                            if mem_tup in memory_function_params:
                                mem_param = memory_function_params[mem_tup]
                            elif (memory_model[o] < num_nodes and (mem_tup[0] % memory_model[o], mem_tup[1], mem_tup[2]) in memory_function_params):
                                # Here, we duplicate existing parameters in case of a non-flat memory model.
                                mem_param = memory_function_params[(mem_tup[0] % memory_model[o], mem_tup[1], mem_tup[2])]
                                memory_function_params[mem_tup] = mem_param
                            else:
                                if not distinct_parameters_for_final_probability and m == M-1:
                                    mem_param = pc.Rational(1)
                                    for m_ in range(M):
                                        if m != m_:
                                            assert (n, o, m_) in memory_function_params
                                            mem_param -= memory_function_params[n, o, m_]
                                    memory_function_params[mem_tup] = mem_param
                                else:
                                    p_o_name = f"p{counter}_n{n}_o{o}_m{m}"
                                    assert pycarl.variable_with_name(p_o_name).is_no_variable, (p_o_name, action_function_params)
                                    mem_param = pycarl.Variable(p_o_name)
                                    resolution[mem_param] = pc.Rational(1 / M)
                                    if self.use_softmax: parameter_resolution[mem_param] = random.gauss(mu=0, sigma=1)
                                    memory_function_params[mem_tup] = mem_param
                                    counter += 1

                            action_poly = pc.Polynomial(act_param)
                            mem_poly = pc.Polynomial(mem_param)
                            action_mem_poly = action_poly * mem_poly * pc.Rational(float(t_prob))

                            if tMC in pmc_transitions[sMC]:
                                pmc_transitions[sMC][tMC] += action_mem_poly
                            else:
                                pmc_transitions[sMC][tMC] = action_mem_poly

                    if sMC in rewards:
                        rewards[sMC] += pc.Polynomial(action_function_params[(n, o, quotient_action)]) * pc.Rational(float(reward))
                    else:
                        rewards[sMC] = pc.Polynomial(action_function_params[(n, o, quotient_action)]) * pc.Rational(float(reward))
        
        if self.use_softmax:
            probabilistic_resolution = self.resolution_to_softmax(action_function_params, memory_function_params, parameter_resolution, num_nodes)
            resolve = probabilistic_resolution
        else:
            resolve = resolution
        
        cache = pycarl.cln.cln._FactorizationCache()
        for s, next_states in sorted(pmc_transitions.items(), key = lambda x : x[0]):
            if sanity_checks:
                all_floats = [float(pc.FactorizedRationalFunction(pc.FactorizedPolynomial(probability_function, cache), denom).evaluate(resolve)) for probability_function in next_states.values()] 
                assert all([f > 0 and f <= 1 for f in all_floats]), all_floats
                total = sum(all_floats)
                # print(s, next_states, total)
                assert next_states == pmc_transitions[s], (next_states, pmc_transitions[s])
                assert list(next_states.values()) == list(pmc_transitions[s].values()), (next_states.values(), pmc_transitions[s].values())
                assert np.isclose(total, 1), (s, next_states, total)
                # if s == 6:
                    # exit()
            rewards[s] = pc.FactorizedRationalFunction(pc.FactorizedPolynomial(rewards[s], cache), denom)
            for t, probability_function in sorted(next_states.items(), key = lambda x : x[0]):
                parametric_transition = pc.FactorizedRationalFunction(pc.FactorizedPolynomial(probability_function, cache), denom)
                # evaluation = float(parametric_transition.evaluate(resolution))
                builder.add_next_value(s, t, parametric_transition)
                if sanity_checks: 
                    assert parametric_transition.evaluate(resolve) > 0 and parametric_transition.evaluate(resolve) <= 1
                    assert probability_function.evaluate(resolve) > 0 and probability_function.evaluate(resolve) <= 1

        del pmc_transitions
        
        gc.collect()
        
        print("Building pDTMC transition matrix:")
        p_matrix = builder.build()
        print("Done.")

        if sanity_checks:
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
                action_params = [action_function_params[(n,o,a)] for a in range(self.nA) if (n,o,a) in action_function_params]
                if action_params == []:
                    continue
                assert len(action_params) > 0, action_function_params
                action_parameter_values = np.array([float(parameter_resolution[var]) for var in action_params if var in parameter_resolution])
                if action_parameter_values.size == 0:
                    continue
                assert action_parameter_values.size > 0, (n, o, action_params, action_parameter_values)
                softmax_action_probs = stablesoftmax(action_parameter_values)
                assert math.isclose(sum(softmax_action_probs), 1)
                for var, softmax_prob in zip(action_params, softmax_action_probs):
                    assert softmax_prob > 0 and softmax_prob <= 1, (softmax_action_probs, action_parameter_values)
                    probabilistic_resolution[var] = pycarl.cln.cln.Rational(softmax_prob)
                    
                node_params = [memory_function_params[(n,o,m)] for m in range(num_nodes) if (n,o,m) in memory_function_params]
                if node_params == []:
                    continue
                memory_parameter_values = np.array([float(parameter_resolution[var]) for var in node_params if var in parameter_resolution])
                softmax_memory_probs = stablesoftmax(np.array([float(parameter_resolution[var]) for var in node_params]))
                assert math.isclose(sum(softmax_memory_probs), 1)
                for var, softmax_prob in zip(node_params, softmax_memory_probs):
                    assert softmax_prob > 0 and softmax_prob <= 1, (softmax_memory_probs, memory_parameter_values)
                    probabilistic_resolution[var] = pycarl.cln.cln.Rational(softmax_prob)
        
        return probabilistic_resolution

    def softmax_gradients(self, action_function_params, memory_function_params, parameter_resolution, num_nodes, gradients):
        softmax_grad = {}
        for n in range(num_nodes):
            for o in range(self.nO):
                action_params = [action_function_params[n,o,a] for a in range(self.nA) if (n,o,a) in action_function_params]
                if action_params == []:
                    continue
                
                action_parameter_values = np.array([float(parameter_resolution[var]) for var in action_params if var in parameter_resolution])
                if action_parameter_values.size == 0:
                    continue
                softmax_action_probs = stablesoftmax(action_parameter_values)
                
                for var, softmax_prob in zip(action_params, softmax_action_probs):
                    softmax_grad[var] = sum([gradients[var_j] * ((softmax_prob * (1 - softmax_prob_j)) if var == var_j else (-softmax_prob * softmax_prob_j)) for var_j, softmax_prob_j in zip(action_params, softmax_action_probs) if var_j in gradients])
                
                node_params = [memory_function_params[n,o,m] for m in range(num_nodes) if (n,o,m) in memory_function_params]
                if node_params == []:
                    continue

                softmax_memory_probs = stablesoftmax(np.array([float(parameter_resolution[var]) for var in node_params]))

                for var, softmax_prob in zip(node_params, softmax_memory_probs):
                    softmax_grad[var] = sum([gradients[var_j] * ((softmax_prob * (1 - softmax_prob_j)) if var == var_j else (-softmax_prob * softmax_prob_j)) for var_j, softmax_prob_j in zip(node_params, softmax_memory_probs) if var_j in gradients])

        return softmax_grad
    
    def clip_gradient(self, x, doclip=True):
        if doclip:
            return np.clip(x, -self.clip_gradient_value, self.clip_gradient_value)
        else:
            return x

    def gradient_descent_on_single_pomdp_from_hole_assignment(self, hole_assignment, num_iters : int, num_nodes : int, **kwargs):
        print("Building pDTMC for POMDP:", str(hole_assignment))
        pomdp_class = self.pomdp_sketch.build_pomdp(hole_assignment)
        pomdp = pomdp_class.model
        return self.gradient_descent_on_single_pomdp(pomdp, num_iters, num_nodes, **kwargs)

    def gradient_descent_on_single_pomdp(self, pomdp, num_iters : int, num_nodes : int, action_function_params = {}, memory_function_params = {}, resolution = {}, parameter_resolution = None, timeout = None, memory_model = None):
        # print("BEFORE")
        # print("PARAM", parameter_resolution, sep='\n')
        # print("RESOL", resolution, sep='\n')
        pmc, action_function_params, memory_function_params, resolution, parameter_resolution = self.construct_pmc(pomdp, self.pomdp_sketch, self.reward_model_name, num_nodes, distinct_parameters_for_final_probability=self.use_softmax, parameter_resolution=parameter_resolution, resolution=resolution, action_function_params=action_function_params, memory_function_params=memory_function_params, memory_model=memory_model)
        # print("AFTER")
        # print("PARAM", parameter_resolution, sep='\n')
        # print("RESOL", resolution, sep='\n')
        current_parameters = list(pmc.collect_all_parameters())
        
        if self.use_softmax:
            resolution = self.resolution_to_softmax(action_function_params, memory_function_params, parameter_resolution, num_nodes)
            # self.sanity_check_pmc_at_instantiation(pmc, resolution)
            checker = payntbind.synthesis.SparseDerivativeInstantiationModelCheckerFamily(pmc)
            task = stormpy.ParametricCheckTask(self.pomdp_sketch.get_property().formula, only_initial_states=False)
            checker.specifyFormula(stormpy.Environment(), task)
        else:
            wrapper = payntbind.synthesis.GradientDescentInstantiationSearcherFamily(pmc, self.lr, self.mbs, self.gd_steps)
            wrapper.setup(self.env, self.synth_task)
            wrapper.resetDynamicValues()

        action_function_params_no_const = {index : var for index, var in action_function_params.items() if isinstance(var, pycarl.Variable)}
        memory_function_params_no_const = {index : var for index, var in memory_function_params.items() if isinstance(var, pycarl.Variable)}
        for p in pmc.collect_all_parameters():
            assert p in action_function_params_no_const.values() or p in memory_function_params_no_const.values()
        
        assert set(pmc.collect_all_parameters()) <= set(list(resolution.keys())), "\n".join(["The parameters of the pDTMC:", str(pmc.collect_all_parameters()), "should be a subset of the probabilistic resolution mapping of the parameters:", str(list(resolution.keys()))])
        
        if timeout:
            tik = time.time()
            
        instantiator = stormpy.pars.PDtmcInstantiator(pmc)
        instantiated_model = instantiator.instantiate(resolution)
        result = stormpy.model_checking(instantiated_model, self.pomdp_sketch.get_property().property.raw_formula)
        
        all_seen_parameters = list(resolution.keys())
        
        print("Before GD value:", result.at(0), 'Number of parameters:', len(current_parameters))
        
        if self.use_momentum and (self.reset_momentum or self.momentum is None):
            self.momentum = defaultdict(lambda : 0)
            # self.momentum = dict(zip(all_seen_parameters, [0] * len(all_seen_parameters)))

        direction_operator = operator.sub if self.minimizing else operator.add
        
        for i in range(num_iters):
            if self.use_softmax:
                new_resolution = {}
                new_parameter_resolution = {}
                grads = {}
                grads = checker.checkMultipleParameters(self.env, resolution, current_parameters, result.get_values())
                softmax_grads = self.softmax_gradients(action_function_params, memory_function_params, parameter_resolution, num_nodes, grads)
                for p in current_parameters:
                    if self.use_momentum:
                        self.momentum[p] = self.beta * self.momentum[p] + (1 - self.beta) * self.clip_gradient(softmax_grads[p], doclip=True)
                        new_parameter_resolution[p] = direction_operator(float(parameter_resolution[p]), self.lr * self.momentum[p])
                    else:
                        new_parameter_resolution[p] = direction_operator(float(parameter_resolution[p]), self.lr * self.clip_gradient(softmax_grads[p], doclip=True))
                # parameter_resolution.update(new_parameter_resolution)
                new_resolution = self.resolution_to_softmax(action_function_params, memory_function_params, new_parameter_resolution, num_nodes)
                # resolution.update(new_resolution)
                instantiator = stormpy.pars.PDtmcInstantiator(pmc)
                instantiated_model = instantiator.instantiate(new_resolution)
                result = stormpy.model_checking(instantiated_model, self.pomdp_sketch.get_property().property.raw_formula)
                current_value = result.at(0)
                parameter_resolution = new_parameter_resolution
                resolution = new_resolution
            else:
                try:
                    # wrapper.resetDynamicValues() # TODO
                    current_value, new_resolution = wrapper.stochasticGradientDescent(resolution)
                    # resolution.update(new_resolution)
                    resolution = new_resolution
                except Exception as e:
                    print([float(x) for x in resolution.values()])
                    self.sanity_check_pmc_at_instantiation(pmc, resolution)
                    raise e
            print(i, current_value)
            self.gd_trace.append(current_value)
            if timeout and time.time() - tik > timeout:
                break
        
        return current_value, resolution, action_function_params, memory_function_params, parameter_resolution

    
    def run_gradient_descent_on_family(self, num_iters : int, num_nodes : int, assignments : list = None, timeout : int = None, random_selection : bool = False, memory_model : list[int] = None):
        current_value = None
        
        if self.dynamic_memory:
            assert num_nodes == 1
            assert memory_model is None
            memory_model_cache = {}
        
        if memory_model is None:
            memory_model = [num_nodes] * self.nO
        
        assert max(memory_model) == num_nodes
        
        action_function_params = {}
        memory_function_params = {}
        
        best_family_value = 1e30 if self.minimizing else -1e30
        op = operator.lt if self.minimizing else operator.gt
        
        best_fsc = None
        
        if assignments is None:
            print("Running on 'entire' family of size:", )
        else:
            print("Running on sub-family of size:", len(assignments))
        
        self.current_values = []
        resolution = {}
        
        parameter_resolution = {} if self.use_softmax else None
        
        if timeout: tik = time.time()
        
        for i in range(num_iters):
            
            artificial_upper_bound = None # if current_value is None else current_value * (0.9 if self.minimizing else 1.1)
            
            if i > 0:
                fsc = self.parameters_to_paynt_fsc(action_function_params, memory_function_params, resolution, num_nodes, self.nO, self.pomdp_sketch.observation_to_actions, memory_model=memory_model)
                dtmc_sketch =  self.pomdp_sketch.build_dtmc_sketch(fsc, negate_specification=True)
                hole_assignment, paynt_value = self.paynt_call(dtmc_sketch, assignments=assignments, artificial_upper_bound=artificial_upper_bound, random_selection=random_selection)
                print("Paynt value:", paynt_value, "previous family best:", best_family_value)
                self.family_trace.append(paynt_value)
                if op(paynt_value, best_family_value):
                    best_fsc = fsc
                    best_family_value = paynt_value
                if timeout and time.time() - tik > timeout: break
            else:
                hole_assignment = self.pomdp_sketch.family.pick_any()
            
            pomdp_class = self.pomdp_sketch.build_pomdp(hole_assignment)
            pomdp = pomdp_class.model
            
            if self.dynamic_memory and i % 2 == 0:
                
                hole_assignment_str = str(hole_assignment)
                
                if hole_assignment_str in memory_model_cache:
                    suggest_memory_model = memory_model_cache[hole_assignment_str]
                    print("REUSING PREVIOUSLY COMPUTED MEMORY MODEL FOR", hole_assignment_str)
                else:
                    print("COMPUTING MEMORY MODEL FOR", hole_assignment_str)
                    saynt_controller : paynt.quotient.fsc.FSC = self.solve_pomdp_saynt(pomdp, self.pomdp_sketch.specification.copy(), num_nodes, timeout=10)
                    if saynt_controller is None or saynt_controller.memory_model is None:
                        suggest_memory_model = None
                    else:
                        suggest_memory_model = saynt_controller.memory_model
                    memory_model_cache[hole_assignment_str] = suggest_memory_model
                
                if suggest_memory_model is not None:
                    # suggested_increase = np.array(suggest_memory_model) > np.array(memory_model[:len(suggest_memory_model)])
                    suggested_increase = np.array([suggest_memory_model[o] > memory_model[o] for o in range(len(suggest_memory_model))])
                    print(suggested_increase, np.nonzero(suggested_increase)[0])
                    if (suggested_increase).any():
                        # obs_to_increase_memory = random.choice(np.nonzero(suggested_increase)[0])
                        for obs_to_increase_memory in np.nonzero(suggested_increase)[0]:
                            memory_model[obs_to_increase_memory] += 1
                        num_nodes = max(memory_model)
                        # parameter_resolution = {}
                        # resolution = {}
            
            # current_value, new_resolution, action_function_params, memory_function_params, new_parameter_resolution = self.gradient_descent_on_single_pomdp_from_hole_assignment(hole_assignment, 10 // self.gd_steps, num_nodes, action_function_params=action_function_params, memory_function_params=memory_function_params, resolution=resolution, parameter_resolution=parameter_resolution, memory_model=memory_model)
            current_value, new_resolution, action_function_params, memory_function_params, new_parameter_resolution = self.gradient_descent_on_single_pomdp(pomdp, 10 // self.gd_steps, num_nodes, action_function_params=action_function_params, memory_function_params=memory_function_params, resolution=resolution, parameter_resolution=parameter_resolution, memory_model=memory_model)
            
            self.current_values.append(current_value)
            
            print(f"{i} | Latest GD value: {current_value}")

            resolution.update(new_resolution)
            if self.use_softmax:
                parameter_resolution.update(new_parameter_resolution)
        
        return best_fsc, best_family_value
    
    def get_values_on_subfamily(self, dtmc_sketch, assignments) -> np.ndarray:
        synthesizer = paynt.synthesizer.synthesizer_onebyone.SynthesizerOneByOne(dtmc_sketch)
        evaluations = np.zeros(len(assignments))
        for j, family in enumerate(assignments):
            evaluations[j] = synthesizer.evaluate(family, keep_value_only=True)[0]
        return evaluations
    
    def get_dtmc_sketch(self, fsc):
        return self.pomdp_sketch.build_dtmc_sketch(fsc, negate_specification=True)
    
    def paynt_call_given_fsc(self, fsc, **kwargs) -> tuple[paynt.family.family.Family, float]:
        return self.paynt_call(self.get_dtmc_sketch(fsc), **kwargs)
    
    def paynt_call(self, dtmc_sketch, assignments = None, artificial_upper_bound = None, random_selection = False) -> tuple[paynt.family.family.Family, float]:
        if assignments is None:
            synthesizer = paynt.synthesizer.synthesizer_ar.SynthesizerAR(dtmc_sketch)
            hole_assignment = synthesizer.synthesize(optimum_threshold=artificial_upper_bound, keep_optimum=True)
            paynt_value = synthesizer.best_assignment_value
            if random_selection: 
                hole_assignment = self.pomdp_sketch.family.pick_random()
        else:
            assert assignments is not None
            evaluations = self.get_values_on_subfamily(dtmc_sketch, assignments)
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
                    assert np.isclose(summation, 1), (summation)
                    
            instantiator = stormpy.pars.PDtmcInstantiator(pmc)
            instantiated_model = instantiator.instantiate(resolution)
            print(instantiated_model)
                    
            print("pMC shoud be ok.")
