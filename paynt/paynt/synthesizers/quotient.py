import stormpy
import stormpy.synthesis
import stormpy.pomdp

import math
import re
import itertools
import random
from collections import OrderedDict

from .statistic import Statistic

from ..sketch.jani import JaniUnfolder
from ..sketch.property import Property
from ..sketch.holes import Hole,Holes,DesignSpace,CombinationColoring

from ..profiler import Profiler

from .models import MarkovChain,MDP,DTMC

import logging
logger = logging.getLogger(__name__)


class QuotientContainer:

    # holes with avg choice value difference below this threshold will be considered consistent
    inconsistency_threshold = 0

    def __init__(self, sketch):
        self.sketch = sketch
        
        self.quotient_mdp = None
        self.default_actions = None        
        self.action_to_hole_options = None
        self._quotient_relevant_holes = None

        self.discarded = None

    def select_actions(self, design_space):
        ''' Select actions relevant in the provided design space. '''
        Profiler.start("quotient::select_actions")
        selected_actions = stormpy.BitVector(self.default_actions)
        for act_index in range(self.quotient_mdp.nr_choices):
            if selected_actions.get(act_index):
                continue
            hole_options = self.action_to_hole_options[act_index]
            if design_space.includes(hole_options):
                selected_actions.set(act_index)
        Profiler.resume()
        return selected_actions

    def restrict_quotient(self, selected_actions):
        '''
        Restrict the quotient MDP to the selected actions.
        :return (1) the restricted model
        :return (2) sub- to full state mapping
        :return (3) sub- to full action mapping
        '''
        
        # construct the submodel
        Profiler.start("quotient::restrict_quotient")
        keep_unreachable_states = False
        subsystem_options = stormpy.SubsystemBuilderOptions()
        subsystem_options.build_state_mapping = True
        subsystem_options.build_action_mapping = True
        all_states = stormpy.BitVector(self.quotient_mdp.nr_states, True)
        submodel_construction = stormpy.construct_submodel(
            self.quotient_mdp, all_states, selected_actions, keep_unreachable_states, subsystem_options
        )
        model = submodel_construction.model
        state_map =  submodel_construction.new_to_old_state_mapping
        choice_map = submodel_construction.new_to_old_action_mapping
        Profiler.resume()
        return model,state_map,choice_map

    def build(self, design_space = None):
        if design_space is None or design_space == self.sketch.design_space:
            return MDP(self.quotient_mdp, self, None, None, self.sketch.design_space)
        selected_actions = self.select_actions(design_space)
        model,state_map,choice_map = self.restrict_quotient(selected_actions)
        return MDP(model, self, state_map, choice_map, design_space)

    def build_chain(self, design_space):
        assert design_space.size == 1

        # restrict quotient
        selected_actions = self.select_actions(design_space)
        sub_mdp,state_map,choice_map = self.restrict_quotient(selected_actions)
        
        # convert restricted MDP to DTMC
        tm = sub_mdp.transition_matrix
        tm.make_row_grouping_trivial()
        components = stormpy.storage.SparseModelComponents(tm, sub_mdp.labeling, sub_mdp.reward_models)
        dtmc = stormpy.storage.SparseDtmc(components)

        return DTMC(dtmc,self,state_map,choice_map)

    def scheduler_selection(self, mdp, scheduler):
        ''' Get hole options involved in the scheduler selection. '''
        Profiler.start("quotient::scheduler_selection")

        assert scheduler.memoryless and scheduler.deterministic
        
        selection = [set() for hole_index in mdp.design_space.hole_indices]
        choice_selection = scheduler.compute_action_support(mdp.model.nondeterministic_choice_indices)
        for choice in choice_selection:
            global_choice = mdp.quotient_choice_map[choice]
            choice_options = self.action_to_hole_options[global_choice]
            for hole_index,option in choice_options.items():
                selection[hole_index].add(option)
        selection = [list(options) for options in selection]

        Profiler.resume()
        return selection

    def scheduler_selection_quantitative(self, mdp, prop, result):
        '''
        Get hole options involved in the scheduler selection.
        Use numeric values to filter spurious inconsistencies.
        '''
        Profiler.start("quotient::scheduler_selection_quantitative")

        # get qualitative scheduler selection, filter inconsistent assignments        
        selection = self.scheduler_selection(mdp, result.scheduler)
        inconsistent_assignments = {hole_index:options for hole_index,options in enumerate(selection) if len(options) > 1 }
        if len(inconsistent_assignments) == 0:
            return selection,None
        
        choice_values = stormpy.synthesis.multiply_with_vector(mdp.model.transition_matrix,result.get_values())
        # for each hole, compute its difference sum and a number of affected states
        hole_difference_sum = {hole_index: 0 for hole_index in inconsistent_assignments}
        hole_states_affected = {hole_index: 0 for hole_index in inconsistent_assignments}
        tm = mdp.model.transition_matrix

        if prop.reward:
            # if the associated reward model has state-action rewards, then these must be added to choice values
            reward_name = prop.formula.reward_name
            rm = mdp.model.reward_models.get(reward_name)
            assert not rm.has_transition_rewards and (rm.has_state_rewards != rm.has_state_action_rewards)
            if rm.has_state_action_rewards:
                choice_rewards = list(rm.state_action_rewards)
                assert mdp.choices == len(choice_rewards)
                for choice in range(mdp.choices):
                    choice_values[choice] += choice_rewards[choice]

        for state in range(mdp.states):

            # for this state, compute for each inconsistent hole the difference in choice values between respective options
            hole_min = {hole_index: None for hole_index in inconsistent_assignments}
            hole_max = {hole_index: None for hole_index in inconsistent_assignments}

            for choice in range(tm.get_row_group_start(state),tm.get_row_group_end(state)):
                
                choice_global = mdp.quotient_choice_map[choice]
                if self.default_actions.get(choice_global):
                    continue
                choice_options = self.action_to_hole_options[choice_global]
                
                # collect holes in which this action is inconsistent
                inconsistent_holes = []
                for hole_index,option in choice_options.items():
                    inconsistent_options = inconsistent_assignments.get(hole_index,set())
                    if option in inconsistent_options:
                        inconsistent_holes.append(hole_index)

                value = choice_values[choice]
                for hole_index in inconsistent_holes:
                    current_min = hole_min[hole_index]
                    if current_min is None or value < current_min:
                        hole_min[hole_index] = value
                    current_max = hole_max[hole_index]
                    if current_max is None or value > current_max:
                        hole_max[hole_index] = value

            # compute the difference
            for hole_index,min_value in hole_min.items():
                if min_value is None:
                    continue
                max_value = hole_max[hole_index]
                difference = max_value - min_value
                if math.isnan(difference):
                    assert max_value == min_value and min_value == math.inf
                    difference = 0
                hole_difference_sum[hole_index] += difference
                hole_states_affected[hole_index] += 1

        # aggregate
        inconsistent_differences = {
            hole_index: (hole_difference_sum[hole_index] / hole_states_affected[hole_index])
            for hole_index in inconsistent_assignments
            }

        # filter differences below epsilon
        for hole_index in inconsistent_assignments:
            if inconsistent_differences[hole_index] <= QuotientContainer.inconsistency_threshold:
                selection[hole_index] = [selection[hole_index][0]]

        # sanity check
        for difference in inconsistent_differences.values():
            assert not math.isnan(difference)

        Profiler.resume()
        return selection,inconsistent_differences
        

    def scheduler_consistent(self, mdp, prop, result):
        '''
        Get hole assignment induced by this scheduler and fill undefined
        holes by some option from the design space of this mdp.
        :return hole assignment
        :return whether the scheduler is consistent
        '''
        # selection = self.scheduler_selection(mdp, result.scheduler)
        selection,scores = self.scheduler_selection_quantitative(mdp, prop, result)
        consistent = True
        for hole_index in mdp.design_space.hole_indices:
            options = selection[hole_index]
            if len(options) > 1:
                consistent = False
            if options == []:
                selection[hole_index] = [mdp.design_space[hole_index].options[0]]

        return selection,scores,consistent

    def suboptions_half(self, mdp, splitter):
        ''' Split options of a splitter into to halves. '''
        options = mdp.design_space[splitter].options
        half = len(options) // 2
        suboptions = [options[:half], options[half:]]
        return suboptions

    def suboptions_unique(self, mdp, splitter, used_options):
        ''' Distribute used options of a splitter into different suboptions. '''
        assert len(used_options) > 1
        suboptions = [[option] for option in used_options]
        index = 0
        for option in mdp.design_space[splitter].options:
            if option in used_options:
                continue
            suboptions[index].append(option)
            index = (index + 1) % len(suboptions)
        return suboptions

    def suboptions_enumerate(self, mdp, splitter, used_options):
        assert len(used_options) > 1
        core_suboptions = [[option] for option in used_options]
        other_suboptions = [option for option in mdp.design_space[splitter].options if option not in used_options]
        return core_suboptions, other_suboptions

    def holes_with_max_score(self, hole_score):
        max_score = max(hole_score.values())
        with_max_score = [hole_index for hole_index in hole_score if hole_score[hole_index] == max_score]
        return with_max_score

    def most_inconsistent_holes(self, scheduler_assignment):
        num_definitions = [len(options) for options in scheduler_assignment]
        most_inconsistent = self.holes_with_max_score(num_definitions) 
        return most_inconsistent

    def discard(self, mdp, hole_assignments, core_suboptions, other_suboptions):

        # default result
        reduced_design_space = mdp.design_space.copy()
        if len(other_suboptions) == 0:
            suboptions = core_suboptions
        else:
            suboptions = [other_suboptions] + core_suboptions  # DFS solves core first

        # reduce simple holes
        # ds_before = reduced_design_space.size
        # for hole_index in reduced_design_space.hole_indices:
        #     if mdp.hole_simple[hole_index]:
        #         assert len(hole_assignments[hole_index]) == 1
        #         reduced_design_space.assume_hole_options(hole_index, hole_assignments[hole_index])
        # ds_after = reduced_design_space.size
        # self.discarded += ds_before - ds_after

        # discard other suboptions
        # suboptions = core_suboptions
        # self.discarded += (reduced_design_space.size * len(other_suboptions)) / (len(other_suboptions) + len(core_suboptions))

        return reduced_design_space, suboptions


    def split(self, mdp):
        assert not mdp.is_dtmc
        Profiler.start("quotient::split")

        # split family wrt last undecided result
        assert mdp.scheduler_results is not None
        hole_assignments,scores = mdp.scheduler_results[next(reversed(mdp.scheduler_results))]
        assert scores is not None
        
        # scores = [scores[hole_index] if hole_index in scores else 0 for hole_index in mdp.design_space.hole_indices]
        
        # inconsistent = self.most_inconsistent_holes(hole_assignments)
        # hole_sizes = [mdp.design_space[hole_index].size if hole_index in inconsistent else 0 for hole_index in mdp.design_space.hole_indices]
        # splitters = self.holes_with_max_score(hole_sizes)

        splitters = self.holes_with_max_score(scores)
        splitter = splitters[0]
        assert len(hole_assignments[splitter]) > 1
        core_suboptions,other_suboptions = self.suboptions_enumerate(mdp, splitter, hole_assignments[splitter])

        new_design_space, suboptions = self.discard(mdp, hole_assignments, core_suboptions, other_suboptions)
        
        # construct corresponding design subspaces
        Profiler.start("    create subspaces")
        design_subspaces = []
        for suboption in suboptions:
            design_subspace = new_design_space.copy()
            design_subspace.assume_hole_options(splitter, suboption)
            design_subspace.refinement_depth = mdp.design_space.refinement_depth + 1
            design_subspaces.append(design_subspace)
        Profiler.resume()

        Profiler.resume()
        return design_subspaces

    @property
    def quotient_relevant_holes(self):
        if self._quotient_relevant_holes is not None:
            return self._quotient_relevant_holes

        tm = self.quotient_mdp.transition_matrix
        self._quotient_relevant_holes = []
        for state in range(self.quotient_mdp.nr_states):
            relevant_holes = set()
            for action in range(tm.get_row_group_start(state),tm.get_row_group_end(state)):
                relevant_holes.update(set(self.action_to_hole_options[action].keys()))
            self._quotient_relevant_holes.append(relevant_holes)
        return self._quotient_relevant_holes

    def double_check_assignment(self, assignment, opt_prop):
        '''
        Double-check whether this assignment truly improves optimum.
        :return singleton family if the assignment truly improves optimum
        '''
        assert assignment.size == 1
        dtmc = self.build_chain(assignment)
        opt_result = dtmc.model_check_property(opt_prop)
        if opt_prop.improves_optimum(opt_result.value):
            opt_prop.update_optimum(opt_result.value)
            return assignment
        else:
            return None



class DTMCQuotientContainer(QuotientContainer):
    
    def __init__(self, *args):
        super().__init__(*args)

        # unfold jani program
        unfolder = JaniUnfolder(self.sketch)
        self.sketch.specification = unfolder.specification

        # build quotient MDP       
        edge_to_hole_options = unfolder.edge_to_hole_options
        logger.debug("Constructing quotient MDP ... ")
        self.quotient_mdp = stormpy.build_sparse_model_with_options(unfolder.jani_unfolded, MarkovChain.builder_options)
        logger.debug(f"Constructed quotient MDP with {self.quotient_mdp.nr_states} states")

        # associate each action of a quotient MDP with hole options
        # remember default actions (actions taken in each hole assignment)
        # TODO handle overlapping colors
        num_choices = self.quotient_mdp.nr_choices

        self.default_actions = stormpy.BitVector(num_choices, False)
        self.action_to_hole_options = []
        tm = self.quotient_mdp.transition_matrix
        for choice in range(num_choices):
            edges = self.quotient_mdp.choice_origins.get_edge_index_set(choice)            
            hole_options = {}
            for edge in edges:
                combination = edge_to_hole_options.get(edge, None)
                if combination is None:
                    continue
                for hole_index,option in combination.items():
                    options = hole_options.get(hole_index,set())
                    options.add(option)
                    hole_options[hole_index] = options

            for hole_index,options in hole_options.items():
                assert len(options) == 1
            hole_options = {hole_index:list(options)[0] for hole_index,options in hole_options.items()}

            self.action_to_hole_options.append(hole_options)
            if len(hole_options) == 0:
                self.default_actions.set(choice)


class CTMCQuotientContainer(QuotientContainer):
    def __init__(self, *args):
        super().__init__(*args)


class MDPQuotientContainer(QuotientContainer):
    def __init__(self, *args):
        super().__init__(*args)

    def build_chain(self, assignment):
        model = self.sketch.restrict_prism(assignment)
        return MDP(model, self, None, None, assignment)


class POMDPQuotientContainer(QuotientContainer):

    def __init__(self, *args):
        super().__init__(*args)

        # default quotient attributes
        self.quotient_mdp = None
        self.default_actions = None        
        self.action_to_hole_options = None
        self._quotient_relevant_holes = None

        # POMDP attributes
        self.pomdp = None
        self.observation_labels = None
        self.actions_at_observation = None

        # (unfolded) quotient MDP attributes
        self.pomdp_manager = None

        # construct quotient POMDP
        if not self.sketch.is_explicit:
            MarkovChain.builder_options.set_build_choice_labels(True)
            self.pomdp = stormpy.build_sparse_model_with_options(self.sketch.prism, MarkovChain.builder_options)
            assert self.pomdp.labeling.get_states("overlap_guards").number_of_set_bits() == 0
        else:
            self.pomdp = self.sketch.explicit_model
        self.pomdp = stormpy.pomdp.make_canonic(self.pomdp)

        # ^ this also asserts that states with the same observation have the same number of available actions
        # print("observations: ", self.pomdp.observations)
        
        # extract observation labels
        if self.pomdp.has_observation_valuations():
            ov = self.pomdp.observation_valuations
            self.observation_labels = [ov.get_string(obs) for obs in range(self.observations)]
            self.observation_labels = [self.process_label(label) for label in self.observation_labels]
        else:
            self.observation_labels = list(range(self.observations))
        # print("observation labels: ", self.observation_labels)

        # compute actions available at each observation
        self.actions_at_observation = [0] * self.observations
        for state in range(self.pomdp.nr_states):
            obs = self.pomdp.observations[state]
            if self.actions_at_observation[obs] != 0:
                continue
            self.actions_at_observation[obs] = self.pomdp.get_nr_available_actions(state)
        # print("actions at observations: ", self.actions_at_observation)

        # collect labels of actions available at each observation
        self.action_labels_at_observation = [[] for obs in range(self.observations)]
        for state in range(self.pomdp.nr_states):
            obs = self.pomdp.observations[state]
            if self.action_labels_at_observation[obs] != []:
                continue
            actions = self.pomdp.get_nr_available_actions(state)
            for offset in range(actions):
                choice = self.pomdp.get_choice_index(state,offset)
                labels = self.pomdp.choice_labeling.get_labels_of_choice(choice)
                self.action_labels_at_observation[obs].append(labels)
        # print("labels of actions at observations: ", self.action_labels_at_observation)

        # create POMDP manager
        self.pomdp_manager = stormpy.synthesis.PomdpManager(self.pomdp)

    
    @property
    def observations(self):
        return self.pomdp.nr_observations

    def process_label(self,label):
        label = re.sub(r"\s+", "", label)
        label = label[1:-1]

        output = "[";
        first = True
        for p in label.split("&"):
            if not p.endswith("=0"):
                if first:
                    first = False
                else:
                    output += " & "
                output += p
        output += "]"
        return output

    def unfold_partial_memory(self):
        
        # reset attributes
        self.quotient_mdp = None
        self.default_actions = None        
        self.action_to_hole_options = None
        self._quotient_relevant_holes = None
        
        self.quotient_mdp = self.pomdp_manager.construct_mdp()

        # shortcuts
        pm = self.pomdp_manager
        pomdp = self.pomdp
        mdp = self.quotient_mdp
        
        # create holes
        holes = Holes()
        for hole_index in range(pm.num_holes):
            holes.append(None)

        # create action holes
        for obs,hole_indices in enumerate(pm.action_holes):
            obs_label = self.observation_labels[obs]
            action_labels = self.action_labels_at_observation[obs]
            for mem,hole_index in enumerate(hole_indices):
                name = "A({},{})".format(obs_label,mem)
                options = list(range(pm.hole_options[hole_index]))
                option_labels = [str(labels) for labels in action_labels]
                hole = Hole(name, options, option_labels)
                holes[hole_index] = hole

        # create memory holes
        for obs,hole_indices in enumerate(pm.memory_holes):
            obs_label = self.observation_labels[obs]
            for mem,hole_index in enumerate(hole_indices):
                name = "M({},{})".format(obs_label,mem)
                options = list(range(pm.hole_options[hole_index]))
                option_labels = [str(o) for o in options]
                hole = Hole(name, options, option_labels)
                holes[hole_index] = hole

        # create domains for each hole
        self.sketch.design_space = DesignSpace(holes)
        self.sketch.design_space.property_indices = self.sketch.specification.all_constraint_indices()
        
        # printo info about this unfolding
        print("# of observations:" , pomdp.nr_observations)
        print("# of holes: ", pm.num_holes)
        # print("action holes: ", pm.action_holes)
        # print("memory holes: ", pm.memory_holes)
        # print("hole options: ", pm.hole_options)
        print("design space size: ", self.sketch.design_space.size)
        print("", flush=True)

        # associate actions with hole combinations (colors)
        self.action_to_hole_options = []
        num_choices = mdp.nr_choices

        self.default_actions = stormpy.BitVector(num_choices, False)
        
        row_action_hole = list(pm.row_action_hole)
        row_memory_hole = list(pm.row_memory_hole)
        row_action_option = list(pm.row_action_option)
        row_memory_option = list(pm.row_memory_option)
        num_holes = pm.num_holes
        for row in range(num_choices):
            relevant_holes = {}
            action_hole = row_action_hole[row]
            if action_hole != num_holes:
                relevant_holes[action_hole] = row_action_option[row]
            memory_hole = row_memory_hole[row]
            if memory_hole != num_holes:
                relevant_holes[memory_hole] = row_memory_option[row]
            if not relevant_holes:
                self.action_to_hole_options.append({})
                self.default_actions.set(row)
                continue

            self.action_to_hole_options.append(relevant_holes)

