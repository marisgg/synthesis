# using Paynt for POMDP sketches

import paynt.cli
import paynt.parser.sketch
import paynt.quotient.pomdp_family
import paynt.quotient.fsc
import paynt.synthesizer.synthesizer_onebyone
import paynt.synthesizer.synthesizer_ar

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
    print(pomdp)
    updated = payntbind.synthesis.restoreActionsInAbsorbingStates(pomdp)
    if updated is not None: pomdp = updated
    print(pomdp)
    pomdp,choice_to_true_action = payntbind.synthesis.enableAllActions(pomdp)
    action_labels,choice_to_action = payntbind.synthesis.extractActionLabels(pomdp)
    print(pomdp)

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
    profiling = False
    if profiling:
        profiler = cProfile.Profile()
        profiler.enable()

    # random.seed(11)

    # enable PAYNT logging
    paynt.cli.setup_logger()

    # load sketch
    # project_path="models/pomdp/sketches/obstacles"
    project_path="models/pomdp/sketches/avoid-discounted"
    pomdp_sketch = load_sketch(project_path)

    # construct POMDP from the given hole assignment
    hole_assignment = pomdp_sketch.family.pick_any()
    pomdp = assignment_to_pomdp(pomdp_sketch,hole_assignment)

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