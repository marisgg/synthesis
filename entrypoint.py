import os
from pomdp_families import POMDPFamiliesSynthesis
from pomdp_families import Method

# Can try various models.
# project_path="models/pomdp/sketches/obstacles-8-3"
project_path="models/pomdp/sketches/obstacles-10-2"
# project_path="models/pomdp/sketches/avoid"
# project_path="models/pomdp/sketches/dpm"
# project_path="models/pomdp/sketches/rover"

def run_family():
    gd = POMDPFamiliesSynthesis(project_path, use_softmax=False, steps=10)
    gd.run_gradient_descent_on_family(1000, 2)

def run_family_softmax():
    gd = POMDPFamiliesSynthesis(project_path, use_softmax=True, steps=1, learning_rate=0.01)
    gd.run_gradient_descent_on_family(1000, 2)

def run_subfamily(subfamily_size = 10):
    gd = POMDPFamiliesSynthesis(project_path, use_softmax=True, steps=1, learning_rate=0.01)
    subfamily_assigments = gd.create_random_subfamily(subfamily_size)
    subfamily_other_results = gd.experiment_on_subfamily(subfamily_assigments, 2, Method.GRADIENT, num_gd_iterations=250, timeout=10, evaluate_on_whole_family=True) # breaks without softmax

    import pickle
    print(subfamily_other_results)
    dr = f"./outputs/{project_path.split('/')[-1]}/{subfamily_size}"
    os.makedirs(dr, exist_ok=True)
    with open(f"{dr}/gd.pickle", 'wb') as handle:
        pickle.dump(subfamily_other_results, handle)

    subfamily_other_results = gd.experiment_on_subfamily(subfamily_assigments, 2, Method.PAYNT, num_gd_iterations=250, timeout=10, evaluate_on_whole_family=True)

    import pickle
    print(subfamily_other_results)
    dr = f"./outputs/{project_path.split('/')[-1]}/{subfamily_size}/"
    os.makedirs(dr, exist_ok=True)
    with open(f"{dr}/paynt.pickle", 'wb') as handle:
        pickle.dump(subfamily_other_results, handle)

    best_gd_fsc, subfamily_gd_best_value = gd.run_gradient_descent_on_family(1000, 2, subfamily_assigments, timeout=subfamily_size*10)
    print(subfamily_gd_best_value)

    our_evaluations = gd.get_values_on_subfamily(gd.get_dtmc_sketch(best_gd_fsc), subfamily_assigments)


    best_gd_fsc, subfamily_gd_best_value = gd.run_gradient_descent_on_family(1000, 2, subfamily_assigments, timeout=subfamily_size*10, random_selection=True)
    print(subfamily_gd_best_value)

    random_evaluations = gd.get_values_on_subfamily(gd.get_dtmc_sketch(best_gd_fsc), subfamily_assigments)
    print("RANDOM:", random_evaluations)
    print("OURS:", our_evaluations)

# run_family()
run_family_softmax()
# run_subfamily()
