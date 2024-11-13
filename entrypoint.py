from pomdp_families import POMDPFamiliesSynthesis
from pomdp_families import Method

# Can try various models.
# project_path="models/pomdp/sketches/obstacles-8-3"
project_path="models/pomdp/sketches/obstacles-10-2"
# project_path="models/pomdp/sketches/avoid"
# project_path="models/pomdp/sketches/dpm"

gd = POMDPFamiliesSynthesis(project_path, use_softmax=True, steps=1, learning_rate=0.01)

subfamily_assigments = gd.create_random_subfamily(5)
# print(type(subfamily_assigments[0]))


# gd.run_gradient_descent_on_family(1000, 3, timeout=60)

# gd.run_gradient_descent_on_whole_family(1000, 2, random=True)
# exit()



subfamily_other_results = gd.experiment_on_subfamily([subfamily_assigments[1]], 3, Method.GRADIENT, timeout=None, evaluate_on_whole_family=True)

# import pickle
# print(subfamily_other_results)
# with open("./output.pickle", 'wb') as handle:
    # pickle.dump(subfamily_other_results, handle)

# best_gd_fsc, subfamily_gd_best_value = gd.run_gradient_descent_on_family(100000, 2, subfamily_assigments, timeout=None)
# print(subfamily_gd_best_value)
