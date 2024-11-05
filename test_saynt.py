from pomdp_families import POMDPFamiliesSynthesis

# Can try various models.
# project_path="models/pomdp/sketches/obstacles-10-2"
project_path="models/pomdp/sketches/avoid"
# project_path="models/pomdp/sketches/dpm"

gd = POMDPFamiliesSynthesis(project_path)
# Errors may very depending on the timeout. Timeout needs to be large enough for Paynt to find a controller.
gd.experiment_on_subfamily(gd.pomdp_sketch, 2, saynt=True, timeout=5, evaluate_on_whole_family=True)