# project_path = '\"models/pomdp/sketches/avoid\"'
# idx = 0
# saynt_str = f"""
# from pomdp_families import POMDPFamiliesSynthesis
# import pickle

# gd = POMDPFamiliesSynthesis(project_path={project_path})
# hole_assignment = gd.stratified_subfamily_sampling(5, 11)[{idx}]
# pomdp = gd.pomdp_sketch.build_pomdp(hole_assignment).model
# fsc = gd.solve_pomdp_paynt(pomdp=pomdp, specification=gd.pomdp_sketch.specification.copy(), timeout=5, k=5)
# with open('temp.pickle', 'wb') as handle:
#     pickle.dump(fsc, handle)
# """
# exec(saynt_str)
import random
import sys
from pomdp_families import POMDPFamiliesSynthesis
import pickle
import argparse
from config import *

parser = argparse.ArgumentParser()
parser.add_argument("hole_combination", type=str)
parser.add_argument("timeout", type=int)
parser.add_argument("project_path", type=str, choices=ENVS)
parser.add_argument("filename", type=str)
args = parser.parse_args()

gd = POMDPFamiliesSynthesis(project_path=args.project_path)
evaluated = eval(args.hole_combination)
all_hole_combinations = list(gd.pomdp_sketch.family.all_combinations())
assert evaluated in set(all_hole_combinations)
hole_assignment = gd.pomdp_sketch.family.construct_assignment(evaluated)
pomdp = gd.pomdp_sketch.build_pomdp(hole_assignment).model
fsc = gd.solve_pomdp_saynt(pomdp=pomdp, specification=gd.pomdp_sketch.specification.copy(), timeout=args.timeout, k=5)
with open(args.filename, 'wb') as handle:
    pickle.dump(fsc, handle)
