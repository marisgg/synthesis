MAX_THREADS = 10

BASE_OUTPUT_DIR = "./output-parallel-subfamily"

BASE_SKETCH_DIR = 'models/pomdp/sketches'

USE_SAYNT_HOTFIX = True

# WIP models, might get deleted:
ACO = f"{BASE_SKETCH_DIR}/aco"
INTERCEPT = f"{BASE_SKETCH_DIR}/intercept"

# Can try various models.
OBSTACLES_EIGHTH_THREE = f"{BASE_SKETCH_DIR}/obstacles-8-3"
OBSTACLES_TEN_TWO = f"{BASE_SKETCH_DIR}/obstacles-10-2"
AVOID = f"{BASE_SKETCH_DIR}/avoid"
DPM = f"{BASE_SKETCH_DIR}/dpm"
ROVER = f"{BASE_SKETCH_DIR}/rover"
NETWORK = f"{BASE_SKETCH_DIR}/network"

ENVS = [OBSTACLES_TEN_TWO, OBSTACLES_EIGHTH_THREE, DPM, AVOID, ROVER, NETWORK]