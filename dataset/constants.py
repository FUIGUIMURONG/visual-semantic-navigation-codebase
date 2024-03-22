GOAL_SUCCESS_REWARD = 5
STEP_PENALTY = -0.01
ACTION_FAIL_PENALTY = -0.1
STOP_NOT_SUCCESS = -0.5

TRAIN_SCENE = [f"FloorPlan{ids}" for ids in range(221, 228)]
VAL_SCENE = ["FloorPlan228"]
TEST_SCENE = ["FloorPlan229"]

ACTION_LIST = ["MoveAhead", "MoveRight", "MoveLeft", "RotateRight", "RotateLeft"]
ALL_ACTION_LIST = ["MoveAhead", "MoveRight", "MoveLeft", "RotateRight", "RotateLeft", "Stop"]

ALL_ACTION_TO_ID = {act: id_ for (id_, act) in enumerate(ALL_ACTION_LIST)}
