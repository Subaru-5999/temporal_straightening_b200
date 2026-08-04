import warnings

from gym.envs.registration import register

# PointMaze pulls in the whole MuJoCo stack at import time: env/pointmaze/
# __init__.py -> maze_model.py -> `from gym.envs.mujoco import mujoco_env` and
# `from d4rl import offline_env`. Those are the build-risky planning deps
# (mujoco-py compiles from source, d4rl installs from git), and PushT / Wall /
# deformable need none of them -- yet importing this package used to require
# them for every env. Import lazily so a PushT-only evaluation works without
# the MuJoCo stack. When it IS installed, behaviour is byte-identical to before.
try:
    from .pointmaze import U_MAZE, MEDIUM_MAZE

    _POINTMAZE_AVAILABLE = True
    _POINTMAZE_IMPORT_ERROR = None
except Exception as _e:  # ImportError, or a mujoco_py build/runtime failure
    _POINTMAZE_AVAILABLE = False
    _POINTMAZE_IMPORT_ERROR = _e
    warnings.warn(
        f"PointMaze environments unavailable ({type(_e).__name__}: {_e}). "
        "Expected and harmless when evaluating PushT / Wall / deformable. "
        "Run setup_planning.sh (MuJoCo 210 + mujoco-py + d4rl) to enable them."
    )

register(
    id="pusht",
    entry_point="env.pusht.pusht_wrapper:PushTWrapper",
    max_episode_steps=300,
    reward_threshold=1.0,
)
if _POINTMAZE_AVAILABLE:
    register(
        id='point_maze',
        entry_point='env.pointmaze:PointMazeWrapper',
        max_episode_steps=300,
        kwargs={
            'maze_spec':U_MAZE,
            'reward_type':'sparse',
            'reset_target': False,
            'ref_min_score': 23.85,
            'ref_max_score': 161.86,
            'dataset_url':'http://rail.eecs.berkeley.edu/datasets/offline_rl/maze2d/maze2d-umaze-sparse-v1.hdf5'
        }
    )

    register(
        id="point_maze_medium",
        entry_point="env.pointmaze:PointMazeWrapper",
        max_episode_steps=600,
        kwargs={
            "maze_spec": MEDIUM_MAZE,
            "reward_type": "sparse",
            "reset_target": False,
            "ref_min_score": 13.13,
            "ref_max_score": 277.39,
            "dataset_url": "http://rail.eecs.berkeley.edu/datasets/offline_rl/maze2d/maze2d-medium-sparse-v1.hdf5",
        },
    )

register(
    id="wall",
    entry_point="env.wall.wall_env_wrapper:WallEnvWrapper",
    max_episode_steps=300,
    reward_threshold=1.0,
)

register(
    id="deformable_env",
    entry_point="env.deformable_env.FlexEnvWrapper:FlexEnvWrapper",
    max_episode_steps=300,
    reward_threshold=1.0,
)