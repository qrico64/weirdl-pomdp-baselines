import os

from gymnasium.envs.mujoco import mujoco_env

from .core.serializable import Serializable

ENV_ASSET_DIR = os.path.join(os.path.dirname(__file__), "assets")


class MujocoEnv(mujoco_env.MujocoEnv, Serializable):
    """
    My own wrapper around MujocoEnv.

    The caller needs to declare
    """

    def __init__(
        self,
        model_path,
        frame_skip=1,
        model_path_is_local=True,
        automatically_set_obs_and_action_space=False,  # Kept for backwards compatibility
    ):
        # Initialize Serializable to support pickling for multiprocessing
        Serializable.__init__(self)

        if model_path_is_local:
            model_path = get_asset_xml(model_path)

        # Always call parent __init__ to properly initialize the environment
        mujoco_env.MujocoEnv.__init__(
            self,
            model_path,
            frame_skip,
            observation_space=None,
            default_camera_config=None,
        )

    def init_serialization(self, locals):
        Serializable.quick_init(self, locals)

    def log_diagnostics(self, paths):
        pass


def get_asset_xml(xml_name):
    return os.path.join(ENV_ASSET_DIR, xml_name)
