"""On the fly registeration of env using strings."""
import gym
from gym.envs.registration import register

def _register(env_name):
  class_name, cfg_str, ver = env_name.split('-')
  if env_name not in [env_spec.id for env_spec in gym.envs.registry.all()]:
    register(
      id=env_name,
      entry_point='rl.envs:{:s}'.format(class_name),
      nondeterministic=False,
      kwargs={'cfg_str': cfg_str},
    )
