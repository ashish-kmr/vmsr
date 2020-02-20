import logging
import gym
from gym.envs.registration import register
logger = logging.getLogger(__name__)

if 'MetaEnv-v0' not in [env_spec.id for env_spec in gym.envs.registry.all()]:
  register(
      id='MetaEnv-v0',
      entry_point='rl.envs:MetaEnv',
      nondeterministic=False,
  )

# if 'GoToPos-v0' not in [env_spec.id for env_spec in gym.envs.registry.all()]:
#   register(
#       id='GoToPos-v0',
#       entry_point='rl.envs:GoToPos',
#       nondeterministic=False,
#   )

if 'Dummy-v0' not in [env_spec.id for env_spec in gym.envs.registry.all()]:
  register(
      id='Dummy-v0',
      entry_point='rl.envs:Dummy',
      timestep_limit=1000,
      reward_threshold=1.0,
      nondeterministic=False,
  )
