import os, subprocess, time, signal, numpy as np, json, csv, pickle
import torch.nn as nn
from six.moves import cPickle
import gym
from gym import error, spaces
from gym import utils
from gym.utils import seeding
from .rl_mp_env import MPEnv, get_task_params, get_task_params_from_string, EnvMultiplexer
from .rl_factory import get_dataset
from .rs18_curiosity import resnet18, ForwardModel
from render.swiftshader_renderer_gpu import get_r_obj
from src import utils as _utils
from tensorboardX import SummaryWriter
import torch
from torchvision import transforms
import torchvision.models as models
import torch.optim as optim

data_transforms = {
    'train': transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),}

def load_episode_data(file_name):
  epinfos = []
  done = False
  with open(file_name, 'rb') as f:
    while not done:
      try:
        epinfo = cPickle.load(f)
        epinfos.append(epinfo)
      except:
        done = True
  print('Loaded episodes: ', len(epinfos))
  return epinfos

class CustomEnv(gym.Env):
  def __init__(self):
    self.setup_logger(None)

  def setup_logger(self, dir_name, vis_interval=100, log_interval=32):
    self.logger = _utils.Foo(logger=None, f=None, tstart=time.time(),
      tf_writer=None, dir_name=None, vis_interval=vis_interval, log_interval=log_interval)
    
    if dir_name is not None:
      if hasattr(self, 'task_params'):
        filename = os.path.join(dir_name, 'task_params.json')
        with open(filename, 'w') as fp:
          json.dump(vars(self.task_params), fp)
      
      filename = os.path.join(dir_name, 'dd.monitor.pkl')
      f = open(filename, "wb")
      # str_ = json.dumps({"t_start": self.logger.tstart, 'env_id' : 'a'})
      # f.write('#{:s}\n'.format(str_))
      # logger = csv.DictWriter(f, fieldnames=('r', 'l', 't'))
      # logger.writeheader(); f.flush();
      # self.logger.logger = logger;
      
      self.logger.tf_writer = SummaryWriter(log_dir=dir_name)
      self.logger.f = f; 
      self.logger.dir_name = dir_name
      _utils.mkdir_if_missing(os.path.join(self.logger.dir_name, 'env_vis'))
      self.logger.fp_view_done = False
      print('Logging things to: {:s}'.format(dir_name))

class MetaEnv(CustomEnv):
  metadata = {'render.modes': []}

  def __init__(self):
    self.total_frames = 0
    self.episodes = 0
    None

  def setup_meta_env(self, env, subpolicy, meta_steps, num_meta_actions, device):
    input_res = 224
    self.device = device
    self.env = env
    self.subpolicy = subpolicy
    self.meta_steps = meta_steps
    self.action_space = spaces.Discrete(num_meta_actions)
    self.observation_space = self.env.observation_space
    self.num_ops = subpolicy.num_ops
    self.obs_space_subp = spaces.Dict({
        'view': spaces.Box(low=0, high=255, shape=(3, input_res, input_res), dtype=np.float32),
        'latent': spaces.Box(low=-np.inf, high=np.inf, shape=(subpolicy.num_ops,), dtype=np.float32),
        'prev_act': spaces.Box(low=-np.inf, high=np.inf, shape=(4,), dtype=np.float32)
        })
    

  def reset(self):
    self.last_obs = self.env.reset()
    self.episode_rewards = [[] for _ in self.env.episode_rewards]
    return self.last_obs 
    
  def set_storage(self, rollouts):
      self.rollouts = rollouts

  def set_rlagent(self, agent):
      self.agent = agent

  def step(self, meta_action):
    # This is a macro action.
    rollouts = self.rollouts
    meta_step_reward = []
    num_subp = self.num_ops
    # These are features from the previous call to step()
    last_obs = self.last_obs 
    batch_sz = len(meta_action)
    latent_oh = torch.zeros([len(meta_action),num_subp], dtype= torch.float).to(self.device)
    latent_oh[range(len(meta_action)),meta_action] = 1.0
    hidden_state = torch.zeros([1,batch_sz,256], dtype = torch.float).to(self.device)
    prev_act = torch.zeros([batch_sz,4], dtype = torch.float).to(self.device)
    rollouts.obs_vec['latent'][0].copy_(latent_oh)
    for step in range(self.meta_steps):
      with torch.no_grad():
          value, sample_act, action_prob, hidden_state = self.subpolicy.act(
                  {k: rollouts.obs_vec[k][step] for k in rollouts.obs_vec},
                  rollouts.recurrent_hidden_states[step],
                  rollouts.masks[step])
      
      action = sample_act.detach().cpu().numpy()[:,0].tolist()
      action = [int(itr) for itr in action]

      # This step function will only take a step if the environment is not done
      # yet.  If the environmnet is done, then it won't take any step, and
      # return a done of True and a reward of NaN.

      obs, rewards, dones, infos = self.env.step(action, \
              only_if_not_done=True, reset_done_episodes=False)

      rewards[np.isnan(rewards)] = 0
      obs_subp = {'view': obs['view'],\
              'latent': latent_oh,\
              'prev_act': prev_act}
      masks = torch.FloatTensor([[0.0] if done_ else [1.0] for done_ in dones])

      self.rollouts.insert(obs_subp, hidden_state, sample_act, action_prob, \
              value, torch.FloatTensor(rewards).unsqueeze(1), masks)

      self.last_obs = obs
      meta_step_reward.append(rewards)
      
      prev_act = prev_act * 0.0
      prev_act[range(batch_sz),sample_act[:,0]] = 1

    with torch.no_grad():
        next_value = self.subpolicy.get_value({k: rollouts.obs_vec[k][-1] for k in rollouts.obs_vec},
                rollouts.recurrent_hidden_states[-1],
                rollouts.masks[-1]).detach()
    rollouts.compute_returns(next_value, self.use_gae, self.gamma, self.tau)
    value_loss, action_loss, dist_entropy = self.agent.update(rollouts)
    rollouts.after_update()


    # At this point, the last done is what we need, to restart episodes, and
    # return to the meta controller.
    meta_step_reward = np.array(meta_step_reward)
    meta_step_reward[np.isnan(meta_step_reward)] = 0.
    meta_step_reward = np.sum(meta_step_reward, 0)
    for i in range(meta_step_reward.shape[0]):
      self.episode_rewards[i].append(meta_step_reward[i])
      self.total_frames += self.meta_steps
    
    _infos = self.env.reset_done_episodes(dones)
    infos = self.reset_done_episodes(dones)
    return last_obs, meta_step_reward, dones, infos
  
  def reset_done_episodes(self, dones):
    """ Does not do anything to the underlyinf env, Only cleans up its own
    state."""
    infos = []
    for i in range(len(dones)):
      info = {}
      if dones[i]:
        self.episodes += 1
        info = self.get_episode_info_i(i)
        info = {'episode': info}
        
        # reset that environment
        self.episode_rewards[i] = []
      infos.append(info)
    return infos
  
  def get_episode_info_i(self, i):
    eprew = float(np.sum(self.episode_rewards[i]))
    eplen = len(self.episode_rewards[i])
    epinfo = {
      "r": eprew, "l": eplen, 
      "t": round(time.time() - self.logger.tstart, 6), 
      'n_samples': self.total_frames, 
      'n_episodes': self.episodes,
    }
    if self.logger.f:
        cPickle.dump(epinfo, self.logger.f, cPickle.HIGHEST_PROTOCOL)
        self.logger.f.flush()
    
    if self.logger.tf_writer and np.mod(self.episodes, self.logger.log_interval) == 0:
      n_iter = self.total_frames
      self.logger.tf_writer.add_scalar('episode_reward', eprew, n_iter)
      self.logger.tf_writer.add_scalar('episode_len', eplen, n_iter)
    return epinfo

class Curiosity(CustomEnv):
  metadata = {'render.modes': []}

  def __init__(self, cfg_str):
    self._setup_rs18()
    self.task_params = get_task_params_from_string(cfg_str)
    dataset = get_dataset('mp3d', self.task_params.imset)
    imlist = dataset.get_imset()
    print(self.task_params)
    
    render_res = 256; input_res = 224;
    camera_param = _utils.Foo(width=render_res, height=render_res, z_near=0.01, z_far=20.0,
      fov_horizontal=60., fov_vertical=60., modalities=['rgb'], img_channels=3,
      im_resize=float(input_res)/float(render_res))
    self.r_obj = get_r_obj(camera_param)
    self.env = EnvMultiplexer(imlist, dataset, self.task_params, r_obj=self.r_obj) 
    
    self.action_space = spaces.Discrete(4)
    self.observation_space = spaces.Dict({
        'view': spaces.Box(low=0, high=255, shape=(3, input_res, input_res), dtype=np.float32), 
    })
    self.setup_logger(None)
    self.total_frames = 0
    self.episodes = 0
    # TODO(sgupta): Load the resnet model
  
  def seed(self, seed=None):
    self.rng = np.random.RandomState(seed)
  
  def reset(self):
    self.states = self.env.reset(self.rng)
    feats = self.env.get_features(self.states)
    obs = self.process_features(feats)
    self.last_obs = obs
    self.episode_rewards = [[] for s in self.states]
    return obs
  
  def process_features(self, feats):
    view_list = []
    for view_i in feats['view']:
      view_list.append(data_transforms['train'](view_i))
    obs = {'view': torch.stack(view_list)}
    return obs
    
  def _setup_rs18(self):
    self.resnet18 = resnet18(pretrained=True)
    for child in self.resnet18.children():
      for param in child.parameters():
        param.requires_grad = False
    device = torch.device("cuda:0")
    self.fm = ForwardModel(4)
    self.fm.to(device)
    self.resnet18.to(device)
    self.fm_criterion = torch.nn.MSELoss(reduction='none')
    self.fm_optimizer = optim.SGD(self.fm.parameters(), lr=0.001, momentum=0.9)


  def _compute_intrinsic_reward(self, obs, last_obs, action):
    optimizer = self.fm_optimizer
    optimizer.zero_grad()
    
    device = torch.device("cuda:0")
    view = obs['view']; last_view = last_obs['view']
    x = self.resnet18(view.to(device)).detach()
    last_x = self.resnet18(last_view.to(device)).detach()
    x = x.view(x.shape[0], -1)
    last_x = last_x.view(x.shape[0], -1)
    action = torch.from_numpy(np.array(action)).to(device)
    x_pred = self.fm(last_x, action)

    element_loss = self.fm_criterion(x_pred, x).mean(1)
    total_loss = element_loss.mean()
    total_loss.backward()
    optimizer.step()
    self.logger.tf_writer.add_scalar('fwd_loss', total_loss.detach().cpu().numpy(), self.total_frames)
    reward = element_loss.detach().cpu().numpy() / self.task_params.max_time_steps
    return reward

  def step(self, action, only_if_not_done=False, reset_done_episodes=True):
    self.states, rewards, dones = self.env.take_action(self.states, action, only_if_not_done=True)
    feats = self.env.get_features(self.states)
    obs = self.process_features(feats)
    rewards = np.array(rewards)
    rewards = self._compute_intrinsic_reward(obs, self.last_obs, action)
    self.last_obs = obs

    for i in range(rewards.shape[0]):
      self.total_frames += 1
      self.episode_rewards[i].append(rewards[i])

    # If episodes got done, we should reset them, and let the agent learn again.
    assert(np.all(dones) ^ np.invert(np.any(dones)))
    infos = [{} for _ in range(len(dones))]
    if np.all(dones):
      for i in range(len(dones)):
        info = self.get_episode_info_i(i)
        infos[i]['episode'] = info
        self.episodes += 1
      obs = self.reset()
    return obs, rewards, dones, infos

  def render(self, mode):
    None

  def get_episode_info_i(self, i):
    eprew = float(np.sum(self.episode_rewards[i]))
    eplen = len(self.episode_rewards[i])
    n_iter = self.total_frames
    epinfo = {
      "r": eprew, "l": eplen, 
      "t": round(time.time() - self.logger.tstart, 6), 
      'n_samples': self.total_frames, 
      'n_episodes': self.episodes,
    }
    if self.logger.f:
        cPickle.dump(epinfo, self.logger.f, cPickle.HIGHEST_PROTOCOL)
        self.logger.f.flush()
    
    if np.mod(self.episodes, self.logger.vis_interval) == 0:
      I = self.env.vis_top_view_i(i, os.path.join(self.logger.dir_name, 'env_vis'),  
        prefix='{:010d}_'.format(self.total_frames), suffix='_{:02d}'.format(i))
      self.logger.tf_writer.add_image('top_view', np.transpose(I[:,:,::-1], [2,0,1]), n_iter)
      
      if not self.logger.fp_view_done:
        I = self.env.vis_fp_view_i(i, os.path.join(self.logger.dir_name, 'env_vis'),  
          prefix='{:010d}_'.format(self.total_frames), suffix='_{:02d}'.format(i))
        self.logger.tf_writer.add_image('fp_view', np.transpose(I[:,:,::-1], [2,0,1]), n_iter)
        self.logger.fp_view_done = True
    
    if self.logger.tf_writer and np.mod(self.episodes, self.logger.log_interval) == 0:
      self.logger.tf_writer.add_scalar('episode_reward', eprew, n_iter)
      self.logger.tf_writer.add_scalar('episode_len', eplen, n_iter)
    return epinfo

class SemanticTask(CustomEnv):
  metadata = {'render.modes': []}

  def __init__(self, cfg_str):
    self.task_params = get_task_params_from_string(cfg_str)
    dataset = get_dataset('mp3d', self.task_params.imset)
    print(self.task_params)
    
    imlist = dataset.get_imset()
    render_res = 256; input_res = 224;
    camera_param = _utils.Foo(width=render_res, height=render_res, z_near=0.01, z_far=20.0,
      fov_horizontal=60., fov_vertical=60., modalities=['rgb'], img_channels=3,
      im_resize=float(input_res)/float(render_res))
    self.r_obj = get_r_obj(camera_param)
    self.env = MPEnv(imlist[0], dataset, False, self.task_params, r_obj=self.r_obj, task_type = 'Semantic') 
    
    self.action_space = spaces.Discrete(4)
    
    self.observation_space = spaces.Dict({
        'view': spaces.Box(low=0, high=255, shape=(3, input_res, input_res), dtype=np.float32) 
        #'t': spaces.Box(low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32),
    })
    # Simple observation setup to start with.
    # self.observation_space = spaces.Box(low=0, high=255, shape=(3, input_res, input_res), dtype=np.float32)
    # log_dir = os.path.join('outputs/rl/v1/')
    self.setup_logger(None)
    self.total_frames = 0
    self.episodes = 0
  
 
  def seed(self, seed=None):
    self.rng = np.random.RandomState(seed)
  
  def reset(self):
    self.states = self.env.reset(self.rng)
    feats = self.env.get_features(self.states)
    obs = self.process_features(feats)
    self.episode_rewards = [[] for s in self.states]
    return obs
  
  def process_features(self, feats):
    view_list = []
    for view_i in feats['view']:
        view_list.append(data_transforms['train'](view_i))
    obs = {
      'view': torch.stack(view_list)
      #'t': feats['t'],
    }
    return obs
  
  def step(self, action, only_if_not_done=False, reset_done_episodes=True):
    self.states, rewards, dones = self.env.take_action(self.states, action, only_if_not_done=only_if_not_done)
    rewards = np.array(rewards)
    for i in range(rewards.shape[0]):
      self.total_frames += 1
      if only_if_not_done and np.isnan(rewards[i]):
        # nan reward is because the episode has ended already.
        self.episode_rewards[i].append(0)
      else:
        self.episode_rewards[i].append(rewards[i])
    
    if reset_done_episodes:
      infos = self.reset_done_episodes(dones)
    else:
      infos = [{} for i in range(rewards.shape[0])]
    feats = self.env.get_features(self.states)
    obs = self.process_features(feats)
    return obs, rewards, dones, infos

  def reset_done_episodes(self, dones):
    infos = []
    for i in range(len(dones)):
      info = {}
      if dones[i]:
        self.episodes += 1
        info = self.get_episode_info_i(i)
        info = {'episode': info}
        
        # reset that environment
        self.states[i] = self.env.reset_i(i, self.rng)
        self.episode_rewards[i] = []
      infos.append(info)
    return infos

  def render(self, mode):
    None

  def get_episode_info_i(self, i):
    eprew = float(np.sum(self.episode_rewards[i]))
    eplen = len(self.episode_rewards[i])
    n_iter = self.total_frames
    # TODO(sgupta): Add code to export other metrics for the finished episode
    # (things like success %age, final distance to goal, SPL, initial_distance
    # to goal, num_collisions).
    metrics = self.env.get_metrics_i(i)
    epinfo = {
      "r": eprew, "l": eplen, 
      "t": round(time.time() - self.logger.tstart, 6), 
      'n_samples': self.total_frames, 
      'n_episodes': self.episodes,
    }
    epinfo.update(metrics)
    if self.logger.f:
        cPickle.dump(epinfo, self.logger.f, cPickle.HIGHEST_PROTOCOL)
        self.logger.f.flush()
    
    if np.mod(self.episodes, self.logger.vis_interval) == 0:
      I = self.env.vis_top_view_i(i, os.path.join(self.logger.dir_name, 'env_vis'),  
        prefix='{:010d}_'.format(self.total_frames), suffix='_{:02d}'.format(i))
      self.logger.tf_writer.add_image('top_view', np.transpose(I[:,:,::-1], [2,0,1]), n_iter)
      
      if not self.logger.fp_view_done:
        I = self.env.vis_fp_view_i(i, os.path.join(self.logger.dir_name, 'env_vis'),  
          prefix='{:010d}_'.format(self.total_frames), suffix='_{:02d}'.format(i))
        self.logger.tf_writer.add_image('fp_view', np.transpose(I[:,:,::-1], [2,0,1]), n_iter)
        self.logger.fp_view_done = True
    
    if self.logger.tf_writer and np.mod(self.episodes, self.logger.log_interval) == 0:
      self.logger.tf_writer.add_scalar('episode_reward', eprew, n_iter)
      self.logger.tf_writer.add_scalar('episode_len', eplen, n_iter)
      self.logger.tf_writer.add_scalar('exp_nsteps', epinfo['exp_nsteps'], n_iter)
      self.logger.tf_writer.add_scalar('exp_nsteps_s', epinfo['exp_nsteps_s'], n_iter)
      self.logger.tf_writer.add_scalar('metrics/successful', epinfo['successful'], n_iter)
      self.logger.tf_writer.add_scalar('metrics/dist_init', epinfo['state_dists'][0], n_iter)
      self.logger.tf_writer.add_scalar('metrics/dist_end', epinfo['state_dists'][-1], n_iter)
      self.logger.tf_writer.add_scalar('metrics/spl', epinfo['spl'], n_iter)
      self.logger.tf_writer.add_scalar('metrics/collisions', epinfo['num_collisions'], n_iter)
      self.logger.tf_writer.add_scalar('max_metrics/max_task_reward', epinfo['max_task_reward'], n_iter)
      self.logger.tf_writer.add_scalar('max_metrics/avg_max_task_reward', epinfo['avg_max_task_reward'], n_iter)
      self.logger.tf_writer.add_scalar('max_metrics/avg_task_tried', epinfo['avg_task_tried'], n_iter)
      self.logger.tf_writer.add_scalar('max_metrics/avg_task_tried_reward', epinfo['avg_task_tried_reward'], n_iter)
    return epinfo

class GoToPos(CustomEnv):
  metadata = {'render.modes': []}

  def __init__(self, cfg_str):
    self.task_params = get_task_params_from_string(cfg_str)
    dataset = get_dataset('mp3d', self.task_params.imset)
    print(self.task_params)
    
    imlist = dataset.get_imset()
    render_res = 256; input_res = 224;
    camera_param = _utils.Foo(width=render_res, height=render_res, z_near=0.01, z_far=20.0,
      fov_horizontal=60., fov_vertical=60., modalities=['rgb'], img_channels=3,
      im_resize=float(input_res)/float(render_res))
    self.r_obj = get_r_obj(camera_param)
    self.env = MPEnv(imlist[0], dataset, False, self.task_params, r_obj=self.r_obj) 
    
    self.action_space = spaces.Discrete(4)
    
    self.observation_space = spaces.Dict({
        'view': spaces.Box(low=0, high=255, shape=(3, input_res, input_res), dtype=np.float32), 
        'goal_rt': spaces.Box(low=-np.inf, high=np.inf, shape=(2,), dtype=np.float32),
        'goal_xy': spaces.Box(low=-np.inf, high=np.inf, shape=(2,), dtype=np.float32),
        'goal_rcossin': spaces.Box(low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32),
        't': spaces.Box(low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32),
    })
    # Simple observation setup to start with.
    # self.observation_space = spaces.Box(low=0, high=255, shape=(3, input_res, input_res), dtype=np.float32)
    # log_dir = os.path.join('outputs/rl/v1/')
    self.setup_logger(None)
    self.total_frames = 0
    self.episodes = 0
  
 
  def seed(self, seed=None):
    self.rng = np.random.RandomState(seed)
  
  def reset(self):
    self.states = self.env.reset(self.rng)
    feats = self.env.get_features(self.states)
    obs = self.process_features(feats)
    self.episode_rewards = [[] for s in self.states]
    return obs
  
  def process_features(self, feats):
    view_list = []
    for view_i in feats['view']:
        view_list.append(data_transforms['train'](view_i))
    obs = {
      'view': torch.stack(view_list),#np.transpose(feats['view'], [0,3,1,2]), 
      'goal_rt': feats['goal_rt'], 
      'goal_xy': feats['goal_xy'], 
      'goal_rcossin': feats['goal_rcossin'],
      't': feats['t'],
    }
    return obs
  
  def step(self, action, only_if_not_done=False, reset_done_episodes=True):
    self.states, rewards, dones = self.env.take_action(self.states, action, only_if_not_done=only_if_not_done)
    rewards = np.array(rewards)
    for i in range(rewards.shape[0]):
      self.total_frames += 1
      if only_if_not_done and np.isnan(rewards[i]):
        # nan reward is because the episode has ended already.
        self.episode_rewards[i].append(0)
      else:
        self.episode_rewards[i].append(rewards[i])
    
    if reset_done_episodes:
      infos = self.reset_done_episodes(dones)
    else:
      infos = [{} for i in range(rewards.shape[0])]
    feats = self.env.get_features(self.states)
    obs = self.process_features(feats)
    return obs, rewards, dones, infos

  def reset_done_episodes(self, dones):
    infos = []
    for i in range(len(dones)):
      info = {}
      if dones[i]:
        self.episodes += 1
        info = self.get_episode_info_i(i)
        info = {'episode': info}
        
        # reset that environment
        self.states[i] = self.env.reset_i(i, self.rng)
        self.episode_rewards[i] = []
      infos.append(info)
    return infos

  def render(self, mode):
    None

  def get_episode_info_i(self, i):
    eprew = float(np.sum(self.episode_rewards[i]))
    eplen = len(self.episode_rewards[i])
    n_iter = self.total_frames
    # TODO(sgupta): Add code to export other metrics for the finished episode
    # (things like success %age, final distance to goal, SPL, initial_distance
    # to goal, num_collisions).
    metrics = self.env.get_metrics_i(i)
    epinfo = {
      "r": eprew, "l": eplen, 
      "t": round(time.time() - self.logger.tstart, 6), 
      'n_samples': self.total_frames, 
      'n_episodes': self.episodes,
    }
    epinfo.update(metrics)
    if self.logger.f:
        cPickle.dump(epinfo, self.logger.f, cPickle.HIGHEST_PROTOCOL)
        self.logger.f.flush()
    
    if np.mod(self.episodes, self.logger.vis_interval) == 0:
      I = self.env.vis_top_view_i(i, os.path.join(self.logger.dir_name, 'env_vis'),  
        prefix='{:010d}_'.format(self.total_frames), suffix='_{:02d}'.format(i))
      self.logger.tf_writer.add_image('top_view', np.transpose(I[:,:,::-1], [2,0,1]), n_iter)
      
      if not self.logger.fp_view_done:
        I = self.env.vis_fp_view_i(i, os.path.join(self.logger.dir_name, 'env_vis'),  
          prefix='{:010d}_'.format(self.total_frames), suffix='_{:02d}'.format(i))
        self.logger.tf_writer.add_image('fp_view', np.transpose(I[:,:,::-1], [2,0,1]), n_iter)
        self.logger.fp_view_done = True
    
    if self.logger.tf_writer and np.mod(self.episodes, self.logger.log_interval) == 0:
      self.logger.tf_writer.add_scalar('episode_reward', eprew, n_iter)
      self.logger.tf_writer.add_scalar('episode_len', eplen, n_iter)
      self.logger.tf_writer.add_scalar('exp_nsteps', epinfo['exp_nsteps'], n_iter)
      self.logger.tf_writer.add_scalar('exp_nsteps_s', epinfo['exp_nsteps_s'], n_iter)
      self.logger.tf_writer.add_scalar('metrics/successful', epinfo['successful'], n_iter)
      self.logger.tf_writer.add_scalar('metrics/dist_init', epinfo['state_dists'][0], n_iter)
      self.logger.tf_writer.add_scalar('metrics/dist_end', epinfo['state_dists'][-1], n_iter)
      self.logger.tf_writer.add_scalar('metrics/spl', epinfo['spl'], n_iter)
      self.logger.tf_writer.add_scalar('metrics/collisions', epinfo['num_collisions'], n_iter)
      self.logger.tf_writer.add_scalar('max_metrics/max_task_reward', epinfo['max_task_reward'], n_iter)
      self.logger.tf_writer.add_scalar('max_metrics/avg_max_task_reward', epinfo['avg_max_task_reward'], n_iter)
      self.logger.tf_writer.add_scalar('max_metrics/avg_task_tried', epinfo['avg_task_tried'], n_iter)
      self.logger.tf_writer.add_scalar('max_metrics/avg_task_tried_reward', epinfo['avg_task_tried_reward'], n_iter)
    return epinfo
 

class Dummy(gym.Env):
  metadata = {'render.modes': []}

  def __init__(self):
    self.action_space = spaces.Discrete(4)
    self.step_number = 0
  
  def seed(self, seed=None):
    None
  
  def reset(self):
    return np.zeros((20,20), dtype=np.float32)
    None

  def step(self, action):
    obs = np.zeros((20, 20), dtype=np.float32)
    self.step_number += 1
    reward = 1.
    done = self.step_number > 5
    info = {}
    return obs, reward, done, info

  def render(self, mode):
    None
