from _logging import logging
import copy
import rl
from rl._register import _register
from src import utils as _utils
import glob
import os
import time
from pprint import pformat
from collections import deque
from src.utils import get_time_str
import platform

import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import algo
from arguments import get_args
from envs import make_vec_envs, VecPyTorchMP3D
from model import SubPolicy, Policy
from storage import RolloutStorage
from utils import get_vec_normalize
from visualize import visdom_plot

args = get_args()

assert args.algo in ['a2c', 'ppo', 'acktr']
if args.recurrent_policy:
    assert args.algo in ['a2c', 'ppo'], \
        'Recurrent policy is not implemented for ACKTR'

num_updates = int(args.num_frames) // args.num_steps // args.num_processes

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

# try:
#     os.makedirs(args.log_dir)
# except OSError:
#     files = glob.glob(os.path.join(args.log_dir, '*.monitor.csv'))
#     for f in files:
#         os.remove(f)
# 
# eval_log_dir = args.log_dir + "_eval"
# 
# try:
#     os.makedirs(eval_log_dir)
# except OSError:
#     files = glob.glob(os.path.join(eval_log_dir, '*.monitor.csv'))
#     for f in files:
#         os.remove(f)

def setup_file_logging(logdir):
  file_name = os.path.join(logdir, 'log_{:s}.log'.format(get_time_str()))
  file_handler = logging.FileHandler(filename=file_name)
  file_handler.setFormatter(logging.getLogger().handlers[0].formatter)
  logging.getLogger().addHandler(file_handler)
  logging.error('Logging to file_name: %s', file_name)

def main():
    torch.set_num_threads(1)
    device = torch.device("cuda:0" if args.cuda else "cpu")

    if args.vis:
        from visdom import Visdom
        viz = Visdom(port=args.port)
        win = None
    # envs = make_vec_envs(args.env_name, args.seed, args.num_processes,
    #                     args.gamma, args.log_dir, args.add_timestep, device, False)
    affinitstr = ''
    if args.init_aff not in ['ours']: affinitstr = '_' + args.init_aff
    log_dir = os.path.join(args.env_name, args.init_policy + affinitstr + '_' + str(args.seed), args.algo)
    args.log_dir = args.log_dir_prefix + log_dir + args.log_dir_suffix
    _utils.mkdir_if_missing(args.log_dir)
    setup_file_logging(args.log_dir)
    logging.error('Running on %s', platform.node())
    logging.error(pformat(vars(args), indent=1))
    
    model_dir = os.path.join(args.log_dir, 'models')
    _utils.mkdir_if_missing(model_dir)
    _register(args.env_name)
    
    actual_envs = gym.make(args.env_name)
    batch_size = actual_envs.task_params.batch_size
    actual_envs.setup_logger(args.log_dir)
    actual_envs.seed(args.seed)
    envs = actual_envs
    arch = 'latent_rn5'; pretrained = False; load_path = None; 
    train_meta = False; teacher_forcing = False; meta_steps = None;
    num_ops = None
    base_ours = args.base_model_folder#'output/mp3d/operators_invmodel_lstm_models_sn5/'

    if args.init_policy == 'ours_inv':
        run_no = 77000
        model_folder_name = '4,10,16_10,15,20_30_8_30,-10_60,-30_F_1500/' 
        base_load_path = os.path.join(base_ours, model_folder_name)
        load_path = base_load_path + '{0:03d}'.format(run_no)
        train_meta = True
        teacher_forcing = False
        num_ops = 4

    elif args.init_policy == 'ours_hr_45K_ms':
        run_no = 68000
        model_folder_name = '10_6,12,18_77000__32_4_-1_9,12,15,18_30,-20_80,-40_RN5N__40.4,10,16_10,15,20_30_8_30,-10_60,-30_F_1500/' 
        base_load_path = os.path.join(base_ours, model_folder_name)
        load_path = base_load_path + '{0:03d}'.format(run_no)
        train_meta = True
        teacher_forcing = True
        num_ops = 4
    #elif args.init_policy == 'ours_hr_notf':
    #    run_no = 38000
    #    base_load_path = 'output/mp3d/operators_invmodel_lstm_models_sn5/_6,12,18_100000__32__-1_9,12,15,18_30,-20_80,-40_RN5N/'
    #    load_path = base_load_path + '{0:03d}'.format(run_no)
    #    train_meta = True
    #    teacher_forcing = False

    elif args.init_policy == 'imnet_hr':
        pretrained = True
        train_meta = True
        num_ops = 4

    elif args.init_policy == 'scratch_hr':
        train_meta = True
        num_ops = 4

    elif args.init_policy == 'scratch':
        arch = 'resnet18'
        num_ops = None

    elif args.init_policy == 'scratch_imnet':
        arch = 'resnet18'
        num_ops = None
        pretrained = True

    if train_meta:
      _utils.mkdir_if_missing(os.path.join(args.log_dir, 'meta'))
      feature_dim = 512; 
      if args.init_policy == 'diayn': meta_hidden_sz = 512
      else: meta_hidden_sz = 256
      sub_p = SubPolicy(num_ops,
              base_kwargs={
          'recurrent': True, #args.recurrent_policy,
          'hidden_size': meta_hidden_sz, #args.hidden_size,
          'policy_type': args.policy_type,
          'full_load_path': load_path,
          'pretrained': pretrained,
          'init_policy': args.init_policy})

      meta_envs = gym.make('MetaEnv-v0')
      meta_envs.setup_meta_env(actual_envs, sub_p, args.meta_steps, args.meta_actions, device)
      meta_envs.setup_logger(os.path.join(args.log_dir, 'meta'))
      meta_envs.seed(args.seed)
      envs = meta_envs
    
    envs = VecPyTorchMP3D(envs, device)


    actor_critic = Policy(envs.observation_space, envs.action_space, teacher_forcing,
        base_kwargs={
          'recurrent': args.recurrent_policy, 
          'hidden_size': args.hidden_size, 
          'policy_type': args.policy_type,
          'full_load_path': load_path,
          'pretrained': pretrained,
          'arch':arch,
          'num_ops': num_ops,
          'init_policy':args.init_policy,
          'init_aff': args.init_aff})

    actor_critic.to(device)
    actor_critic.base.image_cnn.to(device)
    actor_critic.device = device

    if train_meta:
        sub_p.to(device)
        sub_p.base.image_cnn.to(device)
        if args.recurrent_policy:
            sub_p.base.gru.to(device)

    if train_meta:
        if args.algo == 'a2c':
            subagent = algo.A2C_ACKTR(sub_p, args.value_loss_coef,
                                   args.entropy_coef, lr=args.lr*0.01,
                                   eps=args.eps, alpha=args.alpha,
                                   max_grad_norm=args.max_grad_norm)
        elif args.algo == 'ppo':
            subagent = algo.PPO(sub_p, args.clip_param, args.ppo_epoch, args.num_mini_batch,
                             args.value_loss_coef, args.entropy_coef, lr=args.lr,
                                   eps=args.eps,
                                   max_grad_norm=args.max_grad_norm)
        elif args.algo == 'acktr':
            subagent = algo.A2C_ACKTR(sub_p, args.value_loss_coef,
                                   args.entropy_coef, acktr=True)

    if args.algo == 'a2c':
        agent = algo.A2C_ACKTR(actor_critic, args.value_loss_coef,
                               args.entropy_coef, lr=args.lr,
                               eps=args.eps, alpha=args.alpha,
                               max_grad_norm=args.max_grad_norm)
    elif args.algo == 'ppo':
        agent = algo.PPO(actor_critic, args.clip_param, args.ppo_epoch, args.num_mini_batch,
                         args.value_loss_coef, args.entropy_coef, lr=args.lr,
                               eps=args.eps,
                               max_grad_norm=args.max_grad_norm)
    elif args.algo == 'acktr':
        agent = algo.A2C_ACKTR(actor_critic, args.value_loss_coef,
                               args.entropy_coef, acktr=True)

    rollouts = RolloutStorage(args.num_steps, batch_size*args.num_processes,
                        envs.observation_space, envs.action_space,
                        actor_critic.recurrent_hidden_state_size)

    if train_meta:
        rollouts_subp = RolloutStorage(args.meta_steps, batch_size*args.num_processes,
                            envs.venv.obs_space_subp, envs.action_space,
                            sub_p.recurrent_hidden_state_size)
    
    episode_rewards = deque(maxlen=10)

    obs = envs.reset()

    for k in rollouts.obs_vec:
      rollouts.obs_vec[k][0].copy_(obs[k])
   
    if train_meta:
        obs_subp = {'view': obs['view'], \
                'latent': torch.zeros([batch_size, num_ops]), \
                'prev_act': torch.zeros([batch_size,4])}

        for k in rollouts_subp.obs_vec:
          rollouts_subp.obs_vec[k][0].copy_(obs_subp[k])
    
    rollouts.to(device)
    if train_meta:
        rollouts_subp.to(device)
        envs.venv.set_storage(rollouts_subp)
        envs.venv.set_rlagent(subagent)
        envs.venv.use_gae = args.use_gae
        envs.venv.gamma = args.gamma
        envs.venv.tau = args.tau
    start = time.time()
    alpha_update_rate = 1000 
    for j in range(num_updates):
        for step in range(args.num_steps):
            # Sample actions
            with torch.no_grad():
                value, action, action_log_prob, recurrent_hidden_states = actor_critic.act(
                        {k: rollouts.obs_vec[k][step] for k in rollouts.obs_vec},
                        rollouts.recurrent_hidden_states[step],
                        rollouts.masks[step])

            # Obser reward and next obs
            #print(value.shape, action.shape, action_log_prob.shape, recurrent_hidden_states.shape)
            obs, reward, done, infos = envs.step(action)

            
            for info in infos:
                if 'episode' in info.keys():
                    episode_rewards.append(info['episode']['r'])

            # If done then clean the history of observations.
            masks = torch.FloatTensor([[0.0] if done_ else [1.0]
                                       for done_ in done])
            rollouts.insert(obs, recurrent_hidden_states, action, action_log_prob, value, reward, masks)

        with torch.no_grad():
            next_value = actor_critic.get_value({k: rollouts.obs_vec[k][-1] for k in rollouts.obs_vec},
                                                rollouts.recurrent_hidden_states[-1],
                                                rollouts.masks[-1]).detach()

        rollouts.compute_returns(next_value, args.use_gae, args.gamma, args.tau)

        value_loss, action_loss, dist_entropy = agent.update(rollouts)

        rollouts.after_update()
        actor_critic.update_if_tf()
        if j % args.save_interval == 0:
            # A really ugly way to save a model to CPU
            save_model = actor_critic
            if args.cuda:
                save_model = copy.deepcopy(actor_critic).cpu()

            save_model = [save_model, getattr(get_vec_normalize(envs), 'ob_rms', None)]
            torch.save(save_model, os.path.join(model_dir, 'modelA.pt'.format(j)))
            torch.save(save_model, os.path.join(model_dir, 'modelB.pt'.format(j)))

        total_num_steps = (j + 1) * batch_size * args.num_processes * args.num_steps
        
        if j % args.log_interval == 0 and len(episode_rewards) > 1:
            end = time.time()
            str_ = "Updates %d, num timesteps %d, FPS %d."
            logging.error(str_, j, total_num_steps, int(total_num_steps / (end - start)))
            str_ = " Last %d training episodes: mean/median reward %.1f/%.1f, min/max reward %.1f/%.1f"
            logging.error(str_, len(episode_rewards), np.mean(episode_rewards),
              np.median(episode_rewards), np.min(episode_rewards),
              np.max(episode_rewards))
            str_ = "   dist_entropy: %0.4f, value_loss %0.4f, action_loss: %0.4f"
            logging.error(str_, dist_entropy, value_loss, action_loss)
            logging.error("")

        if (args.eval_interval is not None
                and len(episode_rewards) > 1
                and j % args.eval_interval == 0):
            eval_envs = make_vec_envs(
                args.env_name, args.seed + args.num_processes, args.num_processes,
                args.gamma, eval_log_dir, args.add_timestep, device, True)

            vec_norm = get_vec_normalize(eval_envs)
            if vec_norm is not None:
                vec_norm.eval()
                vec_norm.ob_rms = get_vec_normalize(envs).ob_rms

            eval_episode_rewards = []

            obs = eval_envs.reset()
            eval_recurrent_hidden_states = torch.zeros(args.num_processes,
                            actor_critic.recurrent_hidden_state_size, device=device)
            eval_masks = torch.zeros(args.num_processes, 1, device=device)

            while len(eval_episode_rewards) < 10:
                with torch.no_grad():
                    _, action, _, eval_recurrent_hidden_states = actor_critic.act(
                        obs, eval_recurrent_hidden_states, eval_masks, deterministic=True)

                # Obser reward and next obs
                obs, reward, done, infos = eval_envs.step(action)

                eval_masks = torch.FloatTensor([[0.0] if done_ else [1.0]
                                                for done_ in done])
                for info in infos:
                    if 'episode' in info.keys():
                        eval_episode_rewards.append(info['episode']['r'])

            eval_envs.close()

            logging.error(" Evaluation using %d episodes: mean reward %.5f\n",
                len(eval_episode_rewards), np.mean(eval_episode_rewards))

        if args.vis and j % args.vis_interval == 0:
            try:
                # Sometimes monitor doesn't properly flush the outputs
                win = visdom_plot(viz, win, args.log_dir, args.env_name,
                  args.algo, args.num_frames)
            except IOError:
                pass

if __name__ == "__main__":
    main()
