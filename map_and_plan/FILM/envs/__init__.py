import numpy as np
import torch

from .utils.vector_env import VectorEnv

import yaml
import yacs.config
import os
import json

import ipdb
st = ipdb.set_trace


def make_vec_envs(args, files):
    envs = construct_envs_alfred(args, files)
    if args.num_processes==1 and not args.run_film_only:
        envs = SingPyTorch(envs, args.device)
    else:
        envs = VecPyTorch(envs, args.device)
    return envs


# Adapted from https://github.com/ikostrikov/pytorch-a2c-ppo-acktr-gail/blob/master/a2c_ppo_acktr/envs.py#L159
class VecPyTorch():

    def __init__(self, venv, device):
        self.venv = venv
        self.num_envs = venv.num_envs
        self.observation_space = venv.observation_space
        self.action_space = venv.action_space
        self.device = device

    def setup_scene(self, traj_data, r_idx, args):
        obs, infos = self.venv.setup_scene(traj_data, r_idx, args)
        return obs, infos

    def to_thor_api_exec(self, action, object_id="", smooth_nav=False):
        obs, reward, done, info, events, actions = self.venv.to_thor_api_exec(action, object_id, smooth_nav)
        obs = torch.from_numpy(obs).float().to(self.device)
        reward = torch.from_numpy(reward).float()
        return obs, reward, done, info, events, actions
    
    def va_interact(self, action, interact_mask=None, smooth_nav=True, mask_px_sample=1, debug=False):
        obs, rew, done, infos, success, event, target_instance_id, emp , api_action = self.venv.va_interact(action, interact_mask, smooth_nav, mask_px_sample, debug)
        obs = torch.from_numpy(obs).float().to(self.device)
        rew = torch.from_numpy(rew).float()
        return obs, rew, done, infos, success[0], event[0], target_instance_id[0], emp[0], api_action[0] 
    
    
    def consecutive_interaction(self,  interaction, target_instance):
        obs, rew, done, info, success = self.venv.consecutive_interaction(interaction, target_instance)
        obs = torch.from_numpy(obs).float().to(self.device)
        rew = torch.from_numpy(rew).float()
        return obs, rew, done, info, success[0]
    
    def decompress_mask(self, mask):
        mask = self.venv.decompress_mask(mask)
        return mask

    def reset_goal(self, load, goal_name, cs):
        infos = self.venv.reset_goal(load, goal_name, cs)
        return infos

    def reset(self):
        obs, info = self.venv.reset()
        obs = torch.from_numpy(obs).float().to(self.device)
        return obs, info
    
    def evaluate(self, e):
        log_entry, success = self.venv.evaluate(e)
        return log_entry, success
    
    def load_initial_scene(self):
        obs, info, actions_dict = self.venv.load_initial_scene()
        obs = torch.from_numpy(obs).float().to(self.device)
        return obs, info, actions_dict
    
    def load_next_scene(self, load):
        obs, info, actions_dict = self.venv.load_next_scene(load)
        obs = torch.from_numpy(obs).float().to(self.device)
        return obs, info, actions_dict

    def step_async(self, actions):
        actions = actions.cpu().numpy()
        self.venv.step_async(actions)

    def step_wait(self):
        obs, reward, done, info = self.venv.step_wait()
        obs = torch.from_numpy(obs).float().to(self.device)
        reward = torch.from_numpy(reward).float()
        return obs, reward, done, info

    def step(self, actions):
        actions = actions.cpu().numpy()
        obs, reward, done, info = self.venv.step(actions)
        obs = torch.from_numpy(obs).float().to(self.device)
        reward = torch.from_numpy(reward).float()
        return obs, reward, done, info

    def get_rewards(self, inputs):
        reward = self.venv.get_rewards(inputs)
        reward = torch.from_numpy(reward).float()
        return reward

    def plan_act_and_preprocess(self, inputs, goal_spotted):
        obs, reward, done, info, gs, next_step_dict  = self.venv.plan_act_and_preprocess(inputs, goal_spotted)
        obs = torch.from_numpy(obs).float().to(self.device)
        reward = torch.from_numpy(reward).float()
        #return obs, reward, done, info, gs[0], next_step_dict[0]
        return obs, reward, done, info, gs, next_step_dict
    
    def get_instance_mask(self):
        return self.venv.get_instance_mask()
    
    def reset_total_cat(self, total_cat_dict, categories_in_inst):
        self.venv.reset_total_cat(total_cat_dict, categories_in_inst)

    def close(self):
        return self.venv.close()


# Adapted from https://github.com/ikostrikov/pytorch-a2c-ppo-acktr-gail/blob/master/a2c_ppo_acktr/envs.py#L159
class SingPyTorch():

    def __init__(self, venv, device):
        self.venv = venv
        self.num_envs = venv.num_envs
        self.observation_space = venv.observation_space
        self.action_space = venv.action_space
        self.device = device

    def setup_scene(self, traj_data, r_idx, args):
        st()
        obs, infos = self.venv.setup_scene(traj_data, r_idx, args)
        return [obs], [infos]

    def to_thor_api_exec(self, action, object_id="", smooth_nav=False):
        st()
        obs, reward, done, info, events, actions = self.venv.to_thor_api_exec(action, object_id, smooth_nav)
        obs = torch.from_numpy(obs).float().to(self.device).unsqueeze(0)
        reward = torch.from_numpy(reward).float().unsqueeze(0)
        return obs, reward, [done], [info], [events], [actions]
    
    def va_interact(self, action, interact_mask=None, smooth_nav=True, mask_px_sample=1, debug=False):
        st()
        obs, rew, done, infos, success, event, target_instance_id, emp , api_action = self.venv.va_interact(action, interact_mask, smooth_nav, mask_px_sample, debug)
        obs = torch.from_numpy(obs).float().to(self.device).unsqueeze(0)
        rew = torch.from_numpy(rew).float().unsqueeze(0)
        return obs, rew, [done], [infos], [success[0]], [event[0]], [target_instance_id[0]], [emp[0]], [api_action[0]]
    
    
    def consecutive_interaction(self,  interaction, target_instance):
        st()
        obs, rew, done, info, success = self.venv.consecutive_interaction(interaction, target_instance)
        obs = torch.from_numpy(obs).float().to(self.device).unsqueeze(0)
        rew = torch.from_numpy(rew).float().unsqueeze(0)
        return obs, rew, [done], [info], [success[0]]
    
    def decompress_mask(self, mask):
        st()
        mask = self.venv.decompress_mask(mask)
        return mask

    def reset_goal(self, load, goal_name, cs):
        infos = self.venv.reset_goal(load[0], goal_name, cs)
        return [infos]

    def reset(self):
        obs, info = self.venv.reset()
        obs = torch.from_numpy(obs).float().to(self.device).unsqueeze(0)
        return obs, [info]
    
    def evaluate(self):
        log_entry, success = self.venv.evaluate()
        return log_entry, success
    
    def load_initial_scene(self):
        obs, info, actions_dict = self.venv.load_initial_scene()
        obs = torch.from_numpy(obs).float().to(self.device).unsqueeze(0)
        return obs, [info], [actions_dict]
    
    def load_next_scene(self, load):
        obs, info, actions_dict = self.venv.load_next_scene(load[0])
        obs = torch.from_numpy(obs).float().to(self.device).unsqueeze(0)
        return obs, [info], [actions_dict]

    def step_async(self, actions):
        st()
        actions = actions.cpu().numpy()
        self.venv.step_async(actions)

    def step_wait(self):
        obs, reward, done, info = self.venv.step_wait()
        obs = torch.from_numpy(obs).float().to(self.device).unsqueeze(0)
        reward = torch.from_numpy(reward).float().unsqueeze(0)
        return obs, reward, [done], [info]

    def step(self, actions):
        st()
        actions = actions.cpu().numpy()
        obs, reward, done, info = self.venv.step(actions)
        obs = torch.from_numpy(obs).float().to(self.device).unsqueeze(0)
        reward = torch.from_numpy(reward).float().unsqueeze(0)
        return obs, reward, [done], [info]

    def get_rewards(self, inputs):
        st()
        reward = self.venv.get_rewards(inputs)
        reward = torch.from_numpy(reward).float().unsqueeze(0)
        return reward

    def plan_act_and_preprocess(self, inputs, goal_spotted):
        obs, reward, done, info, gs, next_step_dict  = self.venv.plan_act_and_preprocess(inputs[0], goal_spotted[0])
        obs = torch.from_numpy(obs).float().to(self.device).unsqueeze(0)
        reward = torch.tensor([reward]).float().unsqueeze(0)
        #return obs, reward, done, info, gs[0], next_step_dict[0]
        return obs, reward, [done], [info], [gs], [next_step_dict]
    
    def get_instance_mask(self):
        return self.venv.get_instance_mask()
    
    def reset_total_cat(self, total_cat_dict, categories_in_inst):
        st()
        self.venv.reset_total_cat(total_cat_dict, categories_in_inst)

    def close(self):
        return self.venv.close()


def make_env_fn_alfred(args, scene_names, rank):
    from map_and_plan.FILM.agents.sem_exp_thor import Sem_Exp_Env_Agent_Thor
    env = Sem_Exp_Env_Agent_Thor(args, scene_names, rank) 
    return env

def construct_envs_alfred(args, files):
    args_list = []
    scene_names_list = [[] for i in range(args.num_processes)]
    
    # files = json.load(open("alfred_data_small/splits/oct21.json"))[args.eval_split][args.from_idx:args.to_idx]
    # files = json.load(open("alfred/data/json_2.1.0.json"))[args.eval_split][args.from_idx:args.to_idx]
    for e, f in enumerate(files):  
        remainder = e % args.num_processes
        scene_names_list[remainder].append(f)
    del files
    for i in range(args.num_processes):
        args_list.append(args)
    
    if args.num_processes==1 and not args.run_film_only:
        envs = make_env_fn_alfred(args_list[0], scene_names_list[0], 0)
        envs.num_envs = 1
    else:
        envs = VectorEnv(make_env_fn=make_env_fn_alfred,
                        env_fn_args=tuple(tuple(zip(args_list, scene_names_list, range(args.num_processes)))))
    
    return envs

