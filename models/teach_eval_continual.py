import sys
import json

import ipdb
st = ipdb.set_trace
from arguments import args

from time import sleep
from typing import List
import matplotlib.pyplot as plt
from ai2thor.controller import Controller

from task_base.object_tracker import ObjectTrack
from task_base.navigation import Navigation
from task_base.animation_util import Animation
from task_base.teach_base import TeachTask
from backend import saverloader
import pickle

import numpy as np
import os
import glob
import cv2

import csv
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

from teach.dataset.dataset import Dataset
from teach.dataset.definitions import Definitions
from teach.logger import create_logger
from teach.simulators import simulator_factory
from teach.utils import get_state_changes, reduce_float_precision
import torch
import utils
import utils.geom
import logging
from teach.replay.episode_replay import EpisodeReplay
if args.mode in ["teach_eval_tfd", "teach_eval_custom", "teach_eval_continual"]:
    from teach.inference.tfd_inference_runner import TfdInferenceRunner as InferenceRunner
elif args.mode=="teach_eval_edh":
    from teach.inference.edh_inference_runner import EdhInferenceRunner as InferenceRunner
else:
    assert(False) # what mode is this? 
from teach.inference.edh_inference_runner import InferenceRunnerConfig
from teach.utils import (
    create_task_thor_from_state_diff,
    load_images,
    save_dict_as_json,
    with_retry,
    load_json
)
from teach.eval.compute_metrics import create_new_traj_metrics, evaluate_traj
from prompt.run_gpt import LLMPlanner
import copy
import traceback

from task_base.aithor_base import get_rearrangement_categories

import openai

logging.basicConfig(
            level=logging.DEBUG,
            format='%(asctime)s %(levelname)s %(message)s',
                filename='./subgoalcontroller.log',
                filemode='w'
            )

from IPython.core.debugger import set_trace
from PIL import Image
import wandb

from .teach_eval_embodied_llm import SubGoalController

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

torch.manual_seed(args.seed)
np.random.seed(args.seed)

class ContinualSubGoalController(SubGoalController):
    def __init__(
        self, 
        data_dir: str, 
        output_dir: str, 
        images_dir: str, 
        edh_instance: str = None, 
        max_init_tries: int =5, 
        replay_timeout: int = 500, 
        num_processes: int = 1, 
        iteration=0,
        er=None,
        embeddings_continual=None,
        file_order_continual=None,
        depth_network=None,
        segmentation_network=None,
        ) -> None:

        super(ContinualSubGoalController, self).__init__(data_dir, output_dir, images_dir, edh_instance, max_init_tries, replay_timeout, num_processes, iteration, er, depth_network=depth_network, segmentation_network=segmentation_network)

        # self.embeddings_original = self.llm.embeddings
        # self.file_order_original = self.llm.file_order

        if embeddings_continual is not None and file_order_continual is not None:
            self.llm.embeddings = np.concatenate([self.llm.embeddings, embeddings_continual], axis=0)
            self.llm.file_order = self.llm.file_order + file_order_continual

    def run(self):
        if args.mode in ["teach_eval_tfd", "teach_eval_custom", "teach_eval_continual"]:
            self.run_tfd(user_progress_check=self.use_progress_check)
        elif args.mode=="teach_eval_edh":
            self.run_edh()
        self.teach_task.step("Stop", None)

        if self.controller is not None:
            self.controller.stop()
            self.eval()
        self.render_output()

        if self.teach_task.metrics["success"]:
            self.add_plan_to_api()

        return self.teach_task.metrics, self.er

    def add_plan_to_api(self):
        program = self.llm.subgoals_to_program(self.completed_subgoals)
        dialogue = '' #'-----DIALOGUE----\n'
        for line in self.edh_instance['dialog_history_cleaned']:
            dialogue += f'<{line[0]}> {line[1]} '
        with open('prompt/prompt_continual.txt') as f:
            prompt_continual = f.read()
        # prompt_continual = prompt_continual.replace('{dialogue}', f'{dialogue}')
        # prompt_continual = prompt_continual.replace('{program}', f'{program}')
        # llm_output = self.llm.run_gpt(prompt_continual, log_plan=False)
        root = os.path.join('output', 'continual', f'{args.set_name}')
        os.makedirs(root, exist_ok=True)
        # num_files = len(os.listdir(root))
        file = os.path.join(root, f'{self.tag}.txt')
        text = f'dialogue: {dialogue}\nPython script:\n{program}'
        with open(file, 'w') as f:
            f.write(text)

def get_gpt_embeddings_continual():
    root = os.path.join('output', 'continual', f'{args.set_name}')
    files_iterate = glob.glob(f"{root}/*.txt")
    filenames = []
    embeddings = np.zeros((len(files_iterate), 1536), dtype=np.float64)
    for f_i in range(len(files_iterate)):
        
        file = files_iterate[f_i]
        with open(file) as f:
            prompt = f.read()
        # jsut keep description
        # prompt = prompt.split('\n')[0]
        prompt = prompt.split('\n')[0].split('dialogue: ')[-1]
        print(prompt)

        messages = [
        {"role": "user", "content": prompt},
        ]
        if args.use_openai:
            embedding = openai.Embedding.create(
                        model="text-embedding-ada-002",
                        input=prompt,
                        )['data'][0]['embedding']
        else:
            embedding = openai.Embedding.create(
                        engine="kateftextembeddingada002",
                        input=prompt,
                        )['data'][0]['embedding']
        embedding = np.asarray(embedding)
        embeddings[f_i] = embedding

        # file_ = file.split('/')[-1]
        # file_ = os.path.join('prompt', file)
        filenames.append(file)

    # embedding_dir = os.path.join(root, 'embeddings.npy')
    # np.save(embedding_dir, embeddings)
    # file_order = os.path.join(root, 'file_order.txt')
    # with open(file_order, 'w') as fp:
    #     fp.write("\n".join(str(item) for item in filenames))

    return embeddings, filenames

# def get_topk_functions()

def run_continual():
    save_metrics = True
    split_ = args.split
    data_dir = args.teach_data_dir #"/projects/katefgroup/embodied_llm/dataset"
    output_dir = "./plots/subgoal_output"
    images_dir = "./plots/subgoal_output"
    # if args.mode=="teach_eval_tfd":
    #     instance_dir = os.path.join(data_dir, f"tfd_instances/{split_}")
    # elif args.mode=="teach_eval_edh":
    #     instance_dir = os.path.join(data_dir, f"edh_instances/{split_}")
    instance_dir = os.path.join(data_dir, f"tfd_instances/{split_}")
    files = os.listdir(instance_dir) # sample every other

    # print(len(os.listdir(instance_dir.replace(split_, "valid_seen"))))

    if args.sample_every_other:
        files = files[::2]

    # # files = files[:20]   
    # files_idx = files.index('c66573f82df8618c_aad3.tfd.json')
    # files_idx = files.index('4364fccce0c7e619_be95.tfd.json')
    # files_idx = files.index('5ca98b7c823fa3e9_cf7a.tfd.json')
    # files_idx = files.index('f25eed33d1c37931_1a7f.tfd.json')
    # files_idx = files.index('24e4a25886792580_1445.tfd.json')
    # files_idx = files.index('8463e3c141964336_c9cc.tfd.json')
    # files_idx = files.index('ce0a11d25e229099_8adc.tfd.json') # prompt too long
    # files_idx = files.index('612f99763b6978b6_0c1a.tfd.json')
    # files_idx = files.index('0fc859faf40296d7_c87e.tfd.json')
    # files_idx = files.index('064a69cb5d40b41f_16f3.tfd.json')
    # files_idx = files.index('f511e02d3f84b212_e6d3.tfd.json')
    # files_idx = files.index('88f8ad0d2d356270_bb95.tfd.json')
    # files_idx = files.index('9d1c053084e66e7b_9e02.tfd.json')
    # files_idx = files.index('f9b2ea9da7e5220a_b121.tfd.json')
    # files_idx = files.index('36d6493fe1183e3f_975d.tfd.json')
    # files_idx = files.index('f511e02d3f84b212_e6d3.tfd.json')
    # e05a
    # files = files[files_idx:files_idx+1]
    # files = files[2:3]
    if args.episode_file is not None:
        files_idx = files.index(args.episode_file)
        files = files[files_idx:files_idx+1]

    if args.max_episodes is not None:
        files = files[:args.max_episodes]
    # files = files[:1]

    # initialize wandb
    if args.set_name=="test00":
        wandb.init(mode="disabled")
    else:
        wandb.init(project="embodied-llm-teach", name=args.set_name, group=args.group, config=args, dir=args.wandb_directory)

    # metrics = {}
    # metrics_before_feedback = {}
    # metrics_file = os.path.join(args.metrics_dir, f'{args.mode}_metrics_{split_}.txt')
    # metrics_file_before_feedback = os.path.join(args.metrics_dir, f'{args.mode}_metrics_before_feedback_{split_}.txt')
    # if os.path.exists(metrics_file):
    #     metrics = load_json(metrics_file)
    # if os.path.exists(metrics_file_before_feedback):
    #     metrics_before_feedback = load_json(metrics_file_before_feedback)
    # iter_ = 0
    # er = None
    # depth_estimation_network = None
    # segmentation_network = None

    for continual_iter in range(args.num_continual_iter):
        metrics = {}
        metrics_before_feedback = {}
        metrics_file = os.path.join(args.metrics_dir, f'{args.mode}_metrics_{split_}_continualiter{continual_iter}.txt')
        metrics_file_before_feedback = os.path.join(args.metrics_dir, f'{args.mode}_metrics_before_feedback_{split_}_continualiter{continual_iter}.txt')
        if os.path.exists(metrics_file) and args.skip_if_exists:
            metrics = load_json(metrics_file)
        if os.path.exists(metrics_file_before_feedback) and args.skip_if_exists:
            metrics_before_feedback = load_json(metrics_file_before_feedback)
        iter_ = 0
        er = None
        if continual_iter>0:
            embeddings_continual, file_order_continual = get_gpt_embeddings_continual()
        else:
            embeddings_continual, file_order_continual = None, None
        depth_estimation_network = None
        segmentation_network = None
        for file in files:
            print("Running ", file)
            print(f"Iteration {iter_+1}/{len(files)}")
            if args.skip_if_exists and (file in metrics.keys()):
                print(f"File already in metrics... skipping...")
                iter_ += 1
                continue
            task_instance = os.path.join(instance_dir, file)
            subgoalcontroller = ContinualSubGoalController(
                data_dir, 
                output_dir, 
                images_dir, 
                task_instance, 
                iteration=iter_, 
                er=er, 
                embeddings_continual=embeddings_continual,
                file_order_continual=file_order_continual,
                depth_network=depth_estimation_network, 
                segmentation_network=segmentation_network
                )
            if subgoalcontroller.init_success:
                metrics_instance, er = subgoalcontroller.run()
                if segmentation_network is None:
                    segmentation_network = subgoalcontroller.object_tracker.ddetr
                if depth_estimation_network is None:
                    depth_estimation_network = subgoalcontroller.navigation.depth_estimator
            else:
                metrics_instance, er = subgoalcontroller.teach_task.metrics, subgoalcontroller.er
            metrics_instance_before_feedback = subgoalcontroller.metrics_before_feedback
            metrics[file] = metrics_instance
            metrics_before_feedback[file] = metrics_instance_before_feedback
            iter_ += 1

            if save_metrics:
                keys_include = ['goal_condition_success', 'success', 'success_spl', 'path_len_weighted_success_spl', 'goal_condition_spl', 'path_len_weighted_goal_condition_spl']
                metrics_avg = {}
                for f_n in keys_include:
                    metrics_avg[f_n] = 0
                count = 0
                for k in metrics.keys():
                    for f_n in keys_include:
                        metrics_avg[f_n] += metrics[k][f_n]
                    count += 1
                for f_n in keys_include:
                    metrics_avg[f_n] /=  count 

                to_log = []  
                to_log.append('-'*40 + '-'*40)
                list_of_files = list(metrics.keys())
                to_log.append(f'Files: {str(list_of_files)}')
                to_log.append(f'Split: {split_}')
                to_log.append(f'Number of files: {len(list_of_files)}')
                for f_n in keys_include:
                    to_log.append(f'{f_n}: {metrics_avg[f_n]}') 
                to_log.append('-'*40 + '-'*40)

                os.makedirs(args.metrics_dir, exist_ok=True)
                path = os.path.join(args.metrics_dir, f'{args.mode}_summary_{split_}_continualiter{continual_iter}.txt')
                with open(path, "w") as fobj:
                    for x in to_log:
                        fobj.write(x + "\n")

                save_dict_as_json(metrics, metrics_file)

                metrics_avg["num episodes"] = iter_
                tbl = wandb.Table(columns=list(metrics_avg.keys()))
                tbl.add_data(*list(metrics_avg.values()))
                wandb.log({f"Metrics_summary/Summary_continualiter{continual_iter}": tbl, 'step':iter_})

                # before feedback
                metrics_avg_before_feedback = {}
                for f_n in keys_include:
                    metrics_avg_before_feedback[f_n] = 0
                count = 0
                for k in metrics.keys():
                    for f_n in keys_include:
                        metrics_avg_before_feedback[f_n] += metrics_before_feedback[k][f_n]
                    count += 1
                for f_n in keys_include:
                    metrics_avg_before_feedback[f_n] /=  count 

                metrics_avg_before_feedback["num episodes"] = iter_
                tbl = wandb.Table(columns=list(metrics_avg_before_feedback.keys()))
                tbl.add_data(*list(metrics_avg_before_feedback.values()))
                wandb.log({f"Metrics_summary/Summary_before_feedback_continualiter{continual_iter}": tbl, 'step':iter_})

                save_dict_as_json(metrics_before_feedback, metrics_file_before_feedback)

                cols = ["file"]+list(metrics_instance.keys())
                cols.remove('pred_actions')
                tbl = wandb.Table(columns=cols)
                for f_k in metrics.keys():
                    to_add_tbl = [f_k]
                    for k in list(metrics[f_k].keys()):
                        if k=="pred_actions":
                            continue
                        to_add_tbl.append(metrics[f_k][k])
                    # list_values = [f_k] + list(metrics[f_k].values())
                    tbl.add_data(*to_add_tbl)
                wandb.log({f"Metrics_summary/Metrics_continualiter{continual_iter}": tbl, 'step':iter_})

                cols = ["file"]+list(metrics_instance_before_feedback.keys())
                cols.remove('pred_actions')
                tbl = wandb.Table(columns=cols)
                for f_k in metrics_before_feedback.keys():
                    to_add_tbl = [f_k]
                    for k in list(metrics_before_feedback[f_k].keys()):
                        if k=="pred_actions":
                            continue
                        to_add_tbl.append(metrics_before_feedback[f_k][k])
                    # list_values = [f_k] + list(metrics[f_k].values())
                    tbl.add_data(*to_add_tbl)
                wandb.log({f"Metrics_summary/Metrics_continualiter{continual_iter}": tbl, 'step':iter_})

