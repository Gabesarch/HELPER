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
if args.mode in ["teach_eval_tfd", "teach_eval_custom"]:
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

class CustomSubGoalController(SubGoalController):
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
        ) -> None:

        super(CustomSubGoalController, self).__init__(data_dir, output_dir, images_dir, edh_instance, max_init_tries, replay_timeout, num_processes, iteration, er)

        self.customize_instance()

    def customize_instance(self):
        # replace with custom instructions
        custom_file = os.path.join(self.runner_config.data_dir, 'custom', self.runner_config.split, f"{self.edh_instance['game_id']}.custom.json")
        # custom_file = edh_instance.replace('tfd_instances', 'custom').replace('.tfd.json', '.custom.json')
        custom_dict = load_json(custom_file)
        self.edh_instance.update(custom_dict)
        self.edh_instance['dialog_history_cleaned'] = [['Driver', 'What is my task?'], ['Commander', 'Make me the Gabe sandwich']]
        # self.edh_instance['dialog_history_cleaned'] = [['Driver', 'What is my task?'], ['Commander', 'Make me the Gabe sandwich with only 1 slice of tomato instead of 2']]


        self.edh_instance['dialog_history_cleaned'] = [['Driver', 'What is my task?'], ['Commander', "Make me a sandwich. The name of this sandwich is called the Gabe sandwich. The sandwich has two slices of toast, 2 slices of tomato, and 1 slice of lettuce on a clean plate."]]
        self.edh_instance['dialog_history_cleaned'] = [['Driver', 'What is my task?'], ['Commander', "Make me a sandwich. The name of this sandwich is called the Larry sandwich. The sandwich has two slices of toast, 3 slices of tomato, and 3 slice of lettuce on a clean plate."]]
        self.edh_instance['dialog_history_cleaned'] = [['Driver', 'What is my task?'], ['Commander', 'Make me a salad. The name of this salad is called the David salad. The salad has two slices of tomato and three slices of lettuce on a clean plate.']]
        self.edh_instance['dialog_history_cleaned'] = [['Driver', 'What is my task?'], ['Commander', "Make me a salad. The name of this salad is called the Dax salad. The salad has two slices of cooked potato. You'll need to cook the potato on the stove. The salad also has a slice of lettuce and a slice of tomato. Put all components on a clean plate."]]
        self.edh_instance['dialog_history_cleaned'] = [['Driver', 'What is my task?'], ['Commander', 'Make me breakfast. The name of this breakfast is called the Mary breakfast. The breakfast has a mug of coffee, and two slices of toast on a clean plate.']]
        self.edh_instance['dialog_history_cleaned'] = [['Driver', 'What is my task?'], ['Commander', 'Make me breakfast. The name of this breakfast is called the Lion breakfast. The breakfast has a mug of coffee, and four slices of tomato on a clean plate.']]
        self.edh_instance['dialog_history_cleaned'] = [['Driver', 'What is my task?'], ['Commander', 'Rearrange some objects. The name of this rearrangement is called the Lax rearrangement. Place three pillows on the sofa.']]
        self.edh_instance['dialog_history_cleaned'] = [['Driver', 'What is my task?'], ['Commander', 'Rearrange some objects. The name of this rearrangement is called the Pax rearrangement. Place two pencils and two pens on the desk.']]
        self.edh_instance['dialog_history_cleaned'] = [['Driver', 'What is my task?'], ['Commander', 'Clean some objects. The name of this cleaning is called the Gax cleaning. Clean two plates and two cups.']]
        self.edh_instance['dialog_history_cleaned'] = [['Driver', 'What is my task?'], ['Commander', 'Clean some objects. The name of this cleaning is called the Kax cleaning. Clean a mug and a pan.']]

        files = [
            "prompt/examples/examples_custom/gabe_sandwich.txt",
            "prompt/examples/examples_custom/larry_sandwich.txt",
            "prompt/examples/examples_custom/david_salad.txt",
            "prompt/examples/examples_custom/dax_salad.txt",
            "prompt/examples/examples_custom/mary_breakfast.txt",
            "prompt/examples/examples_custom/lion_breakfast.txt",
            "prompt/examples/examples_custom/lax_placement.txt",
            "prompt/examples/examples_custom/pax_placement.txt",
            "prompt/examples/examples_custom/gax_clean.txt",
            "prompt/examples/examples_custom/kax_clean.txt",
        ]

        no_change = [
            "Make me the Gabe sandwich", 
            "Make me the Larry sandwich", 
            "Make me the David salad",
            "Make me the Dax salad",
            "Make me the Mary breakfast",
            "Make me the Lion breakfast",
            "Complete the Lax rearrangement",
            "Complete the Pax rearrangement",
            "Perform the Gax cleaning",
            "Perform the Kax cleaning",
            ]

        one_change = [
            "Make me the Gabe sandwich with only 1 slice of tomato", 
            "Make me the Larry sandwich with four slices of lettuce", 
            "Make me the David salad with a slice of potato",
            "Make me the Dax salad without lettuce",
            "Make me the Mary breakfast with no coffee",
            "Make me the Lion breakfast with three slice of tomato",
            "Complete the Lax rearrangement with two pillows",
            "Complete the Pax rearrangement but use one pencil instead of the the two pencils",
            "Perform the Gax cleaning with three plates instead of two",
            "Perform the Kax cleaning with only a mug",
            ]

        two_change = [
            "Make me the Gabe sandwich with only 1 slice of tomato and two slices of lettuce", 
            "Make me the Larry sandwich with four slices of lettuce and two slices of tomato", 
            "Make me the David salad but add a slice of potato and add one slice of egg",
            "Make me the Dax salad without lettuce and without potato",
            "Make me the Mary breakfast with no coffee and add an egg",
            "Make me the Lion breakfast with three slice of tomato and two mugs of coffee",
            "Complete the Lax rearrangement with two pillows and add a remote",
            "Complete the Pax rearrangement but use one pencil instead of the two pencils and add a book",
            "Perform the Gax cleaning with three plates instead of the two plates and include a fork",
            "Perform the Kax cleaning without the pan and include a spoon",
            ]

        three_change = [
            "Make me the Gabe sandwich with only 1 slice of tomato, two slices of lettuce, and add a slice of egg", 
            "Make me the Larry sandwich with four slices of lettuce, two slices of tomato, and place all components directly on the countertop", 
            "Make me the David salad and add a slice of potato, add one slice of egg, and bring a fork with it",
            "Make me the Dax salad without lettuce, without potato, and add an extra slice of tomato",
            "Make me the Mary breakfast with no coffee, add an egg, and add a cup filled with water",
            "Make me the Lion breakfast with three slice of tomato, two mugs of coffee, and add a fork",
            "Complete the Lax rearrangement with two pillows, a remote, and place it on the arm chair instead",
            "Complete the Pax rearrangement but use one pencil instead of the two pencils and include a book and a baseball bat",
            "Perform the Gax cleaning with three plates instead of the two plates, include a fork, and do not clean any cups",
            "Perform the Kax cleaning without the pan, include a spoon, and include a pot",
            ]

        correct = []
        for idx in range(len(no_change)):
            self.edh_instance['dialog_history_cleaned'] = [['Driver', 'What is my task?'], ['Commander', f'{no_change[idx]}']]
            program = self.run_llm(self.edh_instance)
            with open(files[idx]) as f:
                actual = f.read()
            text = f'\n\n\nActual:\n{actual}\n\nPredicted:\n{program}'
            print(text)
            correct_ = input('Correct?(y/n)\n')
            if correct_=="y":
                correct.append(True)
            else:
                correct.append(False)
        correct_total = sum(correct)/len(correct)
        print(f"Percent correct no change: {correct_total}")
        # Percent correct no change: 1.0

        correct = []
        for idx in range(len(one_change)):
            self.edh_instance['dialog_history_cleaned'] = [['Driver', 'What is my task?'], ['Commander', f'{one_change[idx]}']]
            program = self.run_llm(self.edh_instance)
            with open(files[idx]) as f:
                actual = f.read()
            print(f'\n\n\n{one_change[idx]}')
            text = f'\n\n\nActual:\n{actual}\n\nPredicted:\n{program}'
            print(text)
            correct_ = input('Correct?(y/n)\n')
            if correct_=="y":
                correct.append(True)
            else:
                correct.append(False)
        correct_total = sum(correct)/len(correct)
        print(f"Percent correct one change: {correct_total}")
        # Percent correct one change: 1.0
        # completely removes pens when not specified 

        correct = []
        for idx in range(len(two_change)):
            self.edh_instance['dialog_history_cleaned'] = [['Driver', 'What is my task?'], ['Commander', f'{two_change[idx]}']]
            program = self.run_llm(self.edh_instance)
            with open(files[idx]) as f:
                actual = f.read()
            print(f'\n\n\n{two_change[idx]}')
            text = f'\n\n\nActual:\n{actual}\n\nPredicted:\n{program}'
            print(text)
            correct_ = input('Correct?(y/n)\n')
            if correct_=="y":
                correct.append(True)
            else:
                correct.append(False)
        correct_total = sum(correct)/len(correct)
        print(f"Percent correct two change: {correct_total}")
        # plan includes two coffees but it tries to place both in the coffee maker at the same time.
        # removes mug when it shouldnt
        # Percent correct two change: 0.8

        correct = []
        for idx in range(len(three_change)):
            self.edh_instance['dialog_history_cleaned'] = [['Driver', 'What is my task?'], ['Commander', f'{three_change[idx]}']]
            program = self.run_llm(self.edh_instance)
            with open(files[idx]) as f:
                actual = f.read()
            print(f'\n\n\n{three_change[idx]}')
            text = f'\n\n\nActual:\n{actual}\n\nPredicted:\n{program}'
            print(text)
            correct_ = input('Correct?(y/n)\n')
            if correct_=="y":
                correct.append(True)
            else:
                correct.append(False)
        correct_total = sum(correct)/len(correct)
        print(f"Percent correct two change: {correct_total}")
        # again, mug planning in wrong but adds fork and tomato change
        # Percent correct two change: 0.9

        assert(False) # END

    def run_llm(self, task_dict, log_tag=''):
        
        prompt = self.llm.get_prompt_plan(task_dict)
        program = self.llm.run_gpt(prompt)

        return program


def run_custom():
    save_metrics = True
    split_ = args.split
    data_dir = args.teach_data_dir
    output_dir = "./plots/subgoal_output"
    images_dir = "./plots/subgoal_output"
    instance_dir = instance_dir = os.path.join(data_dir, f"tfd_instances/{split_}")
    files = os.listdir(instance_dir) # sample every other

    if args.sample_every_other:
        files = files[::2]

    files_idx = files.index('f511e02d3f84b212_e6d3.tfd.json')
    files_idx = files.index('88f8ad0d2d356270_bb95.tfd.json')
    files_idx = files.index('9d1c053084e66e7b_9e02.tfd.json')
    files_idx = files.index('f9b2ea9da7e5220a_b121.tfd.json')
    files_idx = files.index('36d6493fe1183e3f_975d.tfd.json')
    files_idx = files.index('f511e02d3f84b212_e6d3.tfd.json')
    # e05a
    files = files[files_idx:files_idx+1]

    files = files[:100]

    # initialize wandb
    if args.set_name=="test00":
        wandb.init(mode="disabled")
    else:
        wandb.init(project="embodied-llm-teach", name=args.set_name, group=args.group, config=args, dir=args.wandb_directory)

    metrics = {}
    metrics_before_feedback = {}
    iter_ = 0
    er = None
    for file in files:
        print("Running ", file)
        print(f"Iteration {iter_+1}/{len(files)}")
        task_instance = os.path.join(instance_dir, file)
        subgoalcontroller = CustomSubGoalController(data_dir, output_dir, images_dir, task_instance, iteration=iter_, er=er)
        if subgoalcontroller.init_success:
            metrics_instance, er = subgoalcontroller.run()
        else:
            metrics_instance, er = subgoalcontroller.metrics, subgoalcontroller.er
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
            path = os.path.join(args.metrics_dir, 'metrics_teach_EDH_summary.txt')
            with open(path, "w") as fobj:
                for x in to_log:
                    fobj.write(x + "\n")

            metrics_file = os.path.join(args.metrics_dir, 'metrics_teach_EDH_instances.txt')
            save_dict_as_json(metrics, metrics_file)

            metrics_avg["num episodes"] = iter_
            tbl = wandb.Table(columns=list(metrics_avg.keys()))
            tbl.add_data(*list(metrics_avg.values()))
            wandb.log({f"Metrics/Summary": tbl, 'step':iter_})

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
            wandb.log({f"Metrics/Summary_before_feedback": tbl, 'step':iter_})

