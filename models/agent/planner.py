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
from task_base.alfred_base import AlfredTask
from backend import saverloader
import pickle

import numpy as np
import os

import cv2

import csv
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

import torch
import utils
import utils.geom
import logging
from prompt.run_gpt import LLMPlanner
import copy
import traceback

from task_base.aithor_base import get_rearrangement_categories

from IPython.core.debugger import set_trace
from PIL import Image
import wandb

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

torch.manual_seed(args.seed)
np.random.seed(args.seed)


class PlannerController:
    def __init__(self):
        '''
        subgoal planning controller
        inherited by models.SubGoalController
        '''
        pass        

    def run_llm(self, task_dict, log_tag=''):
        
        prompt = self.llm.get_prompt_plan(task_dict)
        program = self.llm.run_gpt(prompt)
        subgoals, objects, search_dict = self.llm.response_to_subgoals(program)

        hl_goal = task_dict['turk_annotations']['anns'][task_dict['repeat_idx']]['task_desc']
        ll_subgoals = task_dict['turk_annotations']['anns'][task_dict['repeat_idx']]['high_descs']
        command = f'High Level Goal: {hl_goal}\nLow Level Goal:\n'
        subgoal_num = 1
        for subgoal in ll_subgoals:
            command += f'({subgoal_num}) {subgoal}\n'
            subgoal_num += 1

        subgoals_text = str([[i,j] for i,j in zip(subgoals, objects)])
        tbl = wandb.Table(columns=["Dialogue", "LLM output", "subgoals", "full_prompt"])
        tbl.add_data(command, program, subgoals_text, prompt)
        wandb.log({f"LLM/{self.tag}{log_tag}": tbl})
        self.llm_log = {"Dialogue":command, "LLM output":program, "subgoals":subgoals_text, "full_prompt":prompt}

        save_output = False
        if save_output:
            assert NotImplementedError
            os.makedirs(args.llm_output_dir, exist_ok=True)
            path = os.path.join(args.llm_output_dir, f'{self.tag}_llm_program{log_tag}.txt')
            with open(path, "w") as fobj:
                fobj.write('-----DIALOGUE----\n')
                for line in task_dict['dialog_history_cleaned']:
                    fobj.write(f'<{line[0]}> {line[1]}\n')
                fobj.write('\n\n\n-----LLM OUTPUT----\n')
                fobj.write(program)
            path = os.path.join(args.llm_output_dir, f'{self.tag}_llm_subgoals{log_tag}.txt')
            with open(path, "w") as f:
                print(subgoals, file=f)
                print(objects, file=f)

            path = os.path.join(args.llm_output_dir, f'{self.tag}_llm_fullprompt{log_tag}.txt')
            with open(path, "w") as fobj:
                fobj.write(prompt)

        if self.vis is not None:
            text_lines = ['Plan'] + program.split('\n')
            for _ in range(40):
                self.vis.add_text_only(text_lines)

        return subgoals, objects, search_dict

    def run_llm_replan(self, task_dict):
        
        # execution error messages
        execution_error = ''
        if self.task.err_message is not None:
            execution_error += self.err_message + ' '
        if self.task.help_message is not None:
            execution_error += self.help_message

        state_text = None
        completed_program = None #self.llm.subgoals_to_program(self.completed_subgoals)
        future_program = None #self.llm.subgoals_to_program(self.future_subgoals, self.object_tracker.get_label_of_holding())
        failed_program = self.llm.subgoals_to_program([self.current_subgoal], self.object_tracker.get_label_of_holding())

        prompt = self.llm.get_prompt_replan(completed_program, future_program, failed_program, execution_error, state_text)
        program = self.llm.run_gpt(prompt, log_plan=False)
        subgoals, objects, search_dict = self.llm.response_to_subgoals(program, remove_objects=False)

        hl_goal = task_dict['turk_annotations']['anns'][task_dict['repeat_idx']]['task_desc']
        ll_subgoals = task_dict['turk_annotations']['anns'][task_dict['repeat_idx']]['high_descs']
        command = f'High Level Goal: {hl_goal}\nLow Level Goal:\n'
        subgoal_num = 1
        for subgoal in ll_subgoals:
            command += f'({subgoal_num}) {subgoal}\n'
            subgoal_num += 1

        subgoals_text = str([[i,j] for i,j in zip(subgoals, objects)])
        print(f"Corrective subgoals:\n {subgoals_text}")
        tbl = wandb.Table(columns=["Dialogue", "LLM output", "subgoals", "retrieved", "replan_prompt"])
        tbl.add_data(command, program, subgoals_text, self.llm.examples, self.llm.populated_replan_prompt)
        wandb.log({f"LLM_replan/{self.tag}_replan{self.replan_num}": tbl})

        self.replan_num += 1
        
        save_output = False
        if save_output:
            os.makedirs(args.llm_output_dir, exist_ok=True)
            path = os.path.join(args.llm_output_dir, f'{self.tag}_llm_replan{self.replan_num}_fullprompt.txt')
            with open(path, "w") as fobj:
                fobj.write(prompt)

        if self.vis is not None:
            text_lines = ['Replan', ''] + ['Failed program:'] + failed_program.split('\n') + ['', f'Failure Feedback: {execution_error}'] + ['', 'Fix program:'] + program.split('\n')
            for _ in range(40):
                self.vis.add_text_only(text_lines)

        return subgoals, objects, search_dict

    def get_search_objects(self, object_name):
        search_objects = self.llm.get_get_search_categories(object_name)
        if object_name not in self.search_dict.keys():
            self.search_dict[object_name] = []
        self.search_dict[object_name].extend(search_objects)
        self.search_dict[object_name] = list(dict.fromkeys(self.search_dict[object_name]))[:3] # top 3