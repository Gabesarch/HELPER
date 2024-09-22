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
import random

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

logging.basicConfig(
            level=logging.DEBUG,
            format='%(asctime)s %(levelname)s %(message)s',
                filename='./subgoalcontroller.log',
                filemode='w'
            )

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

    def save_llm_outofplace_examples(self):
        scene_graph = self.get_relations_pickupable() 

        oop_names = self.env.oop_IDs

        text = ''
        for id_ in scene_graph.keys():
            is_out_of_place = scene_graph[id_]["name"] in oop_names
            answer = 'yes' if is_out_of_place else 'no'
            cat_text = self.utils.format_class_name(scene_graph[id_]["category"])
            text = f'Input:\n"""\nObject Category: {cat_text}, Object Description: {". ".join(scene_graph[id_]["relations"])}\n"""\nOutput:\n"""\nIs this object out of place? {answer}\n"""'
            if is_out_of_place:
                example_file_name = f'prompt/examples/examples_oop/{self.tag}_{scene_graph[id_]["name"]}.txt'
            else:
                example_file_name = f'prompt/examples/examples_ip/{self.tag}_{scene_graph[id_]["name"]}.txt'
            with open(example_file_name, 'w') as f:
                f.write(text)

        objects_clean = self.env.objects_original
        receptacles = []
        for obj in objects_clean:
            if obj['objectType'] in self.RECEPTACLE_OBJECTS:
                receptacles.append(self.utils.format_class_name(obj['objectType']))

        targets = []
        for obj in objects_clean:
            if obj['objectType'] in self.PICKUPABLE_OBJECTS:
                if obj['parentReceptacles'] is None:
                    continue
                targets.append([self.utils.format_class_name(obj['objectType']), self.utils.format_class_name(obj['parentReceptacles'][0].split('|')[0])])
        
        for target in targets:
            target_ = target[0]
            answer = target[1]
            text = f'Input:\n"""\nTarget Object: {target_}\nContainer List: {", ".join(receptacles)}\n"""\nOutput:\n"""\nPlace the target item in or on the following container: {answer}\n"""'
            number = random.randint(1000,9999)
            example_file_name = f'prompt/examples/examples_placement/{self.tag}_{target_}_{number}.txt'
            with open(example_file_name, 'w') as f:
                f.write(text)

    def run_llm(self, log_tag=''):

        instruction = "Tidy up the house. "

        object_dict = self.object_tracker.objects_track_dict
        if args.do_random_oop:
            valid_ids = []
            valid_labels = []
            for k in object_dict.keys():
                if object_dict[k]["label"] in self.PICKUPABLE_OBJECTS:
                    valid_ids.append(k)
                    valid_labels.append(object_dict[k]["label"])
            oop_indices = random.sample(range(len(valid_ids)), min(len(valid_ids), np.random.randint(10)))
            out_of_place_objs_ids = [v for idx, v in enumerate(valid_ids) if idx in oop_indices]
            out_of_place_objs = [v for idx, v in enumerate(valid_labels) if idx in oop_indices]
        else:
            out_of_place_objs_ids = [v for v in object_dict.keys() if object_dict[v]["out_of_place"]]
            out_of_place_objs = [object_dict[v]["label"] for v in object_dict.keys() if object_dict[v]["out_of_place"]]
        for id_ in out_of_place_objs_ids:
            self.object_tracker.objects_track_dict[id_]["scores"] = 1.01
        if len(out_of_place_objs)==0:
            out_of_place_objs = [None]
        receptacle_objs = list(set([object_dict[v]["label"] for v in object_dict.keys() if object_dict[v]["label"] in self.RECEPTACLE_OBJECTS]))
        instruction += 'These are the out of place objects: '
        for obj in out_of_place_objs:
            instruction += f'{obj}, '
        instruction = instruction[:-2] + '.'
        instruction += ' These are the receptacles in the current scene: '
        for obj in receptacle_objs:
            instruction += f'{obj}, '
        instruction = instruction[:-2] + '.'
        self.instruction = instruction
        
        if args.do_random_receptacles:
            valid_labels = []
            for k in object_dict.keys():
                if object_dict[k]["label"] in self.RECEPTACLE_OBJECTS:
                    valid_labels.append(object_dict[k]["label"])
            subgoals = []
            objects = []
            search_dict = {}
            for oop_obj in out_of_place_objs:
                receptacle = random.choice(valid_labels)
                subgoals.extend(["Navigate", "Pickup", "Navigate", "Place"])
                objects.extend([oop_obj, oop_obj, receptacle, receptacle])
            prompt = ''
            program = ''
            self.llm.search_dict = search_dict
            self.llm.command = instruction
        else:
            prompt = self.llm.get_prompt_plan(instruction)
            program = self.llm.run_gpt(prompt)
            subgoals, objects, search_dict = self.llm.response_to_subgoals(program)

        # subgoals_text = str([[i,j] for i,j in zip(subgoals, objects)])
        tbl = wandb.Table(columns=["Dialogue", "LLM output", "full_prompt"])
        tbl.add_data(instruction, program, prompt)
        wandb.log({f"LLM/oop_{self.tag}{log_tag}": tbl})

        return subgoals, objects, search_dict

    def run_llm_replan(self):
        
        # execution error messages
        execution_error = ''
        if self.task.err_message is not None:
            execution_error += self.err_message + ' '
        if self.task.help_message is not None:
            execution_error += self.help_message

        state_text = None
        completed_program = None 
        future_program = None 
        failed_program = self.llm.subgoals_to_program([self.current_subgoal], self.object_tracker.get_label_of_holding())

        prompt = self.llm.get_prompt_replan(completed_program, future_program, failed_program, execution_error, state_text)
        program = self.llm.run_gpt(prompt, log_plan=False)
        subgoals, objects, search_dict = self.llm.response_to_subgoals(program, remove_objects=False)

        command = self.instruction

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
            text_lines = ['Replan', ''] + ['Failed program:'] + failed_program.split('\n') + ['', f'Failure Feedback: {execution_error}'] + ['', 'Fix program:'] + failed_program.split('\n')
            for _ in range(40):
                self.vis.add_text_only(text_lines)

        return subgoals, objects, search_dict

    def get_search_objects(self, object_name):
        search_objects = self.llm.get_get_search_categories(object_name)
        if object_name not in self.search_dict.keys():
            self.search_dict[object_name] = []
        self.search_dict[object_name].extend(search_objects)
        self.search_dict[object_name] = list(dict.fromkeys(self.search_dict[object_name]))[:3] # top 3

    def get_relations_pickupable(self):
        
        tracker_centroids, tracker_labels, IDs = self.object_tracker.get_centroids_and_labels(return_ids=True)
        tracker_centroids[:,1] = -tracker_centroids[:,1]

        obj_rels = {}
        obj_recs = {}
        # rel_idx = 0
        for obj_i in range(len(tracker_centroids)):

            centroid = tracker_centroids[obj_i]
            obj_category_name = tracker_labels[obj_i]
            id_ = IDs[obj_i]

            key_name = f"{obj_category_name}_{id_}"

            if obj_category_name in self.RECEPTACLE_OBJECTS:
                obj_recs[key_name] = obj_category_name

            if obj_category_name not in self.PICKUPABLE_OBJECTS:
                continue

            dists = np.sqrt(np.sum((tracker_centroids - np.expand_dims(centroid, axis=0))**2, axis=1))

            # remove centroids directly overlapping
            dist_thresh = dists>0.05 #self.OT_dist_thresh
            tracker_centroids_ = tracker_centroids[dist_thresh]
            tracker_labels_ = list(np.array(tracker_labels)[dist_thresh])

            # keep only centroids of different labels to compare against
            keep = np.array(tracker_labels_)!=obj_category_name
            tracker_centroids_ = tracker_centroids_[keep]
            tracker_labels_ = list(np.array(tracker_labels_)[keep])

            keep = []
            for l in tracker_labels_:
                if l not in self.PICKUPABLE_OBJECTS:
                    keep.append(True)
                else:
                    keep.append(False)
            keep = np.array(keep)
            tracker_centroids_ = tracker_centroids_[keep]
            tracker_labels_ = list(np.array(tracker_labels_)[keep])

            # ignore floor for now
            relations = self.extract_relations_centroids(centroid, obj_category_name, tracker_centroids_, tracker_labels_, floor_height=-self.navigation.obs.camera_height)

            obj_rels[key_name] = {}
            obj_rels[key_name]['relations'] = relations
            obj_rels[key_name]['centroids'] = centroid
            obj_rels[key_name]['category'] = self.utils.format_class_name(obj_category_name)
            if args.use_gt_seg:
                obj_rels[key_name]['name'] = self.object_tracker.objects_track_dict[id_]["name"]

            # rel_idx += 1

        return obj_rels, obj_recs

    def extract_relations_centroids(self, centroid_target, label_target, obj_centroids, obj_labels, floor_height, pos_translator=None, overhead_map=None, visualize_relations=False): 

        '''Extract relationships of interest from a list of objects'''

        obj_labels_np = np.array(obj_labels.copy())

        ################# Check Relationships #################
        # check pairwise relationships. this loop is order agnostic, since pairwise relationships are mostly invertible
        if visualize_relations:
            relations_dict = {}
            for relation in self.relations_executors_pairs:
                relations_dict[relation] = []
        relations = []
        for relation in self.relations_executors_pairs:
            relation_fun = self.relations_executors_pairs[relation]
            if relation=='closest-to' or relation=='farthest-to' or relation=='supported-by':
                if relation=='supported-by':
                    if label_target in self.RECEPTACLE_OBJECTS:
                        continue
                    yes_recept = []
                    for obj_label_i in obj_labels:
                        if obj_label_i in self.RECEPTACLE_OBJECTS:
                            yes_recept.append(True)
                        else:
                            yes_recept.append(False)
                    yes_recept = np.array(yes_recept)
                    obj_centroids_ = obj_centroids[yes_recept]
                    obj_labels_ = list(obj_labels_np[yes_recept])
                    relation_ind = relation_fun(centroid_target, obj_centroids_, ground_plane_h=floor_height)
                    if relation_ind==-2:
                        pass
                    elif relation_ind==-1:
                        relations.append("The {0} is {1} the {2}".format(self.utils.format_class_name(label_target), relation.replace('-', ' '), self.utils.format_class_name('Floor')))
                        if visualize_relations:
                            relations_dict[relation].append(centroid_target)
                    else:
                        relations.append("The {0} is {1} the {2}".format(self.utils.format_class_name(label_target), relation.replace('-', ' '), self.utils.format_class_name(obj_labels_[relation_ind])))
                        if visualize_relations:
                            relations_dict[relation].append(obj_centroids_[relation_ind])
    
                else:
                    relation_ind = relation_fun(centroid_target, obj_centroids)
                    if relation_ind==-2:
                        pass
                    elif relation_ind==-1:
                        relations.append("The {0} is {1} the {2}".format(self.utils.format_class_name(label_target), relation.replace('-', ' '), self.utils.format_class_name('Floor')))
                        if visualize_relations:
                            relations_dict[relation].append(centroid_target)
                    else:
                        relations.append("The {0} is {1} the {2}".format(self.utils.format_class_name(label_target), relation.replace('-', ' '), self.utils.format_class_name(obj_labels[relation_ind])))
                        if visualize_relations:
                            relations_dict[relation].append(obj_centroids[relation_ind])
            else:
                for i in range(len(obj_centroids)):

                    is_relation = relation_fun(centroid_target, obj_centroids[i])
                
                    if is_relation:
                        relations.append("The {0} is {1} the {2}".format(self.utils.format_class_name(label_target), relation.replace('-', ' '), self.utils.format_class_name(obj_labels[i])))
                        if visualize_relations:
                            relations_dict[relation].append(obj_centroids[i])

        if visualize_relations:
            colors_rels = {
            'next-to': (0, 255, 0),
            'supported-by': (0, 255, 0),
            'closest-to': (0, 255, 255)
            }
            img = overhead_map.copy()

            c_target = pos_translator(centroid_target)
            color = (255, 0, 0)
            thickness = 1
            cv2.circle(img, c_target[[1,0]], 7, color, thickness)
            radius = 5
            for relation in list(relations_dict.keys()):
                centers_relation = relations_dict[relation]
                color = colors_rels[relation]
                for c_i in range(len(centers_relation)):
                    center_r = centers_relation[c_i]
                    c_rel_im = pos_translator(center_r)
                    cv2.circle(img, c_rel_im[[1,0]], radius, color, thickness)

            plt.figure(figsize=(8,8))
            plt.imshow(img)
            plt.savefig('images/test.png')
            st()

        return relations