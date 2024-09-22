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
from task_base.tidy_base import TidyTask
from ai2thor.controller import Controller
from backend import saverloader
import pickle
from utils.wctb import Utils, Relations_CenterOnly

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
import glob

from utils.misc import save_dict_as_json

from task_base.aithor_base import get_rearrangement_categories
from tidy_task.tidy_task import TIDEE_TASK

logging.basicConfig(
            level=logging.DEBUG,
            format='%(asctime)s %(levelname)s %(message)s',
                filename='./subgoalcontroller.log',
                filemode='w'
            )

from IPython.core.debugger import set_trace
from PIL import Image
import wandb

from .agent.executor import ExecuteController
from .agent.planner import PlannerController

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

torch.manual_seed(args.seed)
np.random.seed(args.seed)

class SubGoalController(ExecuteController, PlannerController):
    def __init__(
        self, 
        max_init_tries: int =5, 
        replay_timeout: int = 500, 
        num_processes: int = 1, 
        iteration=0,
        env=None,
        depth_network=None,
        segmentation_network=None,
        ) -> None:

        super(SubGoalController, self).__init__()

        self.env = env

        split = args.split

        ###########%%%%%% PARAMS %%%%%%%#########
        keep_head_down = True # keep head oriented dwon when navigating (want this True if estimating depth)
        keep_head_straight = False # keep head oriented straight when navigating (want this False if estimating depth)

        self.log_every = args.log_every # log every X iters if generating video

        self.teleport_to_objs = args.teleport_to_objs # teleport to objects instead of navigating
        self.render = args.create_movie and (iteration % self.log_every == 0) # render video? NOTE: video is rendered to self.root folder
        use_gt_objecttrack = args.use_gt_seg # if navigating, use GT object masks for getting object detections + centroids?
        use_gt_depth = args.use_gt_depth # if navigating, use GT depth maps? 
        self.use_GT_seg_for_interaction = args.use_gt_seg # use GT seg for interaction? 
        self.use_GT_constraint_checks = args.use_GT_constraint_checks
        self.use_GT_error_feedback = args.use_GT_error_feedback
        self.use_gt_subgoals = args.use_gt_subgoals
        self.use_llm_search = args.use_llm_search
        self.use_progress_check = args.use_progress_check
        self.add_back_objs_progresscheck = args.add_back_objs_progresscheck
        self.use_constraint_check = args.use_constraint_check
        self.use_mask_rcnn_pred = args.use_mask_rcnn_pred
        
        do_masks = args.do_masks # use masks from detector. If False, use boxes (Note: use_gt_objecttrack must be False)
        use_solq = args.use_solq # use SOLQ model? need this for masks
                
        self.episode_in_try_except = args.episode_in_try_except # Continue to next episode if assertion error occurs? 

        self.dist_thresh = args.dist_thresh # distance threshold for point goal navigation 
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.mode = args.mode

        if self.teleport_to_objs:
            self.use_GT_seg_for_interaction = True
            self.add_map = False
        else:
            self.add_map = True

        self.interacted_ids = [] # for teleporting to keep track
        self.navigation_action_fails = 0
        self.obj_center_camX0 = None
        self.err_message = ''
        self.help_message = ''

        self.failed_subgoals = []
        self.successful_subgoals = []
        self.error_correct_fix = []
        self.attempted_subgoals = []
        
        self.traj_steps_taken: int = 0
        self.iteration = iteration
        self.replan_num = 0
        self.num_subgoal_fail = 0
        self.max_subgoal_fail = 10
        self.errors = []
        self.could_not_find = []
        self.completed_subgoals = []
        self.object_tracker_ids_removed = []
        self.run_error_correction_llm = args.run_error_correction_llm
        self.run_error_correction_basic = args.run_error_correction_basic
        self.visibility_distance = args.visibilityDistance

        self.action_to_mappedaction = {
            'MoveAhead':"MoveAhead", 
            "RotateRight":"RotateRight", 
            "RotateLeft":"RotateLeft", 
            "LookDown":"LookDown",
            "LookUp":"LookUp",
            "Stop":"Done",
            "Place":"PutObject",
            "Pickup":"PickupObject",
            "Slice":"SliceObject",
            "ToggleOn":"ToggleObjectOn",
            "ToggleOff":"ToggleObjectOff",
            "Close":"CloseObject",
            "Open":"OpenObject",
            }

        self.include_classes = [
            'ShowerDoor', 'Cabinet', 'CounterTop', 'Sink', 'Towel', 'HandTowel', 'TowelHolder', 'SoapBar', 
            'ToiletPaper', 'ToiletPaperHanger', 'HandTowelHolder', 'SoapBottle', 'GarbageCan', 'Candle', 'ScrubBrush', 
            'Plunger', 'SinkBasin', 'Cloth', 'SprayBottle', 'Toilet', 'Faucet', 'ShowerHead', 'Box', 'Bed', 'Book', 
            'DeskLamp', 'BasketBall', 'Pen', 'Pillow', 'Pencil', 'CellPhone', 'KeyChain', 'Painting', 'CreditCard', 
            'AlarmClock', 'CD', 'Laptop', 'Drawer', 'SideTable', 'Chair', 'Blinds', 'Desk', 'Curtains', 'Dresser', 
            'Watch', 'Television', 'WateringCan', 'Newspaper', 'FloorLamp', 'RemoteControl', 'HousePlant', 'Statue', 
            'Ottoman', 'ArmChair', 'Sofa', 'DogBed', 'BaseballBat', 'TennisRacket', 'VacuumCleaner', 'Mug', 'ShelvingUnit', 
            'Shelf', 'StoveBurner', 'Apple', 'Lettuce', 'Bottle', 'Egg', 'Microwave', 'CoffeeMachine', 'Fork', 'Fridge', 
            'WineBottle', 'Spatula', 'Bread', 'Tomato', 'Pan', 'Cup', 'Pot', 'SaltShaker', 'Potato', 'PepperShaker', 
            'ButterKnife', 'StoveKnob', 'Toaster', 'DishSponge', 'Spoon', 'Plate', 'Knife', 'DiningTable', 'Bowl', 
            'LaundryHamper', 'Vase', 'Stool', 'CoffeeTable', 'Poster', 'Bathtub', 'TissueBox', 'Footstool', 'BathtubBasin', 
            'ShowerCurtain', 'TVStand', 'Boots', 'RoomDecor', 'PaperTowelRoll', 'Ladle', 'Kettle', 'Safe', 'GarbageBag', 'TeddyBear', 
            'TableTopDecor', 'Dumbbell', 'Desktop', 'AluminumFoil', 'Window', 'LightSwitch']
        self.special_classes = ['AppleSliced', 'BreadSliced', 'EggCracked', 'LettuceSliced', 'PotatoSliced', 'TomatoSliced']
        self.include_classes += self.special_classes
        if self.use_mask_rcnn_pred:
            self.include_classes += ['Cart', 'PaintingHanger', 'Glassbottle', 'LaundryHamperLid', 'PaperTowel', 'ToiletPaperRoll']
        self.include_classes.append('no_object') # ddetr has no object class

        # user defined action list
        self.NONREMOVABLE_CLASSES = ["Toilet", "Desk", "StoveKnob", "Faucet", "Fridge", "SinkBasin", "Sink", "Bed", "Microwave", "CoffeeTable", "HousePlant", "DiningTable", "Sofa", 'ArmChair', 'Toaster', 'CoffeeMachine', 'Lettuce', 'Tomato', 'Bread', 'Potato', 'Plate']
        self.FILLABLE_CLASSES = ["Bottle", "Bowl", "Cup", "HousePlant", "Kettle", "Mug", "Pot", "WateringCan", "WineBottle"]
        # self.RETRY_ACTIONS = ["Pickup", "Place", "Slice", "Pour"]
        self.RETRY_ACTIONS_IMAGE = ["Place"]
        self.RETRY_DICT_IMAGE = {a:self.include_classes for a in self.RETRY_ACTIONS_IMAGE}
        self.RETRY_DICT_IMAGE["Place"] = ["CounterTop", "Bed", "DiningTable", "CoffeeTable", "SinkBasin", "Sink", "Sofa", 'ArmChair', 'Plate']
        self.RETRY_ACTIONS_LOCATION = ["Pickup", "Place", "Slice", "Pour", "ToggleOn", "ToggleOff", "Open", "Close"]
        self.RETRY_DICT_LOCATION = {a:self.include_classes for a in self.RETRY_ACTIONS_LOCATION}
        self.OPENABLE_CLASS_LIST = set(['Fridge', 'Cabinet', 'Microwave', 'Drawer', 'Safe', 'Box'])

        _, _, self.PICKUPABLE_OBJECTS, self.OPENABLE_OBJECTS, self.RECEPTACLE_OBJECTS = get_rearrangement_categories()

        self.general_receptacles_classes = [
                'CounterTop', 'DiningTable', 'CoffeeTable', 'SideTable',
                'Desk', 'Bed', 'Sofa', 'ArmChair', 'ShelvingUnit',
                'Drawer', 'Chair', 'Shelf', 'Dresser', 'Ottoman', 'DogBed', 
                'Footstool', 'Safe', 'TVStand'
                ]

        # tidy/rearrangement specific
        self.relations_util = Relations_CenterOnly(args.H, args.W)
        self.utils = Utils(args.H, args.W)
        self.relations_executors_pairs = {
            # 'above': self.relations_util._is_above,
            # 'below': self.relations_util._is_below,
            'next-to': self.relations_util._is_next_to,
            'supported-by': self.relations_util._is_supported_by,
            # 'similar-height-to': self.relations_util._is_similar_height,
            # 'farthest-to': self.relations_util._farthest,
            'closest-to': self.relations_util._closest,
        }
        self.rel_to_id = {list(self.relations_executors_pairs.keys())[i]:i for i in range(len(self.relations_executors_pairs))}

        if not self.use_gt_subgoals:
            self.llm = LLMPlanner(
                args.gpt_embedding_dir, 
                fillable_classes=self.FILLABLE_CLASSES, 
                openable_classes=self.OPENABLE_CLASS_LIST,
                include_classes=self.include_classes,
                example_mode=args.mode,
                )

        self.name_to_id = {}
        self.id_to_name = {}
        self.instance_counter = {}
        idx = 0
        for name in self.include_classes:
            self.name_to_id[name] = idx
            self.id_to_name[idx] = name
            self.instance_counter[name] = 0
            idx += 1

        if self.use_GT_seg_for_interaction:
            self.name_to_mapped_name = {'DeskLamp':'FloorLamp', 'Sink':'SinkBasin', 'Bathtub':'BathtubBasin', 'EggCracked':'Egg'}
            self.id_to_mapped_id = {self.name_to_id[k]:self.name_to_id[v] for k, v in self.name_to_mapped_name.items()}
        else:
            self.name_to_mapped_name = {'DeskLamp':'FloorLamp', 'Sink':'SinkBasin', 'Bathtub':'BathtubBasin', 'AppleSliced':'Apple', 'BreadSliced':'Bread', 'EggCracked':'Egg', 'LettuceSliced':'Lettuce', 'PotatoSliced':'Potato', 'TomatoSliced':'Tomato'}
            if self.use_mask_rcnn_pred:
                self.name_to_mapped_name.update({'Cart':'Desk', 'PaintingHanger':'Painting', 'Glassbottle':'Bottle', 'LaundryHamperLid':'LaundryHamper','PaperTowel':'PaperTowelRoll', 'ToiletPaperRoll':'ToiletPaper'})
            self.id_to_mapped_id = {self.name_to_id[k]:self.name_to_id[v] for k, v in self.name_to_mapped_name.items()}

        self.name_to_mapped_name_subgoals = {'DeskLamp':'FloorLamp', 'Sink':'SinkBasin', 'Bathtub':'BathtubBasin', 'EggCracked':'Egg', 'Cart':'Desk', 'PaintingHanger':'Painting', 'Glassbottle':'Bottle', 'LaundryHamperLid':'LaundryHamper','PaperTowel':'PaperTowelRoll', 'ToiletPaperRoll':'ToiletPaper'}
        self.id_to_mapped_id_subgoals = {self.name_to_id[k]:self.name_to_id[v] for k, v in self.name_to_mapped_name.items()}

        self.W = args.W
        self.H = args.H
        self.web_window_size = args.W
        self.fov = args.fov
        print(f"fov: {self.fov}")
        hfov = float(self.fov) * np.pi / 180.
        self.pix_T_camX = np.array([
            [(self.W/2.)*1 / np.tan(hfov / 2.), 0., 0., 0.],
            [0., (self.H/2.)*1 / np.tan(hfov / 2.), 0., 0.],
            [0., 0.,  1, 0],
            [0., 0., 0, 1]])
        self.pix_T_camX[0,2] = self.W/2.
        self.pix_T_camX[1,2] = self.H/2.

        self.navigate_obj_info = {}
        self.navigate_obj_info["object_class"] = None
        self.navigate_obj_info["object_center"] = None
        self.navigate_obj_info["obj_ID"] = None
        
        print("Initializing episode...")
        self.init_success = self.load_episode()

        self.task = TidyTask(
            self.env,
            action_to_mappedaction=self.action_to_mappedaction,
            approx_last_action_success=not args.use_gt_success_checker,
            max_fails=args.max_api_fails,
            max_steps=args.max_traj_steps,
            remove_unusable_slice=args.remove_unusable_slice,
            use_GT_error_feedback=self.use_GT_error_feedback,
        )
        self.task.metrics = self.metrics

        if not self.init_success:
            print("task initialization failed.. moving to next episode..")
            return 
        self.controller = self.env.controller
        self.tag = self.env.get_episode_name()
        print("DONE.")

        keep_head_straight = False
        search_pitch_explore = False
        block_map_positions_if_fail=True
        if args.use_estimated_depth or args.increased_explore:
            look_down_init = True
            keep_head_down = False
        else:
            look_down_init = True
            keep_head_down = False
        self.navigation = Navigation(
            controller=self.controller, 
            keep_head_down=keep_head_down, 
            keep_head_straight=keep_head_straight, 
            look_down_init=look_down_init,
            search_pitch_explore=search_pitch_explore, 
            block_map_positions_if_fail=block_map_positions_if_fail,
            pix_T_camX=self.pix_T_camX,
            task=self.task,
            depth_estimation_network=depth_network,
            )
        self.navigation.init_navigation(None)

        self.navigation.bring_head_to_angle(update_obs=False)
        
        origin_T_camX0 = utils.aithor.get_origin_T_camX(self.controller.last_event, False)

        self.object_tracker = ObjectTrack(
            self.name_to_id, 
            self.id_to_name, 
            self.include_classes, 
            self.W, self.H, 
            pix_T_camX=self.pix_T_camX, 
            origin_T_camX0=origin_T_camX0, 
            ddetr=segmentation_network,
            controller=self.controller, 
            use_gt_objecttrack=use_gt_objecttrack,
            do_masks=True,
            id_to_mapped_id=self.id_to_mapped_id,
            navigation=self.navigation,
            use_mask_rcnn_pred=self.use_mask_rcnn_pred,
            use_open_set_segmenter=False,
            name_to_parsed_name=None,
            )
        self.task.object_tracker = self.object_tracker

        print("Starting head tilt:", self.env.controller.last_event.metadata["agent"]["cameraHorizon"])
        
        if args.create_movie:
            self.vis = Animation(
                self.W, self.H, 
                navigation=None if args.remove_map_vis else self.navigation, 
                name_to_id=self.name_to_id
                )
            os.makedirs(args.movie_dir, exist_ok=True)
        else:
            self.vis = None

    def load_episode(self):

        # self.env.start_next_episode()
        self.env.initialize_scene()
        self.metrics = {}

        init_success = True
        return init_success

    def check_api_constraints(self, subgoals, objects, check_constraints):
        '''
        API contraints are per-function constraints to check before executing a subgoal. "pre-conditions" and "post-conditions". They can modify subgoals or call other actions.
        Defined ahead of time. If a new skill is added, this should be modified with the corresponding constraint.
        This assumes that this check is done before the subgoal is called, and the subgoals, and object list is not modified
        TODO: automate this?
        '''

        # object checks
        if len(subgoals)==0 or len(objects)==0:
            pass
        elif "Sliced" in objects[0] and ["Slice", objects[0].replace("Sliced", "")] not in self.attempted_subgoals and objects[0].replace("Sliced", "") not in self.could_not_find:
            # cannot interact with a sliced object without slicing it first
            centroids, labels, IDs = self.object_tracker.get_centroids_and_labels(return_ids=True, object_cat=objects[0], include_holding=True)
            if objects[0] not in labels:
                subgoals_ = ["Slice"]
                objects_ = [objects[0].replace("Sliced", "")]
                subgoals = subgoals_ + subgoals
                objects = objects_ + objects
                check_constraints = [False] + check_constraints
                subgoals, objects, check_constraints = self.check_api_constraints(subgoals, objects, check_constraints)
        elif objects[0] in self.could_not_find:
            centroids, labels, IDs = self.object_tracker.get_centroids_and_labels(return_ids=True, object_cat=objects[0], include_holding=True)
            if len(IDs)==0:
                # couldn't find 
                print("Couldn't find object so skipping subgoal...")
                subgoals.pop(0)
                objects.pop(0)
                check_constraints.pop(0)
                subgoals, objects, check_constraints = self.check_api_constraints(subgoals, objects, check_constraints)

        # subgoal checks
        if len(subgoals)==0 or len(objects)==0:
            pass
        elif subgoals[0]=="Slice":
            # constraints for "slice"
            if subgoals[0]=="Slice" and self.object_tracker.get_label_of_holding()!="Knife" and "Knife" not in self.could_not_find:
                # need knife before slicing vegetables
                subgoals_, objects_ = [], []
                subgoals_.extend(["Navigate", "Pickup", "Navigate"])
                objects_.extend(["Knife", "Knife", objects[0]])
                subgoals = subgoals_ + subgoals
                objects = objects_ + objects
                check_constraints = [False, False, False] + check_constraints
                subgoals, objects, check_constraints = self.check_api_constraints(subgoals, objects, check_constraints)
        elif subgoals[0] in ["Place", "PutDown"]:
            if (subgoals[0] in ["Place"]) and objects[0] not in self.RECEPTACLE_OBJECTS:
                subgoals.pop(0)
                objects.pop(0)
                check_constraints.pop(0)
                subgoals, objects, check_constraints = self.check_api_constraints(subgoals, objects, check_constraints)
            elif self.object_tracker.get_ID_of_holding() is None:
                # nothing to place, move onto next subgoal
                subgoals.pop(0)
                objects.pop(0)
                check_constraints.pop(0)
                subgoals, objects, check_constraints = self.check_api_constraints(subgoals, objects, check_constraints)
            elif (subgoals[0] in ["Place"]) and (objects[0] in self.OPENABLE_OBJECTS) and ("Open" not in [s[0] for s in self.attempted_subgoals[-2:]]):
                '''
                If object is openable, we need to open it before placing it
                '''
                subgoals_ = ["Open"]
                objects_ = [objects[0]]
                subgoals = subgoals_ + subgoals
                objects = objects_ + objects
                check_constraints = [False] + check_constraints
                # subgoals, objects, check_constraints = self.check_api_constraints(subgoals, objects)
        elif subgoals[0]=="Pickup":
            if self.object_tracker.get_label_of_holding()==objects[0] or objects[0] not in self.PICKUPABLE_OBJECTS:
                # if already holding the object, or object not pickupable
                subgoals.pop(0)
                objects.pop(0)
                check_constraints.pop(0)
                subgoals, objects, check_constraints = self.check_api_constraints(subgoals, objects, check_constraints)
            elif self.object_tracker.get_ID_of_holding() is not None:
                subgoals_ = ["PutDown", "Navigate"]
                objects_ = ["PutDown", objects[0]]
                subgoals = subgoals_ + subgoals
                objects = objects_ + objects
                check_constraints = [False, False] + check_constraints
        elif subgoals[0]=="ToggleOn":
            if self.use_GT_constraint_checks:
                # constraint is object must be toggled off to toggle on
                for obj in self.controller.last_event.metadata['objects']:
                    if obj['visible'] and obj['objectType']==objects[0] and obj['isToggled']:
                        # already toggled
                        subgoals.pop(0)
                        objects.pop(0)
                        check_constraints.pop(0)
                        subgoals, objects, check_constraints = self.check_api_constraints(subgoals, objects, check_constraints)
                        break
            else:
                pass
                # assert NotImplementedError
        elif subgoals[0]=="ToggleOff":
            if self.use_GT_constraint_checks:
                # constraint is object must be toggled on to toggle off
                for obj in self.controller.last_event.metadata['objects']:
                    if obj['visible'] and obj['objectType']==objects[0] and not obj['isToggled']:
                        # already toggled
                        subgoals.pop(0)
                        objects.pop(0)
                        check_constraints.pop(0)
                        subgoals, objects, check_constraints = self.check_api_constraints(subgoals, objects, check_constraints)
                        break
            else:
                pass
                # assert NotImplementedError
        elif subgoals[0]=="Open":
            if objects[0] not in self.OPENABLE_OBJECTS:
                subgoals.pop(0)
                objects.pop(0)
                check_constraints.pop(0)
                subgoals, objects, check_constraints = self.check_api_constraints(subgoals, objects, check_constraints)
            elif self.use_GT_constraint_checks:
                # constraint is object must be closed to open
                for obj in self.controller.last_event.metadata['objects']:
                    if obj['visible'] and obj['objectType']==objects[0] and obj['isOpen']:
                        # already toggled
                        subgoals.pop(0)
                        objects.pop(0)
                        check_constraints.pop(0)
                        subgoals, objects, check_constraints = self.check_api_constraints(subgoals, objects, check_constraints)
                        break
                    elif obj['visible'] and obj['objectType']==objects[0] and obj['toggleable'] and obj['isToggled']:
                        subgoals_ = ["ToggleOff"]
                        objects_ = [objects[0]]
                        subgoals = subgoals_ + subgoals
                        objects = objects_ + objects
                        check_constraints = [False] + check_constraints
            else:
                pass
                # assert NotImplementedError
        elif subgoals[0]=="Close":
            if objects[0] not in self.OPENABLE_OBJECTS:
                subgoals.pop(0)
                objects.pop(0)
                check_constraints.pop(0)
                subgoals, objects, check_constraints = self.check_api_constraints(subgoals, objects, check_constraints)
            elif self.use_GT_constraint_checks:
                # constraint is object must be open to close
                for obj in self.controller.last_event.metadata['objects']:
                    if obj['visible'] and obj['objectType']==objects[0] and not obj['isOpen']:
                        # already toggled
                        subgoals.pop(0)
                        objects.pop(0)
                        check_constraints.pop(0)
                        subgoals, objects, check_constraints = self.check_api_constraints(subgoals, objects, check_constraints)
                        break
            else:
                pass
                # assert NotImplementedError
        elif subgoals[0]=="Clean":
            '''
            (1) check if target object is already clean
            (2) pick up object if not already in hand
            (3) add subgoals for cleaning, including changing object state to cleaned
            '''
            subgoals.pop(0)
            object_cat = objects.pop(0)
            check_constraints.pop(0)
            # (1) check if target object is already clean
            centroids, labels, IDs = self.object_tracker.get_centroids_and_labels(return_ids=True, object_cat=object_cat, include_holding=True)
            if (len(IDs)>0 and self.object_tracker.objects_track_dict[IDs[0]]["clean"]):
                pass # object already clean
                subgoals, objects, check_constraints = self.check_api_constraints(subgoals, objects, check_constraints)
            elif len(IDs)==0 and object_cat in self.could_not_find:
                pass # can't find object
                subgoals, objects, check_constraints = self.check_api_constraints(subgoals, objects, check_constraints)
            else:
                subgoals_, objects_ = [], []
                if object_cat in self.name_to_mapped_name_subgoals.keys():
                    object_cat = self.name_to_mapped_name_subgoals[object_cat]
                if self.object_tracker.get_label_of_holding()!=object_cat:
                    # if not holding object, then go pick it up to clean it
                    if self.object_tracker.get_label_of_holding() is None:
                        subgoals_.extend(["Navigate", "Pickup"]) # need to pickup before washing
                        objects_.extend([object_cat, object_cat]) # need to pickup before washing
                        check_constraints = [False, False] + check_constraints
                    else:
                        subgoals_.extend(["PutDown", "Navigate", "Pickup"]) # need to pickup before washing
                        objects_.extend(["PutDown", object_cat, object_cat]) # need to pickup before washing
                        check_constraints = [False, False, False] + check_constraints
                subgoals_.extend(["Navigate", "Place", "ToggleOn", "ToggleOff", "Pickup", "ChangeAttribute"])
                objects_.extend(["Sink", "Sink", "Faucet", "Faucet", object_cat, "clean"])
                if object_cat in self.FILLABLE_CLASSES:
                    subgoals_.append("Pour")
                    objects_.append("Sink")
                    check_constraints = [False] + check_constraints
                subgoals = subgoals_ + subgoals
                objects = objects_ + objects
                check_constraints = [False, False, False, False, False, False] + check_constraints
                subgoals, objects, check_constraints = self.check_api_constraints(subgoals, objects, check_constraints)
        elif subgoals[0]=="Toast":
            '''
            (1) check if target object is already toasted
            (2) pick up object if not already in hand
            (3) add subgoals for toasting, including changing object state to toasting
            '''
            subgoals.pop(0)
            object_cat = objects.pop(0)
            check_constraints.pop(0)
            # (1) check if target object is already toasted
            centroids, labels, IDs = self.object_tracker.get_centroids_and_labels(return_ids=True, object_cat=object_cat, include_holding=True)
            if (len(IDs)>0 and self.object_tracker.objects_track_dict[IDs[0]]["toasted"]):
                pass # object already toasted
                subgoals, objects, check_constraints = self.check_api_constraints(subgoals, objects, check_constraints)
            elif len(IDs)==0 and object_cat in self.could_not_find:
                pass # can't find object
                subgoals, objects, check_constraints = self.check_api_constraints(subgoals, objects, check_constraints)
            else:
                subgoals_, objects_ = [], []
                if object_cat in self.name_to_mapped_name_subgoals.keys():
                    object_cat = self.name_to_mapped_name_subgoals[object_cat]
                if self.object_tracker.get_label_of_holding()!=object_cat:
                    # if not holding object, then go pick it up to toast it
                    if self.object_tracker.get_label_of_holding() is None:
                        subgoals_.extend(["Navigate", "Pickup"]) # need to pickup before toasting
                        objects_.extend([object_cat, object_cat]) # need to pickup before toasting
                        check_constraints = [False, False] + check_constraints
                    else:
                        subgoals_.extend(["PutDown", "Navigate", "Pickup"]) # need to pickup before toasting
                        objects_.extend(["PutDown", object_cat, object_cat]) # need to pickup before toasting
                        check_constraints = [False, False, False] + check_constraints
                subgoals_.extend(["Navigate", "Place", "ToggleOn", "ToggleOff", "Pickup", "ChangeAttribute"])
                objects_.extend(["Toaster", "Toaster", "Toaster", "Toaster", object_cat, "toasted"])
                subgoals = subgoals_ + subgoals
                objects = objects_ + objects
                check_constraints = [False, False, False, False, False, False] + check_constraints
                subgoals, objects, check_constraints = self.check_api_constraints(subgoals, objects, check_constraints)
        elif subgoals[0]=="Cook":
            '''
            (1) check if target object is already cooked
            (2) pick up object if not already in hand
            (3) add subgoals for cooking, including changing object state to cooked
            '''
            subgoals.pop(0)
            object_cat = objects.pop(0)
            check_constraints.pop(0)
            # (1) check if target object is already cooked
            centroids, labels, IDs = self.object_tracker.get_centroids_and_labels(return_ids=True, object_cat=object_cat, include_holding=True)
            if (len(IDs)>0 and self.object_tracker.objects_track_dict[IDs[0]]["cooked"]):
                pass # object already cooked
                subgoals, objects, check_constraints = self.check_api_constraints(subgoals, objects, check_constraints)
            elif len(IDs)==0 and object_cat in self.could_not_find:
                pass # can't find object
                subgoals, objects, check_constraints = self.check_api_constraints(subgoals, objects, check_constraints)
            else:
                subgoals_, objects_ = [], []
                if object_cat in self.name_to_mapped_name_subgoals.keys():
                    object_cat = self.name_to_mapped_name_subgoals[object_cat]
                if self.object_tracker.get_label_of_holding()!=object_cat:
                    # if not holding object, then go pick it up to toast it
                    if self.object_tracker.get_label_of_holding() is None:
                        subgoals_.extend(["Navigate", "Pickup"]) # need to pickup before cooking
                        objects_.extend([object_cat, object_cat]) # need to pickup before cooking
                        check_constraints = [False, False] + check_constraints
                    else:
                        subgoals_.extend(["PutDown", "Navigate", "Pickup"]) # need to pickup before cooking
                        objects_.extend(["PutDown", object_cat, object_cat]) # need to pickup before cooking
                        check_constraints = [False, False, False] + check_constraints
                subgoals_.extend(["Navigate", "Open", "Place", "Close", "ToggleOn", "ToggleOff", "Open", "Pickup", "ChangeAttribute"])
                objects_.extend(["Microwave", "Microwave", "Microwave", "Microwave", "Microwave", "Microwave", "Microwave", object_cat, "cooked"])
                subgoals = subgoals_ + subgoals
                objects = objects_ + objects
                check_constraints = [False, False, False, False, False, False, False, False, False] + check_constraints
                subgoals, objects, check_constraints = self.check_api_constraints(subgoals, objects, check_constraints)
        else:
            # no checks for other subgoals
            pass

        if len(check_constraints)>0:
            # now we've checked constraints for this subgoal
            check_constraints[0] = False

        return subgoals, objects, check_constraints

    def expand_subgoals(self, subgoals, objects, check_constraints):
        '''
        Expand meta-subgoals "clean", "cook", etc. into the low level actions "navigate", "place", etc. 
        '''

        if subgoals[0]=="Clean":
            '''
            (1) check if target object is already clean
            (2) pick up object if not already in hand
            (3) add subgoals for cleaning, including changing object state to cleaned
            '''
            subgoals.pop(0)
            object_cat = objects.pop(0)
            subgoals_, objects_ = [], []
            subgoals_.extend(["Navigate", "Place", "ToggleOn", "ToggleOff", "Pickup", "ChangeAttribute"])
            objects_.extend(["Sink", "Sink", "Faucet", "Faucet", object_cat, "clean"])
            if object_cat in self.FILLABLE_CLASSES:
                subgoals_.append("Pour")
                objects_.append("Sink")
                check_constraints = [False] + check_constraints
            subgoals = subgoals_ + subgoals
            objects = objects_ + objects
            check_constraints = [False, False, False, False, False, False] + check_constraints
        elif subgoals[0]=="Toast":
            '''
            (1) check if target object is already toasted
            (2) pick up object if not already in hand
            (3) add subgoals for toasting, including changing object state to toasting
            '''
            subgoals.pop(0)
            object_cat = objects.pop(0)
            subgoals_, objects_ = [], []
            subgoals_.extend(["Navigate", "Place", "ToggleOn", "ToggleOff", "Pickup", "ChangeAttribute"])
            objects_.extend(["Toaster", "Toaster", "Toaster", "Toaster", object_cat, "toasted"])
            check_constraints = [False, False, False, False, False, False] + check_constraints
            subgoals = subgoals_ + subgoals
            objects = objects_ + objects
        elif subgoals[0]=="Cook":
            '''
            (1) check if target object is already cooked
            (2) pick up object if not already in hand
            (3) add subgoals for cooking, including changing object state to cooked
            '''
            subgoals.pop(0)
            object_cat = objects.pop(0)
            subgoals_, objects_ = [], []
            subgoals_.extend(["Navigate", "Open", "Place", "Close", "ToggleOn", "ToggleOff", "Open", "Pickup", "ChangeAttribute"])
            objects_.extend(["Microwave", "Microwave", "Microwave", "Microwave", "Microwave", "Microwave", "Microwave", object_cat, "cooked"])
            check_constraints = [False, False, False, False, False, False, False, False, False] + check_constraints
            subgoals = subgoals_ + subgoals
            objects = objects_ + objects
        else:
            # no checks for other subgoals
            pass

        return subgoals, objects, check_constraints


    def run_subgoals(self, subgoals: List, objects: List, render: bool = False, max_explore_steps: int = 10, max_attempts: int = 2, run_error_correction=False, completed_subgoals=[]):
        self.just_navigated = False
        succ = 0
        subgoal_attempts = 0
        check_constraints = [True] * len(subgoals) # only check constraints for initial subgoals
        while len(subgoals)>0:
            if self.task.is_done() or self.num_subgoal_fail>=self.max_subgoal_fail:
                break
            obj = None
            if self.use_constraint_check and check_constraints[0]:
                subgoals, objects, check_constraints = self.check_api_constraints(subgoals, objects, check_constraints)
            else:
                subgoals, objects, check_constraints = self.expand_subgoals(subgoals, objects, check_constraints) 

            if len(subgoals)==0:
                break
            subgoal_name = subgoals.pop(0)
            object_name = objects.pop(0)
            check_constraints.pop(0)
            self.future_subgoals = [[s,o] for s,o in zip(subgoals, objects)]
            self.current_subgoal = [subgoal_name, object_name]
            self.attempted_subgoals.append(self.current_subgoal)
            if object_name in self.name_to_mapped_name_subgoals.keys():
                print(f"Mapping {object_name} to {self.name_to_mapped_name_subgoals[object_name]}")
                object_name = self.name_to_mapped_name_subgoals[object_name]

            if subgoal_name == "Navigate":
                if self.teleport_to_objs:
                    success = self.teleport_to_object(object_name)
                else:
                    success = self.navigate(object_name)
                if not success:
                    msg = "Can no longer navigate to " + object_name +  ", terminating!"
                    print(msg)
                    self.num_subgoal_fail += 1
                self.just_navigated = True
            elif subgoal_name == "ObjectDone":
                pass # this should be handled in next else statement
            elif subgoal_name == "ChangeAttribute":
                if self.navigate_obj_info["obj_ID"] in self.object_tracker.objects_track_dict.keys():
                    self.object_tracker.objects_track_dict[self.navigate_obj_info["obj_ID"]][object_name] = True
            elif object_name=="Agent":
                success = self.run_corrective_action(subgoal_name)
                self.just_navigated = True
            else:
                if self.just_navigated and not success:
                    print("Navigation failed!")
                    self.failed_subgoals.append([subgoal_name, object_name])
                    continue
                object_done_holding = False
                if subgoal_name in ["Place", "PutDown"]:
                    if len(subgoals)>0 and subgoals[0]=="ObjectDone":
                        subgoals.pop(0)
                        objects.pop(0)
                        check_constraints.pop(0)
                        object_done_holding = True # object should not be interacted with again
                retry_image = True if (subgoal_name in self.RETRY_ACTIONS_IMAGE and object_name in self.RETRY_DICT_IMAGE[subgoal_name]) else False
                retry_location = True if (subgoal_name in self.RETRY_ACTIONS_LOCATION and object_name in self.RETRY_DICT_LOCATION[subgoal_name] and self.run_error_correction_llm) else False
                success, error = self.execute_action(
                    subgoal_name, 
                    object_name, 
                    object_done_holding=object_done_holding, 
                    retry_image=retry_image, 
                    retry_location=retry_location
                    )   

                if not success:
                    '''
                    Error correction (Rectifier)
                    '''
                    self.num_subgoal_fail += 1
                    if self.task.num_fails<int((2/3)*self.task.max_fails):
                        if run_error_correction:
                            if self.run_error_correction_llm:
                                if subgoal_attempts==max_attempts:
                                    subgoal_attempts = 0
                                elif subgoal_attempts==0:
                                    subgoals_, objects_, self.search_dict_ = self.run_llm_replan()
                                    search_dict_prev = self.search_dict
                                    self.search_dict = self.search_dict_
                                    success = self.run_subgoals(subgoals_, objects_, run_error_correction=False, completed_subgoals=self.completed_subgoals)
                                    if success:
                                        self.error_correct_fix.append([subgoal_name, object_name])
                                    self.search_dict = search_dict_prev
                                else:
                                    # error correction: retry with a different object instance
                                    subgoals.insert(0, subgoal_name)
                                    subgoals.insert(0, "Navigate")
                                    objects.insert(0, object_name)
                                    objects.insert(0, object_name)
                                    check_constraints = [False, False] + check_constraints
                            elif self.run_error_correction_basic:
                                if subgoal_attempts==max_attempts:
                                    subgoal_attempts = 0
                                else:
                                    # retry with a different object instance
                                    subgoals.insert(0, subgoal_name)
                                    subgoals.insert(0, "Navigate")
                                    objects.insert(0, object_name)
                                    objects.insert(0, object_name)
                                    check_constraints = [False, False] + check_constraints
                            else:
                                pass # no error correction

                                
                            subgoal_attempts += 1
                            
                    msg = "Can no longer perform " + subgoal_name + " on " + object_name + " terminating!"
                    print(msg)                      
                    
                self.just_navigated = False

            if success:
                self.successful_subgoals.append([subgoal_name, object_name])
                succ += 1
                msg = subgoal_name + ":" + object_name + " was successful"
                print(msg)
                logging.info(msg)
                completed_subgoals.append([subgoal_name, object_name])
                if self.vis is not None and self.render:
                    for _ in range(5):
                        rgb = np.float32(self.get_image(self.controller))
                        self.vis.add_frame(rgb, text=f"SUBGOAL {subgoal_name} SUCCESS!", add_map=self.add_map)
            else:
                self.failed_subgoals.append([subgoal_name, object_name])
                if self.vis is not None and self.render:
                    for _ in range(5):
                        rgb = np.float32(self.get_image(self.controller))
                        self.vis.add_frame(rgb, text=f"SUBGOAL {subgoal_name} FAILED.", add_map=self.add_map)
        return succ

    def eval(self, additional_tag=''):

        metrics = self.task.get_metrics()

        self.task.metrics.update(metrics)

        tbl = wandb.Table(columns=list(self.task.metrics.keys()))
        tbl.add_data(*list(self.task.metrics.values()))
        wandb.log({f"Metrics{additional_tag}/{self.tag}": tbl})

        print("Eval:")
        print(metrics)
        # print(f"Task success: {success}")
        # print(f"Final subgoal success: {float(goal_condition_success_rate)}")

    def run_episode(self, user_progress_check=True):
        if self.episode_in_try_except:
            try:
                self.search_dict = {}
                camX0_T_camXs = self.map_and_explore()
                print("RUNNING IN TRY EXCEPT")
                if self.use_gt_subgoals:
                    raise NotImplementedError
                else:
                    subgoals, objects, self.search_dict = self.run_llm()
                    print("SUBGOALS:", subgoals)
                    print("ARGUMENTS:", objects)
                # goal_instr = self.traj_data['turk_annotations']['anns'][self.traj_data['repeat_idx']]['task_desc']
                # print(f"High level goal: {goal_instr}")
                total_tasks = len(subgoals)
                succ = self.run_subgoals(subgoals, objects, run_error_correction=True, completed_subgoals=self.completed_subgoals)
            except KeyboardInterrupt:
                sys.exit(0)
            except Exception as e:
                tbl = wandb.Table(columns=["Error", "Traceback"])
                tbl.add_data(str(e), str(traceback.format_exc()))
                wandb.log({f"Errors/{self.tag}": tbl})
                print(e)
                print(traceback.format_exc())
        else:
            self.search_dict = {}
            camX0_T_camXs = self.map_and_explore()
            # self.random_search(None, max_steps=200)
            if self.use_gt_subgoals:
                assert NotImplementedError
            else:
                subgoals, objects, self.search_dict = self.run_llm()
                print("SUBGOALS:", subgoals)
                print("ARGUMENTS:", objects)
            succ = self.run_subgoals(subgoals, objects, run_error_correction=True, completed_subgoals=self.completed_subgoals)

    def run_examples(self):
        self.search_dict = {}
        camX0_T_camXs = self.map_and_explore()
        self.random_search(None, max_steps=100)
        self.save_llm_outofplace_examples()

    def run(self):

        if args.mode in ["tidy_eval"]:
            self.run_episode(user_progress_check=self.use_progress_check)
        elif args.mode in ["tidy_examples"]:
            self.run_examples()
            return self.task.metrics, self.env
        else:
            raise NotImplementedError

        if self.controller is not None:
            # self.controller.stop()
            self.eval()
        self.render_output()

        return self.task.metrics, self.env

    def render_output(self):
        if self.render:
            print("Rendering!")
            if args.save_movie:
                self.vis.render_movie(args.movie_dir, self.tag, tag=f"Full")
            frames_numpy = np.asarray(self.vis.image_plots)
            frames_numpy = np.transpose(frames_numpy, (0,3,1,2))
            wandb.log({f"movies/{self.tag}": wandb.Video(frames_numpy, fps=10, format="mp4")})

    # Adds Navigate subgoal between two consecutive Object Interaction Subgoals
    def add_navigation_goals(self, subgoals, objects):
        final_subgoals = copy.deepcopy(subgoals)
        final_objects = copy.deepcopy(objects)
        obj_interaction = False
        idx_add = 0 
        for i in range(len(subgoals)):
            if subgoals[i]!="Navigate" and subgoals[i-1]!="Navigate":
                final_subgoals.insert((i-1)+idx_add, "Navigate")
                final_objects.insert((i-1)+idx_add, objects[i])
                idx_add += 1
        return final_subgoals, final_objects

    def get_image(self, controller=None):
        if controller is not None:
            rgb = controller.last_event.frame
        else:
            raise NotImplementedError
        return rgb

def run_tidy():
    save_metrics = True
    split_ = args.split

    if args.start_startx:
        assert(False) # deprecated, use startx.py instead
        server_port = startx()
    else:
        server_port = args.server_port
    print("SERVER PORT=", server_port)
    controller = Controller(
            # scene=mapname, 
            visibilityDistance=args.visibilityDistance,
            gridSize=args.STEP_SIZE,
            width=args.W,
            height=args.H,
            fieldOfView=args.fov,
            renderObjectImage=True,
            renderDepthImage=True,
            renderInstanceSegmentation=True,
            x_display=str(server_port),
            snapToGrid=False,
            rotateStepDegrees=args.DT,
            )

    tidy_task = TIDEE_TASK(controller, split_)

    # initialize wandb
    if args.set_name=="test00":
        wandb.init(mode="disabled")
    else:
        wandb.init(project="HELPER-tidy", name=args.set_name, group=args.group, config=args, dir=args.wandb_directory)

    metrics = {}
    metrics_file = os.path.join(args.metrics_dir, f'{args.mode}_metrics_{split_}.txt')
    if os.path.exists(metrics_file) and args.skip_if_exists:
        with open(metrics_file, 'r') as fp:
            metrics = json.load(fp)
        # metrics = load_json(metrics_file)
    iter_ = 0
    env = None
    # depth_estimation_network = None
    segmentation_network = None
    for episode_idx in range(tidy_task.num_episodes_total):
        tidy_task.set_next_episode_indices()
        episode_name = tidy_task.get_episode_name()
        file_tag = episode_name
        print("Running ", file_tag)
        print(f"Iteration {episode_idx+1}/{tidy_task.num_episodes_total}")
        if args.skip_if_exists and (file_tag in metrics.keys()):
            print(f"File already in metrics... skipping...")
            iter_ += 1
            continue
        subgoalcontroller = SubGoalController(
                iteration=iter_, 
                env=tidy_task, 
                segmentation_network=segmentation_network
                )
        if subgoalcontroller.init_success:
            metrics_instance, tidy_task = subgoalcontroller.run()
            if segmentation_network is None:
                segmentation_network = subgoalcontroller.object_tracker.ddetr
            # if depth_estimation_network is None:
            #     depth_estimation_network = subgoalcontroller.navigation.depth_estimator
        else:
            metrics_instance, tidy_task = subgoalcontroller.task.metrics, subgoalcontroller.env
        metrics[file_tag] = metrics_instance

        iter_ += 1

        aggregrated_metrics = tidy_task.aggregate_metrics(metrics)

        print('\n\n---------- File 1 ---------------')
        to_log = []  
        to_log.append('-'*40 + '-'*40)
        to_log.append(f'Split: {args.eval_split}')
        to_log.append(f'Number of files: {len(list(metrics.keys()))}')
        for f_n in aggregrated_metrics.keys(): #keys_include:
            to_log.append(f'{f_n}: {aggregrated_metrics[f_n]}') 
        to_log.append('-'*40 + '-'*40)

        os.makedirs(args.metrics_dir, exist_ok=True)
        path = os.path.join(args.metrics_dir, f'tidy_task_summary_{args.eval_split}.txt')
        with open(path, "w") as fobj:
            for x in to_log:
                fobj.write(x + "\n")

        save_dict_as_json(metrics, metrics_file)

        aggregrated_metrics["num episodes"] = iter_
        tbl = wandb.Table(columns=list(aggregrated_metrics.keys()))
        tbl.add_data(*list(aggregrated_metrics.values()))
        wandb.log({f"Metrics_summary/Summary": tbl, 'step':iter_})                

        cols = ["file"]+list(metrics_instance.keys())
        tbl = wandb.Table(columns=cols)
        for f_k in metrics.keys():
            to_add_tbl = [f_k]
            for k in list(metrics[f_k].keys()):
                if k=="pred_actions":
                    continue
                to_add_tbl.append(metrics[f_k][k])
            tbl.add_data(*to_add_tbl)
        wandb.log({f"Metrics_summary/Metrics": tbl, 'step':iter_})
