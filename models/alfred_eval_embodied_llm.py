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
import glob

from utils.misc import save_dict_as_json

sys.path.append('./alfred')
sys.path.append('./alfred/gen')
from alfred.gen.utils.py_util import walklevel
from alfred.env.thor_env import ThorEnv

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

from utils.misc import load_json, save_json

from .agent.executor import ExecuteController
from .agent.planner import PlannerController

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

torch.manual_seed(args.seed)
np.random.seed(args.seed)

class SubGoalController(ExecuteController, PlannerController):
    def __init__(
        self, 
        json_file: str = None, 
        max_init_tries: int =5, 
        replay_timeout: int = 500, 
        num_processes: int = 1, 
        iteration=0,
        env=None,
        depth_network=None,
        segmentation_network=None,
        visual_mem=None,
        ) -> None:

        super(SubGoalController, self).__init__()

        self.env = env

        split = args.split

        ###########%%%%%% PARAMS %%%%%%%#########

        self.log_every = args.log_every # log every X iters if generating video
        # self.eval_rows_to_add = eval_rows_to_add

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
        self.llm_log = {"Dialogue":"", "LLM output":"", "subgoals":"", "full_prompt":""}
        
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
        self.PICKUPABLE_OBJECTS += self.special_classes

        self.general_receptacles_classes = [
                'CounterTop', 'DiningTable', 'CoffeeTable', 'SideTable',
                'Desk', 'Bed', 'Sofa', 'ArmChair', 'ShelvingUnit',
                'Drawer', 'Chair', 'Shelf', 'Dresser', 'Ottoman', 'DogBed', 
                'Footstool', 'Safe', 'TVStand'
                ]
        self.clean_classes = [] #["Bowl", "Cup", "Mug", "Plate"]

        if not self.use_gt_subgoals:
            self.llm = LLMPlanner(
                args.gpt_embedding_dir, 
                fillable_classes=self.FILLABLE_CLASSES, 
                openable_classes=self.OPENABLE_CLASS_LIST,
                include_classes=self.include_classes,
                clean_classes=self.clean_classes,
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
            self.name_to_mapped_name = {'DeskLamp':'FloorLamp', 'EggCracked':'Egg'} # 'Sink':'SinkBasin', 'Bathtub':'BathtubBasin', 
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
        self.init_success = self.load_episode(json_file)
        
        self.task = AlfredTask(
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
        self.controller = self.env
        self.tag = f"{self.traj_data['task_id']}_repeat{self.traj_data['repeat_idx']}"
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

        self.navigation.bring_head_to_angle(update_obs=False)

        self.navigation.init_navigation(None)
        
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
        
        if args.create_movie:
            self.vis = Animation(
                self.W, self.H, 
                navigation=None if args.remove_map_vis else self.navigation, 
                name_to_id=self.name_to_id
                )
            os.makedirs(args.movie_dir, exist_ok=True)
        else:
            self.vis = None

        if args.use_visual_mem_holding:
            
            if visual_mem is None:
                from task_base.attribute_detector import AttributeDetectorVisualMem
                self.visual_mem = AttributeDetectorVisualMem(self.W, self.H)
            else:
                self.visual_mem = visual_mem

    def load_episode(self, json_file: str):

        self.repeat_idx = json_file['repeat_idx']
        self.task_id = json_file['task']

        json_file_path = os.path.join(args.alfred_data_dir, "json_2.1.0", args.split, self.task_id, "traj_data.json")
        
        # load json data
        with open(json_file_path) as f:
            traj_data = json.load(f)

        # fresh images list
        traj_data['images'] = list()

        # scene setup
        scene_num = traj_data['scene']['scene_num']
        object_poses = traj_data['scene']['object_poses']
        object_toggles = traj_data['scene']['object_toggles']
        dirty_and_empty = traj_data['scene']['dirty_and_empty']

        if self.env is None:
            if args.start_startx:
                from ai2thor_docker.ai2thor_docker.x_server import startx
                self.server_port = startx()
                args.server_port = self.server_port
                self.server_port = str(self.server_port)
            else:
                self.server_port = str(args.server_port)
            print(f"X display port: {self.server_port}")
            self.env = ThorEnv(player_screen_width=self.W,
                            player_screen_height=self.H,
                            x_display=str(self.server_port),
                            )

        # reset
        scene_name = 'FloorPlan%d' % scene_num
        self.env.reset(scene_name)
        self.env.restore_scene(object_poses, object_toggles, dirty_and_empty)

        self.env.step(dict(traj_data['scene']['init_action']))

        if "test" not in args.split:
            # setup task
            self.env.set_task(traj_data, args, reward_type='dense')

        self.metrics = {}
        traj_data['repeat_idx'] = self.repeat_idx

        self.traj_data = traj_data

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
            elif (subgoals[0] in ["Place"]) and (objects[0] in self.OPENABLE_OBJECTS) and (["Open", objects[0]] not in self.attempted_subgoals):
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
                for obj in self.task.env.last_event.metadata['objects']:
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
                for obj in self.task.env.last_event.metadata['objects']:
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
                for obj in self.task.env.last_event.metadata['objects']:
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
                for obj in self.task.env.last_event.metadata['objects']:
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
                        if args.use_visual_mem_holding:
                            subgoals = subgoals_ + ["Clean"] + subgoals
                            objects = objects_ + [object_cat] + objects
                            check_constraints = [True] + check_constraints
                            check_constraints[:3] = [True, True, True] # check in case skip
                            return subgoals, objects, check_constraints
                    else:
                        subgoals_.extend(["PutDown", "Navigate", "Pickup"]) # need to pickup before washing
                        objects_.extend(["PutDown", object_cat, object_cat]) # need to pickup before washing
                        check_constraints = [False, False, False] + check_constraints
                        if args.use_visual_mem_holding:
                            subgoals = subgoals_ + ["Clean"] + subgoals
                            objects = objects_ + [object_cat] + objects
                            check_constraints = [True] + check_constraints
                            check_constraints[:4] = [True, True, True, True] # check in case skip
                            return subgoals, objects, check_constraints
                is_clean = False
                if args.use_visual_mem_holding:
                    is_clean = self.check_attribute_object(object_cat, "clean")
                    if is_clean:
                        # skip cleaning!
                        subgoals, objects, check_constraints = self.check_api_constraints(subgoals, objects, check_constraints)
                if not is_clean:
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
                        if args.use_visual_mem_holding:
                            subgoals = subgoals_ + ["Cook"] + subgoals
                            objects = objects_ + [object_cat] + objects
                            check_constraints = [True] + check_constraints
                            check_constraints[:3] = [True, True, True] # check in case skip
                            return subgoals, objects, check_constraints
                    else:
                        subgoals_.extend(["PutDown", "Navigate", "Pickup"]) # need to pickup before cooking
                        objects_.extend(["PutDown", object_cat, object_cat]) # need to pickup before cooking
                        check_constraints = [False, False, False] + check_constraints
                        if args.use_visual_mem_holding:
                            subgoals = subgoals_ + ["Cook"] + subgoals
                            objects = objects_ + [object_cat] + objects
                            check_constraints = [True] + check_constraints
                            check_constraints[:4] = [True, True, True, True] # check in case skip
                            return subgoals, objects, check_constraints
                is_cooked = False
                if args.use_visual_mem_holding:
                    is_cooked = self.check_attribute_object(object_cat, "clean")
                    if is_cooked:
                        # skip cleaning!
                        subgoals, objects, check_constraints = self.check_api_constraints(subgoals, objects, check_constraints)
                if not is_cooked:
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

    def check_attribute_object(self, object_cat, attribute):

        self.navigation.bring_head_to_angle(update_obs=True, angle=30)
        # plt.figure()
        # plt.imshow(self.controller.last_event.frame)
        # plt.savefig('output/test.png')

        rgb = self.controller.last_event.frame

        from utils.ddetr_utils import check_for_detections

        with torch.no_grad():
            out = check_for_detections(
                rgb, self.object_tracker.ddetr, self.W, self.H, 
                self.object_tracker.score_labels_name, self.object_tracker.score_boxes_name, 
                score_threshold_ddetr=0., do_nms=False, target_object=None, target_object_score_threshold=0.,
                solq=self.object_tracker.use_solq, return_masks=self.object_tracker.do_masks, 
                nms_threshold=self.object_tracker.nms_threshold, id_to_mapped_id=self.object_tracker.id_to_mapped_id, 
                return_features=False,
                only_keep_target=True,
                )
        pred_labels = out["pred_labels"]
        pred_scores = out["pred_scores"]
        pred_boxes_or_masks = out["pred_masks"] 

        x_check = int(self.W//2) #0.43125
        y_check = int(0.667 * self.H) #0.59792

        # best_conf = 0
        # for mask_i in range(len(pred_boxes_or_masks)):
        #     if self.id_to_name[pred_labels[mask_i]] not in self.PICKUPABLE_OBJECTS:
        #         continue
        #     mask = pred_boxes_or_masks[mask_i]
        #     masks_x, masks_y = np.where(mask)
        #     conf_cur = pred_scores[mask_i]
        #     if x_check in list(masks_x) and y_check in list(masks_y) and conf_cur>best_conf:
        #          best_conf = conf_cur
        #          mask_pickup = mask

        best_conf = 0
        for mask_i in range(len(pred_boxes_or_masks)):
            if self.id_to_name[pred_labels[mask_i]] not in self.PICKUPABLE_OBJECTS:
                # print(self.id_to_name[pred_labels[mask_i]])
                continue
            mask = pred_boxes_or_masks[mask_i]
            masks_x, masks_y = np.where(mask)
            conf_cur = pred_scores[mask_i]
            # print(self.id_to_name[pred_labels[mask_i]])
            if y_check in list(masks_x) and x_check in list(masks_y) and conf_cur>best_conf:
                best_conf = conf_cur
                mask_pickup = mask
                label_pickup = self.id_to_name[pred_labels[mask_i]]
            # else:
            #     print(masks_x)
            #     print(masks_y)
            #     print(self.id_to_name[pred_labels[mask_i]], y_check in list(masks_x), x_check in list(masks_y))
        
        # plt.figure()
        # plt.imshow(mask_pickup)
        # plt.savefig(f'output/test.png')
        # st()

        if best_conf==0:
            print("Found no object.. returning default attribute")
            return self.object_tracker.attributes[attribute]

        attribute_val = self.visual_mem.check_attribute(rgb, mask_pickup, object_cat, attribute)

        if attribute_val is None:
            print("attribute detector gave None.. returning default attribute")
            return self.object_tracker.attributes[attribute]

        print(f"Check for attribute {object_cat} for {attribute}: {attribute_val}")

        return attribute_val

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
            if "test" not in args.split:
                gc_sat, gc_tot = self.get_goal_condition_success()
                print(f"Goal condition success: {gc_sat} / {gc_tot}")
            print(f"Subgoals remaining: {[[s, o] for s, o in zip(subgoals, objects)]}")
            print(f"Subgoal: {subgoal_name}")
            print(f"Object: {object_name}")
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
                retry_location = True if (subgoal_name in self.RETRY_ACTIONS_LOCATION and object_name in self.RETRY_DICT_LOCATION[subgoal_name]) else False
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
                    if self.task.num_fails<int((4/5)*self.task.max_fails):
                        if run_error_correction:
                            if self.run_error_correction_llm:
                                if subgoal_attempts==max_attempts:
                                    subgoal_attempts = 0
                                elif subgoal_attempts==0:
                                    subgoals_, objects_, self.search_dict_ = self.run_llm_replan(self.traj_data)
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

    def progress_check(self):
        raise NotImplementedError

    def get_goal_condition_success(self):
        pcs = self.env.get_goal_conditions_met()
        return pcs[0], float(pcs[1])

    def eval(self, additional_tag=''):

        if "test" in args.split:
            # actseq = {self.traj_data['task_id']: actions}
            goal_instr = self.traj_data['turk_annotations']['anns'][self.traj_data['repeat_idx']]['task_desc']
            log_entry = {'trial': self.traj_data['task_id'],
                        # 'type': self.traj_data['task_type'],
                        'repeat_idx': int(self.repeat_idx),
                        'goal_instr': goal_instr,
                        'actions': self.task.actions,
                        }

            self.task.metrics.update(log_entry)

            subgoal_log = {
                "failed_subgoals":str(self.failed_subgoals), 
                "success_subgoals":str(self.successful_subgoals), 
                "error_correct_subgoals":str(self.error_correct_fix), 
                "attempted_subgoals":str(self.attempted_subgoals)
                }
            self.task.metrics.update(subgoal_log)
            self.task.metrics.update(self.llm_log)
            tbl = wandb.Table(columns=list(self.task.metrics.keys()))
            tbl.add_data(*list(self.task.metrics.values()))
            wandb.log({f"Metrics{additional_tag}/{self.tag}": tbl})

            print("Eval TEST:")
            print(additional_tag)
            return

        # check if goal was satisfied
        goal_satisfied = self.env.get_goal_satisfied()
        if goal_satisfied:
            print("Goal Reached")
            success = True
        else:
            success = False

        t = self.task.steps
        # reward = 0

        # goal_conditions
        pcs = self.env.get_goal_conditions_met()
        goal_condition_success_rate = pcs[0] / float(pcs[1])

        # SPL
        path_len_weight = len(self.traj_data['plan']['low_actions'])
        s_spl = (1 if goal_satisfied else 0) * min(1., path_len_weight / float(t))
        pc_spl = goal_condition_success_rate * min(1., path_len_weight / float(t))

        # path length weighted SPL
        plw_s_spl = s_spl * path_len_weight
        plw_pc_spl = pc_spl * path_len_weight

        # log success/fails
        # lock.acquire()
        goal_instr = self.traj_data['turk_annotations']['anns'][self.traj_data['repeat_idx']]['task_desc']
        successful_steps, failed_steps = self.env.task.goal_subgoals_met(self.controller.last_event)
        log_entry = {'trial': self.traj_data['task_id'],
                     'type': self.traj_data['task_type'],
                     'repeat_idx': int(self.repeat_idx),
                     'goal_instr': goal_instr,
                     'completed_goal_conditions': int(pcs[0]),
                     'total_goal_conditions': int(pcs[1]),
                     'goal_condition_success': float(goal_condition_success_rate),
                     'success_spl': float(s_spl),
                     'path_len_weighted_success_spl': float(plw_s_spl),
                     'goal_condition_spl': float(pc_spl),
                     'path_len_weighted_goal_condition_spl': float(plw_pc_spl),
                     'path_len_weight': int(path_len_weight),
                     'success': success,
                     'successful steps':successful_steps,
                     'failed steps':failed_steps,
                     }

        self.task.metrics.update(log_entry)

        subgoal_log = {
            "failed_subgoals":str(self.failed_subgoals), 
            "success_subgoals":str(self.successful_subgoals), 
            "error_correct_subgoals":str(self.error_correct_fix), 
            "attempted_subgoals":str(self.attempted_subgoals)
            }
        self.task.metrics.update(subgoal_log)
        self.task.metrics.update(self.llm_log)
        tbl = wandb.Table(columns=list(self.task.metrics.keys()))
        tbl.add_data(*list(self.task.metrics.values()))
        wandb.log({f"Metrics{additional_tag}/{self.tag}": tbl})

        print("Eval:")
        print(additional_tag)
        print(f"Task success: {success}")
        print(f"Final subgoal success: {goal_condition_success_rate}")

    def run_episode(self, user_progress_check=True):
        if self.episode_in_try_except:
            try:
                self.search_dict = {}
                if not self.teleport_to_objs:
                    camX0_T_camXs = self.map_and_explore()
                print("RUNNING IN TRY EXCEPT")
                if self.use_gt_subgoals:
                    raise NotImplementedError
                else:
                    subgoals, objects, self.search_dict = self.run_llm(self.traj_data)
                    print("SUBGOALS:", subgoals)
                    print("ARGUMENTS:", objects)
                goal_instr = self.traj_data['turk_annotations']['anns'][self.traj_data['repeat_idx']]['task_desc']
                print(f"High level goal: {goal_instr}")
                total_tasks = len(subgoals)
                succ = self.run_subgoals(subgoals, objects, run_error_correction=True, completed_subgoals=self.completed_subgoals)
                if user_progress_check:
                    raise NotImplementedError

            except KeyboardInterrupt:
                # self.render_output()
                sys.exit(0)
                # pass
            except Exception as e:
                tbl = wandb.Table(columns=["Error", "Traceback"])
                tbl.add_data(str(e), str(traceback.format_exc()))
                wandb.log({f"Errors/{self.tag}": tbl})
                print(e)
                print(traceback.format_exc())
        else:
            self.search_dict = {}
            if not self.teleport_to_objs:
                camX0_T_camXs = self.map_and_explore()
            if self.use_gt_subgoals:
                raise NotImplementedError
            else:
                subgoals, objects, self.search_dict = self.run_llm(self.traj_data)
                print("SUBGOALS:", subgoals)
                print("ARGUMENTS:", objects)
            goal_instr = self.traj_data['turk_annotations']['anns'][self.traj_data['repeat_idx']]['task_desc']
            print(f"High level goal: {goal_instr}")
            succ = self.run_subgoals(subgoals, objects, run_error_correction=True, completed_subgoals=self.completed_subgoals)
            if user_progress_check:
                raise NotImplementedError

    def run(self):
        if args.mode in ["alfred_eval"]:
            self.run_episode(user_progress_check=self.use_progress_check)

        if self.controller is not None:
            self.eval()
        self.render_output()

        return self.task.metrics, self.env

    def render_output(self):
        if self.render:
            print("Rendering!")
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

def run_alfred():
    save_metrics = True
    split_ = args.split
    data_dir = args.alfred_data_dir 

    split_file = os.path.join(data_dir, "splits", "oct21.json")
    with open(split_file) as f:
        split_data = json.load(f)
    files = split_data[split_]

    if args.sample_every_other:
        files = files[::2]

    if args.max_episodes is not None:
        files = files[:args.max_episodes]

    if args.start_episode_index is not None:
        files = files[args.start_episode_index:]

    if args.task_type is not None:
        files_ = []
        for file in files:
            if args.task_type in file['task']:
                files_.append(file)
        files = files_

    if args.episode_file is not None:
        repeat_idx = int(args.episode_file.split('_repeat')[-1])
        trial_name = args.episode_file.split('_repeat')[0] #'trial_T20190907_171933_349922'
        for file in files:
            if trial_name in file['task'] and file['repeat_idx']==repeat_idx:
                files = [file]
                break

    # initialize wandb
    if args.set_name=="test00":
        wandb.init(mode="disabled")
    else:
        wandb.init(project="HELPER-alfred", name=args.set_name, group=args.group, config=args, dir=args.wandb_directory)

    metrics = {}
    metrics_file = os.path.join(args.metrics_dir, f'{args.mode}_metrics_{split_}.txt')
    if os.path.exists(metrics_file) and args.skip_if_exists:
        metrics = load_json(metrics_file)
    if args.use_progress_check:
        metrics_before_feedback = {}
        metrics_before_feedback2 = {}
        metrics_file_before_feedback = os.path.join(args.metrics_dir, f'{args.mode}_metrics_before_feedback_{split_}.txt')
        metrics_file_before_feedback2 = os.path.join(args.metrics_dir, f'{args.mode}_metrics_before_feedback2_{split_}.txt')
        if os.path.exists(metrics_file_before_feedback) and args.skip_if_exists:
            metrics_before_feedback = load_json(metrics_file_before_feedback)
        if os.path.exists(metrics_file_before_feedback2) and args.skip_if_exists and args.use_progress_check:
            metrics_before_feedback2 = load_json(metrics_file_before_feedback2)
    iter_ = 0
    env = None
    depth_estimation_network = None
    segmentation_network = None
    visual_mem = None
    for file in files:
        file_tag = f"{file['task']}_repeat{file['repeat_idx']}"
        print("Running ", file_tag)
        print(f"Iteration {iter_+1}/{len(files)}")
        if args.skip_if_exists and (file_tag in metrics.keys()):
            print(f"File already in metrics... skipping...")
            iter_ += 1
            continue
        task_instance = file #os.path.join(instance_dir, file)
        subgoalcontroller = SubGoalController(
                task_instance, 
                iteration=iter_, 
                env=env, 
                depth_network=depth_estimation_network, 
                segmentation_network=segmentation_network,
                visual_mem=visual_mem,
                )
        if subgoalcontroller.init_success:
            metrics_instance, env = subgoalcontroller.run()
            if segmentation_network is None:
                segmentation_network = subgoalcontroller.object_tracker.ddetr
            if depth_estimation_network is None:
                depth_estimation_network = subgoalcontroller.navigation.depth_estimator
            if visual_mem is None and args.use_visual_mem_holding:
                visual_mem = subgoalcontroller.visual_mem
        else:
            metrics_instance, env = subgoalcontroller.task.metrics, subgoalcontroller.env
        metrics[file_tag] = metrics_instance
        if args.use_progress_check:
            metrics_instance_before_feedback = subgoalcontroller.metrics_before_feedback
            metrics_before_feedback[file_tag] = metrics_instance_before_feedback
            metrics_instance_before_feedback2 = subgoalcontroller.metrics_before_feedback2
            metrics_before_feedback2[file_tag] = metrics_instance_before_feedback2

        iter_ += 1

        if save_metrics and "test" in args.split:
            os.makedirs(args.metrics_dir, exist_ok=True)
            save_dict_as_json(metrics, metrics_file)             

            cols = ["file"]+list(metrics_instance.keys())
            cols.remove('actions')
            tbl = wandb.Table(columns=cols)
            for f_k in metrics.keys():
                to_add_tbl = [f_k]
                for k in list(metrics[f_k].keys()):
                    if k in ["pred_actions", "actions"]:
                        continue
                    to_add_tbl.append(metrics[f_k][k])
                tbl.add_data(*to_add_tbl)
            wandb.log({f"Metrics_summary/Metrics": tbl, 'step':iter_})
        else:

            successes = []
            failures = []
            for k in metrics.keys():
                if metrics[k]["success"]:
                    successes.append(metrics[k])
                else:
                    failures.append(metrics[k])
            aggregrated_metrics = get_metrics(successes, failures)

            print('\n\n---------- File 1 ---------------')
            to_log = []  
            to_log.append('-'*40 + '-'*40)
            list_of_files = files #list(metrics.keys())
            # to_log.append(f'Files: {str(list_of_files)}')
            to_log.append(f'Split: {split_}')
            to_log.append(f'Number of files: {len(list(metrics.keys()))}')
            for f_n in aggregrated_metrics.keys(): #keys_include:
                to_log.append(f'{f_n}: {aggregrated_metrics[f_n]}') 
            to_log.append('-'*40 + '-'*40)

            os.makedirs(args.metrics_dir, exist_ok=True)
            path = os.path.join(args.metrics_dir, f'{args.mode}_summary_{split_}.txt')
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

        if args.use_progress_check:
            raise NotImplementedError

# @classmethod
def get_metrics(successes, failures):
    '''
    compute overall succcess and goal_condition success rates along with path-weighted metrics
    '''
    # stats
    num_successes, num_failures = len(successes), len(failures)
    num_evals = len(successes) + len(failures)
    total_path_len_weight = sum([entry['path_len_weight'] for entry in successes]) + \
                            sum([entry['path_len_weight'] for entry in failures])
    completed_goal_conditions = sum([entry['completed_goal_conditions'] for entry in successes]) + \
                                sum([entry['completed_goal_conditions'] for entry in failures])
    total_goal_conditions = sum([entry['total_goal_conditions'] for entry in successes]) + \
                            sum([entry['total_goal_conditions'] for entry in failures])

    # metrics
    sr = float(num_successes) / num_evals
    pc = completed_goal_conditions / float(total_goal_conditions)
    plw_sr = (float(sum([entry['path_len_weighted_success_spl'] for entry in successes]) +
                    sum([entry['path_len_weighted_success_spl'] for entry in failures])) /
                total_path_len_weight)
    plw_pc = (float(sum([entry['path_len_weighted_goal_condition_spl'] for entry in successes]) +
                    sum([entry['path_len_weighted_goal_condition_spl'] for entry in failures])) /
                total_path_len_weight)

    # result table
    res = dict()
    res['success'] = {'num_successes': num_successes,
                        'num_evals': num_evals,
                        'success_rate': sr}
    res['goal_condition_success'] = {'completed_goal_conditions': completed_goal_conditions,
                                    'total_goal_conditions': total_goal_conditions,
                                    'goal_condition_success_rate': pc}
    res['path_length_weighted_success_rate'] = plw_sr
    res['path_length_weighted_goal_condition_success_rate'] = plw_pc

    return res