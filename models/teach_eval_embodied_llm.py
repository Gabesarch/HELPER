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
        data_dir: str, 
        output_dir: str, 
        images_dir: str, 
        edh_instance: str = None, 
        max_init_tries: int =5, 
        replay_timeout: int = 500, 
        num_processes: int = 1, 
        iteration=0,
        er=None,
        depth_network=None,
        segmentation_network=None,
        ) -> None:

        super(SubGoalController, self).__init__()

        self.er = er
        split = args.split

        ###########%%%%%% PARAMS %%%%%%%#########
        self.log_every = args.log_every # log every X iters if generating video
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
        self.episode_in_try_except = args.episode_in_try_except # Continue to next episode if assertion error occurs? 
        self.dist_thresh = args.dist_thresh # distance threshold for point goal navigation 
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.mode = args.mode

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
            'MoveAhead':"Forward", 
            "RotateRight":"Turn Right", 
            "RotateLeft":"Turn Left", 
            "LookDown":"Look Down",
            "LookUp":"Look Up",
            "Done":"Stop",
            'PutObject':"Place",
            'PickupObject':"Pickup",
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
                'Desk', 'Bed', 'Sofa', 'ArmChair',
                'Chair', 'Dresser', 'Ottoman', 'DogBed', 
                'Footstool', 'Safe', 'TVStand'
                ]
        self.clean_classes = ["Bowl", "Cup", "Mug", "Plate"]

        if not self.use_gt_subgoals or args.use_llm_search or args.run_error_correction_llm:
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
        
        self.runner_config = InferenceRunnerConfig(
            data_dir=data_dir,
            split=split,
            output_dir=output_dir,
            images_dir=images_dir,
            model_class=None,
            model_args=None,
            num_processes=num_processes,
            max_init_tries=max_init_tries,
            replay_timeout=replay_timeout,
            max_api_fails=args.max_api_fails,
            max_traj_steps=args.max_traj_steps,
        )
        print("INIT TfD...")
        self.init_success = self.load_edh_instance(edh_instance)

        self.teach_task = TeachTask(
            self.er,
            action_to_mappedaction=self.action_to_mappedaction,
            approx_last_action_success=not args.use_gt_success_checker,
            max_fails=self.runner_config.max_api_fails,
            max_steps=self.runner_config.max_traj_steps,
            remove_unusable_slice=args.remove_unusable_slice,
            use_GT_error_feedback=self.use_GT_error_feedback,
        )
        self.teach_task.metrics = self.metrics

        if not self.init_success:
            print("task initialization failed.. moving to next episode..")
            return 
        self.controller = self.er.simulator.controller
        self.tag = f"{self.instance_id}_{self.game_id}"
        print("DONE.")
        
        # Find out if the agent is holding something to start 
        object_cat_pickup = None
        action_history = self.edh_instance['driver_action_history']
        in_hand = False
        for action in action_history:
            if action['action_name']=='Pickup' and action['oid'] is not None:
                object_id_pickup = action['oid']
                object_cat_pickup = object_id_pickup.split('|')[0]
                in_hand = True
            if action['action_name']=='Place' and action['oid'] is not None:
                object_id_pickup = None
                object_cat_pickup = None
                in_hand = False

        keep_head_straight = False
        search_pitch_explore = False
        block_map_positions_if_fail=True
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
            task=self.teach_task,
            depth_estimation_network=depth_network,
            )
        self.navigation.init_navigation(None)

        self.navigation.bring_head_to_angle(update_obs=False)
        
        origin_T_camX0 = utils.aithor.get_origin_T_camX(self.controller .last_event, False)

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
        self.teach_task.object_tracker = self.object_tracker

        if object_cat_pickup is not None and in_hand:
            # add holding object
            self.object_tracker.objects_track_dict[self.object_tracker.id_index] = {}
            self.object_tracker.objects_track_dict[self.object_tracker.id_index]['scores'] = 1.01
            self.object_tracker.objects_track_dict[self.object_tracker.id_index]['label'] = object_cat_pickup
            self.object_tracker.objects_track_dict[self.object_tracker.id_index]['locs'] = None
            self.object_tracker.objects_track_dict[self.object_tracker.id_index]['holding'] = True
            self.object_tracker.objects_track_dict[self.object_tracker.id_index]['can_use'] = True
            self.object_tracker.objects_track_dict[self.object_tracker.id_index]['sliced'] = False
            self.object_tracker.id_index += 1
        
        if args.create_movie:
            self.vis = Animation(
                self.W, self.H, 
                navigation=None if args.remove_map_vis else self.navigation, 
                name_to_id=self.name_to_id
                )
            os.makedirs(args.movie_dir, exist_ok=True)
        else:
            self.vis = None

    def load_edh_instance(self, edh_instance: str):
        self.edh_instance: dict = None

        if edh_instance is None:
            logging.info("No EDH instance specified, defaulting to FloorPlan12_physics of world type kitchen")
            if self.er is None:
                self.er: EpisodeReplay = EpisodeReplay("thor", web_window_size=480)
            self.er.simulator.start_new_episode(world="FloorPlan12_physics", world_type="kitchen")
            return
        else:
            if self.er is None:
                self.er = EpisodeReplay("thor", ["ego", "allo", "targetobject"])
            instance = load_json(edh_instance)
            check_task = InferenceRunner._get_check_task(instance, self.runner_config)
            game_file = InferenceRunner.get_game_file(instance, self.runner_config)
            instance_id = InferenceRunner._get_instance_id(edh_instance, instance)
            try:
                init_success, self.er = with_retry(
                    fn=lambda: InferenceRunner._initialize_episode_replay(instance, game_file, check_task,
                                                            self.runner_config.replay_timeout, self.er),
                    retries=self.runner_config.max_init_tries - 1,
                    check_first_return_value=True,
                )
                history_load_success, history_images = InferenceRunner._maybe_load_history_images(instance, self.runner_config)
                init_success = init_success and history_load_success
            except Exception:
                init_success = False
                print(f"Failed to initialize episode replay for instance={instance_id}")

            self.edh_instance = instance
            self.instance_id = instance_id
            self.game_id = self.edh_instance['game_id']
            self.metrics = create_new_traj_metrics(instance_id, self.game_id)
            self.metrics["init_success"] = init_success
            if args.use_progress_check:
                self.metrics_before_feedback = copy.deepcopy(self.metrics)
                self.metrics_before_feedback2 = copy.deepcopy(self.metrics)

            if not init_success:
                return init_success

            if "expected_init_goal_conditions_total" in instance and "expected_init_goal_conditions_satisfied" in instance:
                self.init_gc_total = instance["expected_init_goal_conditions_total"]
                self.init_gc_satisfied = instance["expected_init_goal_conditions_satisfied"]
            else:
                # For TfD instances, goal conditions are not cached so need an initial check
                (
                    _,
                    self.init_gc_total,
                    self.init_gc_satisfied,
                ) = InferenceRunner._check_episode_progress(self.er, check_task)

        self.er.simulator.is_record_mode = True
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
                for obj in self.er.simulator.controller.last_event.metadata['objects']:
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
                for obj in self.er.simulator.controller.last_event.metadata['objects']:
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
                for obj in self.er.simulator.controller.last_event.metadata['objects']:
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
                for obj in self.er.simulator.controller.last_event.metadata['objects']:
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
            if self.teach_task.is_done() or self.num_subgoal_fail>=self.max_subgoal_fail:
                break
            obj = None
            if self.use_constraint_check and check_constraints[0]:
                subgoals, objects, check_constraints = self.check_api_constraints(subgoals, objects, check_constraints)
            else:
                subgoals, objects, check_constraints = self.expand_subgoals(subgoals, objects, check_constraints)
            if args.always_check_constraints:
                print("OVERRIDING CONSTRAINTS TO ALWAYS CHECK")
                check_constraints = [True] * len(subgoals) 

            if len(subgoals)==0:
                break
            subgoal_name = subgoals.pop(0)
            object_name = objects.pop(0)
            check_constraints.pop(0)
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
                    if self.teach_task.num_fails<int((2/3)*self.teach_task.max_fails):
                        if run_error_correction:
                            if self.run_error_correction_llm:
                                if subgoal_attempts==max_attempts:
                                    subgoal_attempts = 0
                                elif subgoal_attempts==0:
                                    subgoals_, objects_, self.search_dict_ = self.run_llm_replan(self.edh_instance)
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
        '''
        User feedback
        '''
        check_task = InferenceRunner._get_check_task(self.edh_instance, self.runner_config)
        progress_check_output = check_task.check_episode_progress(self.er.simulator.get_objects(self.controller.last_event), self.er.simulator)
        task_dict = {'dialog_history_cleaned':[], 'success':progress_check_output['success']}
        if not progress_check_output['success']:
            for subgoal in progress_check_output['subgoals']:
                if not subgoal['success']:
                    subgoal_description = subgoal['description']
                    subgoal_step_description = []
                    objects_mentioned = []
                    idx_to_phrase = ["first", "second", "third", "fourth", "fifth", "sixth", "seventh"]
                    for step in subgoal['steps']:
                        if not step['success']:
                            if step['objectId'] not in objects_mentioned:
                                objects_mentioned.append(step['objectId'])
                            object_idx = objects_mentioned.index(step['objectId'])
                            subgoal_step_description.append(f"For the {idx_to_phrase[object_idx]} {step['objectType']}: {step['desc']}")
                    subgoal_failed_text = f"You failed to complete the subtask: {subgoal['description']} "
                    for step_desc in subgoal_step_description:
                        subgoal_failed_text += f'{step_desc} '
                    task_dict['dialog_history_cleaned'].append(['Commander', subgoal_failed_text])
        return task_dict

    def get_goal_condition_success(self):
        check_task = InferenceRunner._get_check_task(self.edh_instance, self.runner_config)
        (
            success,
            final_goal_conditions_total,
            final_goal_conditions_satisfied,
        ) = InferenceRunner._check_episode_progress(self.er, check_task)
        return final_goal_conditions_satisfied, final_goal_conditions_total

    def eval(self, additional_tag=''):
        '''
        Get task metrics
        '''
        check_task = InferenceRunner._get_check_task(self.edh_instance, self.runner_config)
        (
            success,
            final_goal_conditions_total,
            final_goal_conditions_satisfied,
        ) = InferenceRunner._check_episode_progress(self.er, check_task)

        metrics_diff = evaluate_traj(
            success,
            self.edh_instance,
            self.teach_task.steps,
            self.init_gc_total,
            self.init_gc_satisfied,
            final_goal_conditions_total,
            final_goal_conditions_satisfied,
        )
        self.teach_task.metrics.update(metrics_diff)

        progress_check_output = check_task.check_episode_progress(self.er.simulator.get_objects(self.controller.last_event), self.er.simulator)

        satisfied_objects = []
        if progress_check_output['satisfied_objects'] is not None:
            for idx in range(len(progress_check_output['satisfied_objects'])):
                satisfied_objects.append(progress_check_output['satisfied_objects'][idx]['objectType'])

        candidate_objects = []
        if progress_check_output['candidate_objects'] is not None:
            for idx in range(len(progress_check_output['candidate_objects'])):
                candidate_objects.append(progress_check_output['candidate_objects'][idx]['objectType'])

        failure_subgoal_description = []
        successful_subgoal_description = []
        failure_step_description = []
        successful_step_description = []
        for idx in range(len(progress_check_output['subgoals'])):
            subgoal = progress_check_output['subgoals'][idx]
            if not subgoal['success']:
                failure_subgoal_description.append(subgoal['description'])
            else:
                successful_subgoal_description.append(subgoal['description'])
            for step_idx in range(len(subgoal['steps'])):
                step = subgoal['steps'][step_idx]
                if not step['success']:
                    failure_step_description.append(step["desc"])
                else:
                    successful_step_description.append(step["desc"])

        print(f"Failed steps: {failure_step_description}")
        
        task_log = {
            "task description":progress_check_output["description"],
            "satisfied objects":str(satisfied_objects),
            "candidate objects":str(candidate_objects),
            "failed subtasks":str(failure_subgoal_description),
            "successful subtasks":str(successful_subgoal_description),
            "failed steps":str(failure_step_description),
            "successful steps":str(successful_step_description),
            }
        subgoal_log = {
            "failed_subgoals":str(self.failed_subgoals), 
            "success_subgoals":str(self.successful_subgoals), 
            "error_correct_subgoals":str(self.error_correct_fix), 
            "attempted_subgoals":str(self.attempted_subgoals)
            }
        self.teach_task.metrics.update(task_log)
        self.teach_task.metrics.update(subgoal_log)
        self.teach_task.metrics.update(self.llm_log)
        tbl = wandb.Table(columns=list(self.teach_task.metrics.keys()))
        tbl.add_data(*list(self.teach_task.metrics.values()))
        wandb.log({f"Metrics{additional_tag}/{self.tag}": tbl})

        print("Eval:")
        print(additional_tag)
        print(f"Task success: {success}")
        print(f"Final subgoal success: {final_goal_conditions_satisfied} / {final_goal_conditions_total}")

    def run_tfd(self, user_progress_check=True):
        if self.episode_in_try_except:
            try:
                self.search_dict = {}
                camX0_T_camXs = self.map_and_explore()
                print("RUNNING IN TRY EXCEPT")
                if self.use_gt_subgoals:
                    subgoals = []
                    objects = []
                    for action_idx in range(len(self.edh_instance['driver_actions_future'])):
                        action = self.edh_instance['driver_actions_future'][action_idx]
                        print(action["action_name"])
                        if action["action_name"] in ["Place", "Pickup", 'Open', 'Close', "ToggleOn", "ToggleOff", "Slice", "Pour"]:
                            object_name = action['oid'].split('|')[0]
                            if "Sliced" in action['oid'] and "Sliced" not in object_name:
                                object_name+="Sliced"
                            subgoals.extend(["Navigate", action["action_name"]])
                            objects.extend([object_name, object_name])
                            done_object = True
                            for action_ in self.edh_instance['driver_actions_future'][action_idx+1:]:
                                if action['oid']==action_['oid']:
                                    done_object = False
                            if done_object:
                                subgoals.append("ObjectDone")
                                objects.append(object_name)
                else:
                    subgoals, objects, self.search_dict = self.run_llm(self.edh_instance)
                # print("SUBGOALS:", subgoals)
                # print("ARGUMENTS:", objects)
                print("SUBGOALS:", [[s, o] for s, o in zip(subgoals, objects)])
                dialog = ". ".join([
                    d[-1] for d in self.edh_instance['dialog_history_cleaned']
                    if 'Commander' in d[0]
                ]).lower()
                print(f"Full dialogue (Commander): {dialog}")
                total_tasks = len(subgoals)
                succ = self.run_subgoals(subgoals, objects, run_error_correction=True, completed_subgoals=self.completed_subgoals)
                if user_progress_check:
                    self.eval(additional_tag="_before_progresscheck")
                    self.metrics_before_feedback = copy.deepcopy(self.teach_task.metrics)
                    task_dict = self.progress_check()
                    if not task_dict['success']:
                        if self.add_back_objs_progresscheck:
                            # reset removed objects
                            for obj_ID in self.object_tracker_ids_removed:
                                self.object_tracker.objects_track_dict[obj_ID]["can_use"] = True 
                            self.object_tracker_ids_removed = []
                        subgoals, objects, self.search_dict = self.run_llm(task_dict, log_tag='_user_feedback')
                        self.completed_subgoals = []
                        succ = self.run_subgoals(subgoals, objects, run_error_correction=True, completed_subgoals=self.completed_subgoals)
                        self.eval(additional_tag="_after_progresscheck")
                    self.metrics_before_feedback2 = copy.deepcopy(self.teach_task.metrics)
                    task_dict = self.progress_check()
                    if not task_dict['success']:
                        if args.add_back_objs_progresscheck_v2:
                            # reset removed objects
                            for obj_ID in self.object_tracker_ids_removed:
                                self.object_tracker.objects_track_dict[obj_ID]["can_use"] = True 
                            self.object_tracker_ids_removed = []
                        subgoals, objects, self.search_dict = self.run_llm(task_dict, log_tag='_user_feedback2')
                        self.completed_subgoals = []
                        succ = self.run_subgoals(subgoals, objects, run_error_correction=True, completed_subgoals=self.completed_subgoals)
                        self.eval(additional_tag="_after_progresscheck2")

            except KeyboardInterrupt:
                # sys.exit(0)
                pass
            except Exception as e:
                tbl = wandb.Table(columns=["Error", "Traceback"])
                tbl.add_data(str(e), str(traceback.format_exc()))
                wandb.log({f"Errors/{self.tag}": tbl})
                print(e)
                print(traceback.format_exc())
        else:
            self.search_dict = {}
            camX0_T_camXs = self.map_and_explore()
            if self.use_gt_subgoals:
                subgoals = []
                objects = []
                for action_idx in range(len(self.edh_instance['driver_actions_future'])):
                    action = self.edh_instance['driver_actions_future'][action_idx]
                    print(action["action_name"])
                    if action["action_name"] in ["Place", "Pickup", 'Open', 'Close', "ToggleOn", "ToggleOff", "Slice", "Pour"]:
                        object_name = action['oid'].split('|')[0]
                        if "Sliced" in action['oid'] and "Sliced" not in object_name:
                            object_name+="Sliced"
                        subgoals.extend(["Navigate", action["action_name"]])
                        objects.extend([object_name, object_name])
                        done_object = True
                        for action_ in self.edh_instance['driver_actions_future'][action_idx+1:]:
                            if action['oid']==action_['oid']:
                                done_object = False
                        if done_object:
                            subgoals.append("ObjectDone")
                            objects.append(object_name)
            else:
                subgoals, objects, self.search_dict = self.run_llm(self.edh_instance)
            print("SUBGOALS:", subgoals)
            print("ARGUMENTS:", objects)

            dialog = ". ".join([
                d[-1] for d in self.edh_instance['dialog_history_cleaned']
                if 'Commander' in d[0]
            ]).lower()
            print(f"Full dialogue (Commander): {dialog}")
            succ = self.run_subgoals(subgoals, objects, run_error_correction=True, completed_subgoals=self.completed_subgoals)
            if user_progress_check:
                self.eval(additional_tag="_before_progresscheck") 
                self.metrics_before_feedback = copy.deepcopy(self.teach_task.metrics)
                task_dict = self.progress_check()
                if not task_dict['success']:
                    if self.add_back_objs_progresscheck:
                        # reset removed objects
                        for obj_ID in self.object_tracker_ids_removed:
                            self.object_tracker.objects_track_dict[obj_ID]["can_use"] = True 
                        self.object_tracker_ids_removed = []
                    subgoals, objects, self.search_dict = self.run_llm(task_dict, log_tag='_user_feedback')
                    self.completed_subgoals = []
                    succ = self.run_subgoals(subgoals, objects, run_error_correction=True, completed_subgoals=self.completed_subgoals)
                    self.eval(additional_tag="_after_progresscheck")
                self.metrics_before_feedback2 = copy.deepcopy(self.teach_task.metrics)  
                task_dict = self.progress_check()
                if not task_dict['success']:
                    if args.add_back_objs_progresscheck_v2:
                        # reset removed objects
                        for obj_ID in self.object_tracker_ids_removed:
                            self.object_tracker.objects_track_dict[obj_ID]["can_use"] = True 
                        self.object_tracker_ids_removed = []
                    subgoals, objects, self.search_dict = self.run_llm(task_dict, log_tag='_user_feedback2')
                    self.completed_subgoals = []
                    succ = self.run_subgoals(subgoals, objects, run_error_correction=True, completed_subgoals=self.completed_subgoals)
                    self.eval(additional_tag="_after_progresscheck2")

    def run(self):
        if args.mode in ["teach_eval_tfd", "teach_eval_custom"]:
            self.run_tfd(user_progress_check=self.use_progress_check)
        elif args.mode=="teach_eval_edh":
            self.run_edh()
        self.teach_task.step("Stop", None)
        self.eval()
        if self.controller is not None:
            self.controller.stop()
        self.render_output()

        return self.teach_task.metrics, self.er

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

def run_teach():
    save_metrics = True
    split_ = args.split
    data_dir = args.teach_data_dir 
    output_dir = "./plots/subgoal_output"
    images_dir = "./plots/subgoal_output"
    if args.mode=="teach_eval_tfd":
        instance_dir = os.path.join(data_dir, f"tfd_instances/{split_}")
    elif args.mode=="teach_eval_edh":
        instance_dir = os.path.join(data_dir, f"edh_instances/{split_}")
    files = os.listdir(instance_dir) # sample every other

    if args.sample_every_other:
        files = files[::2]

    if args.episode_file is not None:
        files_idx = files.index(args.episode_file)
        files = files[files_idx:files_idx+1]

    if args.max_episodes is not None:
        files = files[:args.max_episodes]

    # initialize wandb
    if args.set_name=="test00":
        wandb.init(mode="disabled")
    else:
        wandb.init(project="embodied-llm-teach", name=args.set_name, group=args.group, config=args, dir=args.wandb_directory)

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
    er = None
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
        subgoalcontroller = SubGoalController(
                data_dir, 
                output_dir, 
                images_dir, 
                task_instance, 
                iteration=iter_, 
                er=er, 
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
        metrics[file] = metrics_instance
        if args.use_progress_check:
            metrics_instance_before_feedback = subgoalcontroller.metrics_before_feedback
            metrics_before_feedback[file] = metrics_instance_before_feedback
            metrics_instance_before_feedback2 = subgoalcontroller.metrics_before_feedback2
            metrics_before_feedback2[file] = metrics_instance_before_feedback2

        iter_ += 1

        if save_metrics:
            from teach.eval.compute_metrics import aggregate_metrics

            aggregrated_metrics = aggregate_metrics(metrics, args)

            print('\n\n---------- File 1 ---------------')
            to_log = []  
            to_log.append('-'*40 + '-'*40)
            list_of_files = files 
            to_log.append(f'Split: {split_}')
            to_log.append(f'Number of files: {len(list(metrics.keys()))}')
            for f_n in aggregrated_metrics.keys(): 
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
            cols.remove('pred_actions')
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

                aggregrated_metrics_before_feedback = aggregate_metrics(metrics_before_feedback, args)

                print('\n\n---------- File 1 ---------------')
                to_log = []  
                to_log.append('-'*40 + '-'*40)
                list_of_files = files 
                to_log.append(f'Split: {split_}')
                to_log.append(f'Number of files: {len(list(metrics_before_feedback.keys()))}')
                for f_n in aggregrated_metrics_before_feedback.keys(): 
                    to_log.append(f'{f_n}: {aggregrated_metrics_before_feedback[f_n]}') 
                to_log.append('-'*40 + '-'*40)

                aggregrated_metrics_before_feedback["num episodes"] = iter_
                tbl = wandb.Table(columns=list(aggregrated_metrics_before_feedback.keys()))
                tbl.add_data(*list(aggregrated_metrics_before_feedback.values()))
                wandb.log({f"Metrics_summary/Summary_before_feedback": tbl, 'step':iter_})

                save_dict_as_json(metrics_before_feedback, metrics_file_before_feedback)


                aggregrated_metrics_before_feedback2 = aggregate_metrics(metrics_before_feedback2, args)

                print('\n\n---------- File 1 ---------------')
                to_log = []  
                to_log.append('-'*40 + '-'*40)
                list_of_files = files 
                to_log.append(f'Split: {split_}')
                to_log.append(f'Number of files: {len(list(metrics_before_feedback2.keys()))}')
                for f_n in aggregrated_metrics_before_feedback2.keys(): #keys_include:
                    to_log.append(f'{f_n}: {aggregrated_metrics_before_feedback2[f_n]}') 
                to_log.append('-'*40 + '-'*40)

                aggregrated_metrics_before_feedback2["num episodes"] = iter_
                tbl = wandb.Table(columns=list(aggregrated_metrics_before_feedback2.keys()))
                tbl.add_data(*list(aggregrated_metrics_before_feedback2.values()))
                wandb.log({f"Metrics_summary/Summary_before_feedback": tbl, 'step':iter_})

                save_dict_as_json(metrics_before_feedback2, metrics_file_before_feedback2)