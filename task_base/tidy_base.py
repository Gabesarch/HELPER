from utils.aithor import compute_metrics, read_task_data
import numpy as np
from arguments import args
import skimage.morphology
import map_and_plan.FILM.alfred_utils.gen.constants as constants
import ipdb
st = ipdb.set_trace

class TidyTask():
    def __init__(
        self, 
        env, 
        action_to_mappedaction=None,
        object_instance_ids=None,
        max_steps=1000,
        max_fails=30,
        object_tracker=None,
        approx_last_action_success=False,
        use_GT_error_feedback=False,
        remove_unusable_slice=False,
        ):

        self.env = env
        self.steps = 0
        self.action_to_mappedaction = action_to_mappedaction
        self.num_fails = 0
        self.max_steps = max_steps
        self.max_fails = max_fails
        self.remove_unusable_slice = remove_unusable_slice
        self.approx_last_action_success = approx_last_action_success
        self.use_GT_error_feedback = use_GT_error_feedback
        self.last_action = None
        self.object_tracker = object_tracker
        if self.approx_last_action_success:
            self.success_detector = CheckSuccessfulAction()
        if not self.use_GT_error_feedback:
            self.error_feedback = ErrorFeedback()
        self.metrics = {}

        self.W = args.W 
        self.H = args.H 

        self.args = args
        self.err_message = None
        self.help_message = None     
    
    def get_observations(self):
        obs = self.env.get_observations()
        if self.object_tracker is not None:
            obs["is_holding"] = self.object_tracker.is_holding()
        return obs

    def action_success(self):
        if self.approx_last_action_success:
            success = self.success_detector.check_successful_action(self.env.controller.last_event.frame, self.last_action)
        else:
            success = self.env.action_success() #self.env.last_event.metadata["lastActionSuccess"]
        return success

    def get_agent_head_tilt(self):
        return self.env.controller.last_event.metadata["agent"]["cameraHorizon"]

    def is_done(self):
        return self.env.is_done()

    def get_metrics(self):
        metrics = self.env.get_metrics()
        # success = self.env.get_goal_satisfied()
        # metrics = compute_metrics(success, self.reward, self.traj_data, self.steps, self.env.get_goal_conditions_met())
        return metrics #, **self.task_info)

    def step(self, action, obj_relative_coord=None, object_name=None, log_error=True):

        if self.action_to_mappedaction is not None and action in self.action_to_mappedaction.keys():
            action = self.action_to_mappedaction[action]

        if action in ["ToggleOn", "ToggleOff", "ToggleObjectOn", "ToggleObjectOff"]:
            # no toggle
            return True
        
        self.last_action = action
        if self.approx_last_action_success:
            self.success_detector.update_image(self.env.controller.last_event.frame)

        if self.is_done():
            print(f"Max steps ({self.steps}; max={self.max_steps}) or max fails ({self.num_fails}; max={self.max_fails}) reached! Returning without interaction..")
            return

        if obj_relative_coord is not None:
            obj_relative_coord = np.asarray(obj_relative_coord)
            obj_relative_coord = obj_relative_coord[[1,0]] # tidy task requires x,y -> y,x
        sim_succ = self.env.step(action, obj_relative_coord)
        self.sim_succ = sim_succ
        err_message = self.env.controller.last_event.metadata['errorMessage']

        self.steps = self.env.step_count
        print(f"Steps reached: {self.steps}/{self.max_steps} (action: {action})")
        # sim_succ = self.env.last_event.metadata["lastActionSuccess"]
        self.sim_succ = sim_succ

        if not sim_succ:
            self.num_fails += 1
            print(f"Error message: {err_message}")
            print(f"Number of failures: {self.num_fails}/{self.max_fails}")

        if self.remove_unusable_slice and sim_succ and action=="Slice":
            for obj in self.env.controller.last_event.metadata['objects']:
                if "Slice" in obj["name"] and object_name in obj["name"] and "Slice" not in obj["objectId"]:
                    # even though we sliced the object, some of the object remains unsliced and unusable? Let's remove this
                    self.env.controller.step(
                        action="RemoveFromScene",
                        objectId=obj["objectId"]
                    )

        if log_error:
            if self.use_GT_error_feedback:
                self.err_message = err_message #self.env.last_event.metadata['errorMessage']
                self.help_message = ""
                if not self.sim_succ:
                    # if already toggled on, change error message
                    if action=="ToggleOn":
                        for obj in self.env.controller.last_event.metadata['objects']:
                            if obj['isToggled'] and obj['objectType']==object_name and obj['visible']:
                                self.err_message = "Object is already toggled on."
                                self.help_message = "You cannot toggle on an object that is already on."
                    elif action=="ToggleOff":
                        for obj in self.env.controller.last_event.metadata['objects']:
                            if (not obj['isToggled']) and obj['objectType']==object_name and obj['visible']:
                                self.err_message = "Object is already toggled off."
                                self.help_message = "You cannot toggle off an object that is already off."

                    if "not a valid Object Type to be placed" in err_message and "Bread" in err_message and "Toaster" in err_message:
                        self.err_message = "Bread is too big to fit into toaster. Cannot place it there."
                        self.help_message = "You should slice the bread thinner to place it in the toaster."
            else:
                if not self.action_success():
                    if obj_relative_coord is None:
                        self.err_message = "The agent is blocked from moving in that direction."
                        self.help_message = "Find an alternate route or viewpoint."
                    else:
                        feedback = self.error_feedback.get_error_message(self.env.controller.last_event.frame)
                        self.err_message = feedback
                        self.help_message = ""
                else:
                    self.err_message = "The action was successful."
                    self.help_message = "Do nothing and carry on."

        self.event = self.env.controller.last_event

        return sim_succ

class CheckSuccessfulAction():
    def __init__(self, rgb_init=None):
        '''
        rgb_init: the rgb image from the spawn viewpoint W, H, 3
        This class does a simple check with the previous image to see if it completed the action 
        '''
        self.rgb_prev = rgb_init
        # self.perc_diff_thresh = perc_diff_thresh
        # self.H = H
        # self.W = W

    def update_image(self, rgb):
        self.rgb_prev = rgb

    def check_successful_action(self, rgb, action):
        wheres = np.where(self.rgb_prev != rgb)
        wheres_ar = np.zeros(self.rgb_prev.shape)
        wheres_ar[wheres] = 1
        wheres_ar = np.sum(wheres_ar, axis=2).astype(bool)
        connected_regions = skimage.morphology.label(wheres_ar, connectivity=2)
        unique_labels = [i for i in range(1, np.max(connected_regions)+1)]
        max_area = -1
        for lab in unique_labels:
            wheres_lab = np.where(connected_regions == lab)
            max_area = max(len(wheres_lab[0]), max_area)
        if (action in ['OpenObject', 'CloseObject']) and max_area > 500:
            success = True
        elif max_area > 100:
            success = True
        else:
            success = False
        return success

class ErrorFeedback():
    def __init__(self):
        '''
        CLIP error feedback
        '''
        from nets.clip import ALIGN
        self.model = ALIGN()
        with open('task_base/feedback.txt') as f:
            self.lines = [line.rstrip() for line in f]

    def get_error_message(self, rgb):
        probs = self.model.score(rgb, self.lines)
        argmax_error = np.argmax(probs.cpu().numpy())
        feedback = self.lines[argmax_error]
        return feedback