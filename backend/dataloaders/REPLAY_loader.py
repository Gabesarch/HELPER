import numpy as np
import torch
import torch.utils.data
import torchvision.transforms as transforms
from PIL import Image
from arguments import args
from utils.aithor import get_masks_from_seg
import glob
import os
import ipdb
st = ipdb.set_trace
from tqdm import tqdm
import random
import time
import utils.geom
import utils.aithor
from scipy.spatial.transform import Rotation 

import torch
import re
try:
    from torch._six import container_abcs, string_classes, int_classes
except:
    import collections.abc as container_abcs
    int_classes = int
    string_classes = str
import h5py


ORIGINAL_IMAGES_FOLDER = "raw_images"
HISTORY_IMAGES_FOLDER = "raw_history_images"
HIGH_RES_IMAGES_FOLDER = "high_res_images"
DEPTH_IMAGES_FOLDER = "depth_images"
INSTANCE_MASKS_FOLDER = "instance_masks"
TARGETS_FOLDER = "targets"

targets_to_output = ['boxes', 'masks', 'labels', 'obj_targets', 'expert_action']
history_targets = ['masks']

# fix the seed for reproducibility
torch.manual_seed(args.seed)
np.random.seed(args.seed)
random.seed(args.seed)

class RICKDataset(torch.utils.data.Dataset):
    def __init__(
        self, 
        split,
        W, H, 
        max_objs,
        idx2actions,
        id_to_name,
        start_index=None, 
        end_index=None,
        shuffle=False,
        images=None,
        name_to_mapped_name=None,
        subsample=None,
        ):
        """
        Args:
            root_dir (string): root directory with all the images, etc.
            W, H (int): width and height of frame
            max_objs (int): maximum objects per image allowable
            id_to_name (dict): converts object name idx to word
        """
        root_dir, root_dir2, root_dir3 = None, None, None
        if split == 'train':
            root_dir = args.alfred_load_train_root
            root_dir2 = args.alfred_load_train_root2
            root_dir3 = args.alfred_load_train_root3
            root_dir_demos = args.demos_load_train_root
        elif split == 'valid_seen':
            root_dir = args.alfred_load_valid_seen_root
            root_dir_demos = args.demos_load_valid_seen_root
        elif split == "valid_unseen":
            root_dir = args.alfred_load_valid_unseen_root
            root_dir_demos = args.demos_load_valid_unseen_root
        else:
            assert False

        # if args.debug:
        #     print("Running in debug mode")
        #     root_dir = args.alfred_load_valid_seen_root
        #     root_dir2, root_dir3 = None, None
        #     shuffle = False
        #     subsample = None

        if images is None:
            print(f"Getting image paths from {root_dir} ...")
            self.images = [d for d in tqdm(glob.glob(root_dir + 2 * '/*' + f'/{ORIGINAL_IMAGES_FOLDER}/*'))]
            if root_dir2 is not None:
                print(f"Getting image paths from {root_dir2} ...")
                images2 = [d for d in tqdm(glob.glob(root_dir2 + 2 * '/*' + f'/{ORIGINAL_IMAGES_FOLDER}/*'))]
                self.images += images2
            if root_dir3 is not None:
                print(f"Getting image paths from {root_dir3} ...")
                images3 = [d for d in tqdm(glob.glob(root_dir3 + 2 * '/*' + f'/{ORIGINAL_IMAGES_FOLDER}/*'))]
                self.images += images3
        else:
            self.images = images
                
        if shuffle:
            random.shuffle(self.images)
        if args.train_num_steps_away is not None:
            self.images, idxs = self.get_images_X_steps_away(args.train_num_steps_away)
            idxs = np.array(idxs)
        
        if end_index is not None:
            if start_index is None:
                start_index = 0
            self.images = self.images[start_index:end_index]
        if subsample is not None:
            self.images = self.images[::subsample]

        # self.images = self.images[215853:215854]

        # debug setting
        if args.debug:
            self.images = self.images[:1280]

        self.image_mean = np.array([0.485,0.456,0.406]).reshape(1,3,1,1)
        self.image_std = np.array([0.229,0.224,0.225]).reshape(1,3,1,1)
        
        self.W, self.H, self.max_objs, self.fov = int(W), int(H), int(max_objs), int(args.fov)
        hfov = float(self.fov) * np.pi / 180.
        self.pix_T_camX = np.array([
            [(self.W/2.)*1 / np.tan(hfov / 2.), 0., 0., 0.],
            [0., (self.H/2.)*1 / np.tan(hfov / 2.), 0., 0.],
            [0., 0.,  1, 0],
            [0., 0., 0, 1]])
        self.pix_T_camX[0,2] = self.W/2.
        self.pix_T_camX[1,2] = self.H/2.

        self.split = split
        self.idx2actions, self.id_to_name = idx2actions, id_to_name
        self.add_whole_image_mask = args.add_whole_image_mask # add whole image mask to object history 
        print("add_whole_image_mask", args.add_whole_image_mask)
        self.only_decode_intermediate = args.only_decode_intermediate # only decode intermediate/interaction objects (versus all objects in view)
        print("only_decode_intermediate", args.only_decode_intermediate)
        self.roi_pool = args.roi_pool
        print("roi_pool", args.roi_pool)
        self.roi_align = args.roi_align
        print("roi_align", args.roi_align)
        self.name_to_mapped_name = name_to_mapped_name
        self.map_names = args.map_names
        print("map_names", args.map_names)
        self.max_image_history = args.max_image_history
        self.get_centroids_from_masks = args.get_centroids_from_masks
        print("get_centroids_from_masks", args.get_centroids_from_masks)
        self.demo_type = args.demo_type
        self.max_demo_history = args.max_demo_history
        self.max_demo_future = args.max_demo_future
        # self.use_GT_tracklets = args.use_GT_tracklets
        self.use_GT_intermediate_objs = args.use_GT_intermediate_objs
        # print("use_GT_tracklets", args.use_GT_tracklets)
        print("use_GT_intermediate_objs", args.use_GT_intermediate_objs)
        self.do_feature_cloud_history = args.do_feature_cloud_history
        print("do_feature_cloud_history", args.do_feature_cloud_history)
        self.use_object_tracklets = args.use_object_tracklets
        print("use_object_tracklets", args.use_object_tracklets)
        self.do_planner_path = args.do_planner_path
        print("do_planner_path", args.do_planner_path)
        if self.do_planner_path:
            self.remove_without_path = True
            self.remove_samples_without_intermobj = True
        else:
            if args.remove_without_path:
                self.remove_without_path = True
            else:
                self.remove_without_path = False
            if args.remove_samples_without_intermobj:
                self.remove_samples_without_intermobj = True
            else:
                self.remove_samples_without_intermobj = False
        print("remove_without_path", args.remove_without_path)
        print("remove_samples_without_intermobj", args.remove_samples_without_intermobj)
        self.demo_mapping = {}

        if self.demo_type=="augmented_same":
            self.root_dir_demos = root_dir_demos
            print(f"Getting demo image paths from {root_dir_demos} ...")
            self.images_demos = [d for d in tqdm(glob.glob(root_dir_demos + 2 * '/*' + f'/{ORIGINAL_IMAGES_FOLDER}/*'))]
            print("Gettting image->demo mapping (this may take some time)...")
            if args.debug:
                self.images_demos = self.images_demos[:128]
            # here we define a mapping to tell us which demo image to retrieve for each training image
            self.define_augmented_demo_mapping()

        targets_to_output.append('agent_pose')
        targets_to_output.append('obj_centroids')
        if self.use_object_tracklets:
            targets_to_output.append('obj_instance_ids')

        self.action_weights = None

        if not args.debug:

            self.remove_without_hdf5()

            if args.get_action_weights_from_dataloader:
                self.action_weights = self.get_action_weights()                

            if self.remove_samples_without_intermobj:
                self.filter_without_intermobj()

            if self.remove_without_path:
                self.filter_no_path()

            if args.remove_done:
                self.images = self.remove_done_images()

        print(f"Number of unique episodes: {self.number_unique_episodes()}")

    def __len__(self):
        return len(self.images)

    def number_unique_episodes(self):
        unique_eps = set()
        for im in self.images:
            key = im.split('/')[-4:-2]
            unique_eps.add(tuple(key))
        return len(unique_eps)

    def remove_without_hdf5(self):
        images2 = []
        for idx in tqdm(range(len(self.images))):
            image_t = self.images[idx]
            targets_path = image_t.replace(ORIGINAL_IMAGES_FOLDER, TARGETS_FOLDER).replace('.png', '')
            targets_path, tag = os.path.split(targets_path)
            if os.path.exists(targets_path+'/targets.hdf5'):
                images2.append(image_t)
        print(f"NOTE: REMOVING {len(self.images) - len(images2)} IMAGES WITHOUT HDF5 TARGETS!")
        self.images = images2

    def filter_no_path(self):
        images2 = []
        for idx in tqdm(range(len(self.images))):
            image_t = self.images[idx]
            targets_path = image_t.replace(ORIGINAL_IMAGES_FOLDER, TARGETS_FOLDER).replace('.png', '')
            targets_path, tag = os.path.split(targets_path)
            targets_episode = h5py.File(targets_path+'/targets.hdf5', 'r')
            data = targets_episode[tag+'_planner_paths'][()]
            targets_episode.close()
            if not np.all(np.isnan(data)):
                images2.append(image_t)
        print(f"NOTE: REMOVING {len(self.images) - len(images2)} IMAGES WITHOUT PLANNER PATH!")
        self.images = images2

    def get_action_weights(
        self,
    ):
        '''
        Gets proportion of each action for cross entropy loss
        This should just be called once for new data when the weights are needed
        '''
        print("Getting action weights....")

        actions = {i:0 for i in range(13)}
        for idx in tqdm(range(len(self.images))):
            image_t = self.images[idx]
            targets_path = image_t.replace(ORIGINAL_IMAGES_FOLDER, TARGETS_FOLDER).replace('.png', '')
            targets_path, tag = os.path.split(targets_path)
            targets_episode = h5py.File(targets_path+'/targets.hdf5', 'r')
            data = int(targets_episode[tag+'_expert_action'][()])
            actions[data] += 1
            targets_episode.close()
            
        action_counts = np.array(list(actions.values()))
        # action_weights = np.median(action_counts)/action_counts

        # action_weights = 1-action_counts/sum(action_counts)
        action_weights = sum(action_counts) / (len(action_counts)*action_counts)
        action_weights = action_weights/max(action_weights) * 5 # normalize to be maximum value of 5
        print("Action counts:", action_counts)
        print("Action weights:", action_weights)
        return action_weights

    def filter_without_intermobj(
        self,
    ):
        '''
        Removes images without intermediate object in the history
        '''

        images2 = []
        for idx in tqdm(range(len(self.images))):

            image_t = self.images[idx]
            self.image_t = image_t
            image_episode_idx = int(os.path.split(image_t)[-1].split('_')[0])
            image_folder = os.path.split(image_t)[0]
            history_folder = image_folder.replace(ORIGINAL_IMAGES_FOLDER, HISTORY_IMAGES_FOLDER)
            history_images_path = [d for d in glob.glob(image_folder+'/*')+glob.glob(history_folder+'/*')
                if (
                    int(os.path.split(d)[-1].split('_')[0])<image_episode_idx 
                    and (image_episode_idx-int(os.path.split(d)[-1].split('_')[0]))<self.max_image_history
                    )]
            history_targets_path = [d.replace(ORIGINAL_IMAGES_FOLDER, TARGETS_FOLDER).replace('.png', '') for d in history_images_path]
            history_targets_path = [d.replace(HISTORY_IMAGES_FOLDER, TARGETS_FOLDER).replace('.png', '') for d in history_targets_path]
            targets_path = image_t.replace(ORIGINAL_IMAGES_FOLDER, TARGETS_FOLDER).replace('.png', '')
            history_targets_path = history_targets_path + [targets_path]

            targets_path, tag = os.path.split(targets_path)
            targets_episode = h5py.File(targets_path+'/targets.hdf5', 'r')

            subgoal = targets_episode[tag+"_subgoal"][()]
            target_object = subgoal[1]

            no_interms = True
            for target_path in history_targets_path:
                _, tag = os.path.split(target_path)
                labels = targets_episode[tag+"_labels"][()]
                interm_idxs = np.where(labels==target_object)[0]
                if len(interm_idxs)>0:
                    no_interms = False
                    break
            if not no_interms:
                images2.append(image_t)
            targets_episode.close()

        print(f"NOTE: REMOVING {len(self.images) - len(images2)} IMAGES WITHOUT INTERMEDIATE OBJECT IN HISTORY")
        self.images = images2

    def get_images_X_steps_away(self, X):
        
        print(f"Filtering images {X} steps away...")
        images = []
        idxs = []
        for idx in tqdm(range(len(self.images))):
            image_t = self.images[idx]
            image_episode_idx = int(os.path.split(image_t)[-1].split('_')[0])
            image_folder = os.path.split(image_t)[0]
            history_folder = image_folder.replace(ORIGINAL_IMAGES_FOLDER, HISTORY_IMAGES_FOLDER)
            current_subgoal_index = int(image_t[-6:-4]) # get subgoal index
            subgoal_images_path = [d for d in glob.glob(image_folder+'/*')+glob.glob(history_folder+'/*')
                if (
                    int(d[-6:-4])==current_subgoal_index 
                    )]
            subgoal_images_arg_idxs = np.argsort(np.array([int(os.path.split(d)[-1].split('_')[0]) for d in subgoal_images_path]))
            subgoal_images_path = [subgoal_images_path[d_i] for d_i in list(subgoal_images_arg_idxs)]
            goal_subgoal_image_path = subgoal_images_path[-1]
            goal_subgoal_targets_path = goal_subgoal_image_path.replace(ORIGINAL_IMAGES_FOLDER, TARGETS_FOLDER).replace('.png', '')
            image_subgoal_index = [image_t==im for im in subgoal_images_path]
            image_subgoal_index = np.where(image_subgoal_index)[0]
            if (len(subgoal_images_path)-1) - int(image_subgoal_index) <= X: # add one to X since we have "done" action
                images.append(image_t)
                idxs.append(idx)

        return images, idxs

    def remove_done_images(self):
        print("Removing done images...")
        images = []
        for idx in tqdm(range(len(self.images))):
            image_t = self.images[idx]
            targets_path = image_t.replace(ORIGINAL_IMAGES_FOLDER, TARGETS_FOLDER).replace('.png', '')
            targets_path, tag = os.path.split(targets_path)
            targets_episode = h5py.File(targets_path+'/targets.hdf5', 'r')
            data = int(targets_episode[tag+'_expert_action'][()])
            targets_episode.close()
            if data!=len(self.idx2actions)-1:
                # TODO: last subgoal does not have done? fix this in datagen
                images.append(image_t)
        print(f"Removed {len(self.images) - len(images)} done images")

        return images

    def define_augmented_demo_mapping(self):
        # first get distance to goal for each image
        demos_organized = {}
        for image_t in tqdm(self.images_demos):
            # image_episode_idx = int(os.path.split(image_t)[-1].split('_')[0])
            image_folder = os.path.split(image_t)[0]
            history_folder = image_folder.replace(ORIGINAL_IMAGES_FOLDER, HISTORY_IMAGES_FOLDER)
            current_subgoal_index = int(image_t[-6:-4]) # get subgoal index
            subgoal_images_path = [d for d in glob.glob(image_folder+'/*')+glob.glob(history_folder+'/*')
                    if (
                        int(d[-6:-4])==current_subgoal_index 
                        )]
            subgoal_images_arg_idxs = np.argsort(np.array([int(os.path.split(d)[-1].split('_')[0]) for d in subgoal_images_path]))
            subgoal_images_path = [subgoal_images_path[d_i] for d_i in list(subgoal_images_arg_idxs)]
            goal_subgoal_image_path = subgoal_images_path[-1]
            goal_subgoal_targets_path = goal_subgoal_image_path.replace(ORIGINAL_IMAGES_FOLDER, TARGETS_FOLDER).replace('.png', '')
            image_subgoal_index = [image_t==im for im in subgoal_images_path]
            image_subgoal_index = np.where(image_subgoal_index)[0]
            distance_to_goal = (len(subgoal_images_path)-1) - int(image_subgoal_index)
            key = (os.path.split(image_t.split(self.split)[-1])[0], distance_to_goal)
            demos_organized[key] = image_t
            
        self.image_to_demo_mapping = {}
        for image_t in tqdm(self.images):
            image_folder = os.path.split(image_t)[0]
            history_folder = image_folder.replace(ORIGINAL_IMAGES_FOLDER, HISTORY_IMAGES_FOLDER)
            current_subgoal_index = int(image_t[-6:-4]) # get subgoal index
            subgoal_images_path = [d for d in glob.glob(image_folder+'/*')+glob.glob(history_folder+'/*')
                    if (
                        int(d[-6:-4])==current_subgoal_index 
                        )]
            subgoal_images_arg_idxs = np.argsort(np.array([int(os.path.split(d)[-1].split('_')[0]) for d in subgoal_images_path]))
            subgoal_images_path = [subgoal_images_path[d_i] for d_i in list(subgoal_images_arg_idxs)]
            goal_subgoal_image_path = subgoal_images_path[-1]
            goal_subgoal_targets_path = goal_subgoal_image_path.replace(ORIGINAL_IMAGES_FOLDER, TARGETS_FOLDER).replace('.png', '')
            image_subgoal_index = [image_t==im for im in subgoal_images_path]
            image_subgoal_index = np.where(image_subgoal_index)[0]
            distance_to_goal = (len(subgoal_images_path)-1) - int(image_subgoal_index)
            key = (os.path.split(image_t.split(self.split)[-1])[0], distance_to_goal)
            if key in demos_organized:
                self.image_to_demo_mapping[image_t] = demos_organized[key]
            else:
                # this means that augmented generation failed for this episode, so just take unaugmented
                self.image_to_demo_mapping[image_t] = image_t
        del demos_organized

    def format_subgoal(
        self,
        subgoal
        ):
        # format goal instruction
        subgoal = f'{self.idx2actions[subgoal[0]].replace("Object", "")} the {self.id_to_name[subgoal[1]]}.'
        subgoal = subgoal.replace('ToggleOn', 'Turn On')
        subgoal = subgoal.replace('ToggleOff', 'Turn Off')
        subgoal = subgoal.replace('Pickup', 'Pick Up')
        if self.name_to_mapped_name is not None and self.map_names:
            name_to_mapped_name_keys = list(self.name_to_mapped_name.keys())
            subgoal = subgoal.lower().replace('.','').split()
            for word_i in range(len(subgoal)):
                if subgoal[word_i] in name_to_mapped_name_keys:
                    subgoal[word_i] = self.name_to_mapped_name[subgoal[word_i]]
            subgoal = ' '.join(word for word in subgoal) + '.'
        return subgoal

    def transform_images(self, images):
        return (images - self.image_mean) / self.image_std

    def process_agent_position(self, agent_position, add_camera_height=True):
        agent_position = agent_position[[0,1,2,4,6]] # extract x, y & z position, yaw, and pitch
        if add_camera_height: #np.isclose(agent_position[1], 0.909, atol=1e-1):
            agent_position[1] += 0.675 # camera height is above agent height by 0.675
        return agent_position

    def __getitem__(self, idx):

        ###########%%%%%%%%% get paths of current obs and history ###########%%%%%%%%%
        image_t = self.images[idx]
        self.image_t = image_t
        image_episode_idx = int(os.path.split(image_t)[-1].split('_')[0])
        image_folder = os.path.split(image_t)[0]
        history_folder = image_folder.replace(ORIGINAL_IMAGES_FOLDER, HISTORY_IMAGES_FOLDER)
        history_images_path = [d for d in glob.glob(image_folder+'/*')+glob.glob(history_folder+'/*')
            if (
                int(os.path.split(d)[-1].split('_')[0])<image_episode_idx 
                and (image_episode_idx-int(os.path.split(d)[-1].split('_')[0]))<self.max_image_history
                )]
        
        history_segmentations_path = [d.replace(ORIGINAL_IMAGES_FOLDER, INSTANCE_MASKS_FOLDER) for d in history_images_path]
        history_segmentations_path = [d.replace(HISTORY_IMAGES_FOLDER, INSTANCE_MASKS_FOLDER) for d in history_segmentations_path]
        segmentation_image_path = image_t.replace(ORIGINAL_IMAGES_FOLDER, INSTANCE_MASKS_FOLDER)
        history_targets_path = [d.replace(ORIGINAL_IMAGES_FOLDER, TARGETS_FOLDER).replace('.png', '') for d in history_images_path]
        history_targets_path = [d.replace(HISTORY_IMAGES_FOLDER, TARGETS_FOLDER).replace('.png', '') for d in history_targets_path]
        targets_path = image_t.replace(ORIGINAL_IMAGES_FOLDER, TARGETS_FOLDER).replace('.png', '')
        history_idxs = np.array([int(os.path.split(d)[-1].split('_')[0]) for d in history_images_path])
        history_arg_idxs = np.argsort(history_idxs)

        targets_path, tag = os.path.split(targets_path)
        # targets_episode = np.load(targets_path+'/targets.npz', mmap_mode='r')
        targets_episode = h5py.File(targets_path+'/targets.hdf5', 'r')

        # get langauge goal instruction
        subgoal = targets_episode[tag+"_subgoal"][()]
        target_object = subgoal[1] #self.id_to_name[subgoal[1]]
        subgoal = self.format_subgoal(subgoal)

        ###########%%%%%%%%% get history data ###########%%%%%%%%%
        targets_to_get = ['agent_pose', 'obj_centroids']
        if self.use_GT_intermediate_objs:
            # obj_targets only work for current frame
            targets_to_get.append('labels')
        if self.do_feature_cloud_history:
            targets_to_get.extend(['depth', 'camX0_T_camX'])
        if self.use_object_tracklets:
            targets_to_get.append('obj_instance_ids')
        current_position = self.process_agent_position(targets_episode[tag+"_agent_pose"][()])
        # current_position = current_position[[0,1,2,4,6]] # extract x, y & z position, yaw, and pitch
        '''
        Assume relative pose is current position of agent if head were at 0 pitch
        '''
        origin_T_camX0 = utils.aithor.get_origin_T_camX_from_xyz_rot(
            np.array([current_position[0], current_position[1], current_position[2]]), 
            current_position[3], 
            0.0, # pitch relative to zero
            add_camera_height=False
            )
        camX0_T_origin = utils.geom.safe_inverse_single(origin_T_camX0)
        
        images, boxes, interms, _, positions, obj_centroids, obj_instance_ids, _, depth_maps, valid = self.get_history_data(
            history_arg_idxs,
            history_images_path,
            history_segmentations_path,
            history_targets_path,
            targets_to_get,
            max_steps=self.max_image_history-1,
            camX0_T_origin=camX0_T_origin, # get history positions relative to current position
            # tag=tag,
            targets_episode=targets_episode,
            target_object=target_object,
        )

        targets = {}

        # ##### current observation is last index #####
        images.append(np.asarray(Image.open(image_t)))
        segm = np.asarray(Image.open(segmentation_image_path))

        mask_colors = targets_episode[tag+"_masks"][()]
        mask, mask_nopad = get_masks_from_seg(
                        segm, 
                        mask_colors,
                        self.W,self.H,
                        self.max_objs,
                        add_whole_image_mask=self.add_whole_image_mask
                        )

        targets["masks"] = mask_nopad

        for target in targets_to_output:
            if target in ['masks', 'subgoal_percent_complete', 'goal_pose']:
                continue
            data = targets_episode[tag+f'_{target}'][()]
            if target in ['labels', 'obj_targets', 'obj_instance_ids', 'expert_action', 'obj_targets_cat']:
                data = data.astype(np.int64)
            if len(data.shape)==0:
                data = np.asarray([data])
            targets[target] = data

        if self.roi_pool or self.roi_align:
            boxes_ = targets["boxes"]
            if self.add_whole_image_mask:
                # add full image box
                boxes_ = np.concatenate([np.array([[0.5, 0.5, 1., 1.]]), boxes_], axis=0)
            boxes_ = boxes_[:self.max_objs] # clip at max objs
            # pad masks to max objects
            npad = ((0, self.max_objs-boxes_.shape[0]), (0, 0))
            boxes_ = np.pad(boxes_, pad_width=npad, mode='constant', constant_values=False)
            boxes.append(boxes_)

        if self.use_GT_intermediate_objs:
            # can use obj_targets sine it is current frame
            interm = np.zeros(self.max_objs, dtype=bool)
            interm_idxs = targets['obj_targets'].copy()
            if self.add_whole_image_mask:
                interm_idxs += 1 # add one since first index is full image mask
            interm_idxs = interm_idxs[interm_idxs < self.max_objs] # anything larger than max_objs has been removed in get_masks_from_seg
            interm[interm_idxs] = True
            interms.append(interm)
            interms = np.stack(interms, axis=0)

        depth_path = image_t.replace(ORIGINAL_IMAGES_FOLDER, DEPTH_IMAGES_FOLDER)
        depth = np.asarray(Image.open(depth_path)) * (10/255) # convert to meters

        if self.do_feature_cloud_history:
            depth_maps.append(depth)
            depth_maps = np.stack(depth_maps, axis=0)
            depth = depth_maps

        if self.only_decode_intermediate:
            # filter out only the intermediate objects
            interm_idxs = targets['obj_targets']
            targets['masks'] = targets['masks'][interm_idxs]
            targets['boxes'] = targets['boxes'][interm_idxs]
            targets['labels'] = targets['labels'][interm_idxs]
            if len(interm_idxs)>0:
                targets['obj_targets'] = np.arange(len(interm_idxs))

        if self.do_planner_path:
            planner_path = targets_episode[tag+"_planner_paths"][()]

        assert targets['masks'].shape[0]==targets['boxes'].shape[0]==targets['labels'].shape[0]

        images = np.asarray(images) * 1./255
        images = images.transpose(0, 3, 1, 2)
        images = self.transform_images(images).astype(np.float32) # normalize for resnet
        boxes = np.stack(boxes, axis=0)

        positions.append(np.concatenate([np.zeros(4), np.radians(current_position[-1:])],axis=0))

        positions = np.stack(positions, axis=0)
        if self.get_centroids_from_masks:
            '''
            This unprojects the mask, aligns it to current position and extracts median point as centroid
            '''
            depth_ = torch.from_numpy(depth).unsqueeze(0).unsqueeze(0).float()
            xyz = utils.geom.depth2pointcloud(depth_, torch.from_numpy(self.pix_T_camX).unsqueeze(0).float(), device='cpu')
            obj_centroid = utils.aithor.get_centroids_from_masks(
                current_position[-1], 
                xyz, 
                mask_nopad, 
                self.W, self.H,
                camX0_T_camX=None # xyz already in correct reference frame
                )
        else:
            # Use GT centroids and align to current frame
            obj_centroid = targets_episode[tag+'_obj_centroids'][()]
            obj_centroid = utils.geom.apply_4x4(camX0_T_origin.unsqueeze(0), torch.from_numpy(obj_centroid).unsqueeze(0)).cpu().numpy().squeeze(0)
            # st()
            # origin_T_camX = utils.aithor.get_origin_T_camX_from_xyz_rot_batch(
            #     obj_centroid, 
            #     np.zeros(len(obj_centroid)), # zero rotation for objs
            #     np.zeros(len(obj_centroid)) # zero rotation for objs
            #     )
            # camX0_T_camX = torch.bmm(camX0_T_origin.unsqueeze(0).repeat(len(obj_centroid),1,1), origin_T_camX)
            # _, trans = utils.geom.split_rt(camX0_T_camX)
            # obj_centroid = trans.numpy()

            if self.add_whole_image_mask:
                obj_centroid = obj_centroid[:self.max_objs-1]
                npad = ((1, self.max_objs-obj_centroid.shape[0]-1), (0, 0))
            else:
                obj_centroid = obj_centroid[:self.max_objs]
                npad = ((0, self.max_objs-obj_centroid.shape[0]), (0, 0))
            obj_centroid = np.pad(obj_centroid, pad_width=npad, mode='constant', constant_values=-1)
            obj_centroids.append(obj_centroid)
            obj_centroids = np.stack(obj_centroids, axis=0)

        if self.use_object_tracklets:
            obj_instance_id = targets['obj_instance_ids']
            if self.add_whole_image_mask:
                obj_instance_id = obj_instance_id[:self.max_objs-1]
                npad = ((1, self.max_objs-obj_instance_id.shape[0]-1))
            else:
                obj_instance_id = obj_instance_id[:self.max_objs]
                npad = ((0, self.max_objs-obj_instance_id.shape[0]))
            obj_instance_id = np.pad(obj_instance_id, pad_width=npad, mode='constant', constant_values=-1)
            obj_instance_ids.append(obj_instance_id)
            obj_instance_ids = np.stack(obj_instance_ids, axis=0)

        samples = {}

        if self.demo_type is not None:
            targets_episode_demo = None
            if self.demo_type=="same_demo":
                targets_episode_demo = targets_episode
            demos = self.fetch_demos(targets_episode=targets_episode_demo)
            samples["demos"] = demos

        # TODO: Add augmentations to images
        targets_episode.close()      

        trial = image_t.split('/')[-3]
        subgoal_idx = image_t.split('_subgoal')[1].split('.')[0]
        samples["episode_tag"] = (trial, subgoal_idx)
        
        samples["images"] = images
        samples["boxes"] = boxes
        samples["targets"] = targets
        samples["subgoal"] = subgoal
        samples["positions"] = positions
        samples["depth"] = depth
        samples["obj_centroids"] = obj_centroids
        samples["valid"] = valid
        if self.use_GT_intermediate_objs:
            samples["obj_interms"] = interms
        if self.use_object_tracklets:
            samples["obj_instance_ids"] = obj_instance_ids
        if self.do_planner_path:
            samples["planner_path"] = planner_path
        return samples

    def get_history_data(
        self,
        arg_idxs,
        images_path,
        segs_path,
        targets_path,
        targets_to_get,
        max_steps,
        camX0_T_origin,
        targets_episode=None,
        target_object=None,
    ):  
        images = []
        masks = []
        boxes = []
        labels = []
        interms = []
        actions = []
        positions = []
        obj_centroids = []
        obj_instance_ids = []
        depth_maps = []

        # pad observations if fewer history than max steps
        full_inds = np.arange(max_steps-len(arg_idxs))
        for idx in list(full_inds):
            image = np.zeros((self.W, self.H, 3))
            if self.roi_pool or self.roi_align:
                box = np.zeros((self.max_objs, 4))
            images.append(image)
            boxes.append(box)
            obj_centroids.append(np.ones((self.max_objs, 3), dtype=np.float)*-1)
            positions.append(np.zeros(5))
            if self.use_GT_intermediate_objs:
                interms.append(np.zeros(self.max_objs, dtype=bool))
            if 'depth' in targets_to_get:
                depth_maps.append(np.zeros((self.W, self.H)))
            if 'obj_instance_ids' in targets_to_get:
                obj_instance_ids.append(np.ones(self.max_objs, dtype=np.int32)*-1)
        
        valid = np.ones(max_steps+1, dtype=bool)
        valid[full_inds] = False

        for idx in list(arg_idxs):
            '''
            Loop over history and demo indices. 
            Extract (1) images, (2) masks or boxes, (3) actions, (4) (if demo) intermediate object labels
            '''

            image_path = images_path[idx]
            seg_path = segs_path[idx]
            target_path = targets_path[idx]
            _, tag = os.path.split(target_path)

            image = np.asarray(Image.open(image_path))
            segm = np.asarray(Image.open(seg_path))
            if 'depth' in targets_to_get or self.get_centroids_from_masks:
                depth_path = image_path.replace(ORIGINAL_IMAGES_FOLDER, DEPTH_IMAGES_FOLDER)
                depth_path = depth_path.replace(HISTORY_IMAGES_FOLDER, DEPTH_IMAGES_FOLDER)
                depth = np.asarray(Image.open(depth_path)) * (10/255) # convert to meters
                depth_maps.append(depth)

            if self.get_centroids_from_masks:
                # depth_path = image_path.replace(ORIGINAL_IMAGES_FOLDER, DEPTH_IMAGES_FOLDER)
                # depth = np.asarray(Image.open(depth_path)) * (10/255) # convert to meters
                depth_ = torch.from_numpy(depth).unsqueeze(0).unsqueeze(0).float()
                xyz = utils.geom.depth2pointcloud(depth_, torch.from_numpy(self.pix_T_camX).unsqueeze(0).float(), device='cpu')
        
            # get action history and (for demo) intermediate object 
            targets = {}
            for target in targets_to_get: 
                if target in ['depth', 'camX0_T_camX']:
                    continue
                data = targets_episode[tag+f'_{target}'][()]
                if target in ['labels', 'obj_targets', 'obj_instance_ids', 'expert_action']:
                    data = data.astype(np.int64)
                if len(data.shape)==0:
                    data = np.asarray([data])

                if target in ['agent_pose'] and camX0_T_origin is not None:
                    data = self.process_agent_position(data) #[[0,1,2,4,6]] # extract x, y & z position, yaw, and pitch
                    '''
                    Assume relative pose is current position of agent if head were at 0 pitch
                    '''
                    
                    origin_T_camX = utils.aithor.get_origin_T_camX_from_xyz_rot(
                        np.array([data[0], data[1], data[2]]), # xyz
                        data[3], # yaw
                        0.0, # pitch - invert since aithor convention = positive is down
                        add_camera_height=False
                        )
                    camX0_T_camX = torch.matmul(camX0_T_origin, origin_T_camX)
                    rot, trans = utils.geom.split_rt_single(camX0_T_camX)

                    rot_mat =  Rotation.from_matrix(rot)
                    ry, rx, rz = rot_mat.as_euler("yxz",degrees=False)

                    data = np.array([trans[0], trans[1], trans[2], ry, np.radians(data[-1])])

                targets[target] = data
            
            # # get object boxes
            if self.roi_pool or self.roi_align:
                boxes_ = targets_episode[tag+'_boxes'][()]
                if self.add_whole_image_mask:
                    # add full image box
                    boxes_ = np.concatenate([np.array([[0.5, 0.5, 1., 1.]]), boxes_], axis=0)
                boxes_ = boxes_[:self.max_objs] # clip at max objs
                npad = ((0, self.max_objs-boxes_.shape[0]), (0, 0))
                boxes_ = np.pad(boxes_, pad_width=npad, mode='constant', constant_values=False)
                boxes.append(boxes_)
            
            if self.get_centroids_from_masks:
                mask_colors = targets_episode[tag+'_masks'][()]

                filter_inds = None
                
                mask, mask_nopad = get_masks_from_seg(
                            segm, 
                            mask_colors,
                            self.W,self.H,
                            self.max_objs,
                            add_whole_image_mask=self.add_whole_image_mask,
                            filter_inds=filter_inds
                        )

            if self.use_GT_intermediate_objs:
                labels = targets['labels'][()]
                interm = np.zeros(self.max_objs, dtype=bool)
                interm_idxs = np.where(labels==target_object)[0]
                # interm_idxs = targets['obj_targets']
                if self.add_whole_image_mask:
                    interm_idxs += 1 # add one since first index is full image mask
                interm_idxs = interm_idxs[interm_idxs<self.max_objs] # anything larger than max_objs has been removed in get_masks_from_seg
                interm[interm_idxs] = True
                interms.append(interm)

            # if 'labels' in targets.keys():
            #     label = targets['labels']
            #     if self.add_whole_image_mask:
            #         label = label[:self.max_objs-1]
            #         npad = ((1, self.max_objs-label.shape[0]-1))
            #     label = np.pad(label, pad_width=npad, mode='constant', constant_values=-1)
            #     labels.append(label)
            
            images.append(image)
            if self.get_centroids_from_masks:
                '''
                This unprojects the mask, aligns it to current position and extracts median point as centroid
                '''
                obj_centroid = utils.aithor.get_centroids_from_masks(
                    targets['agent_pose'][-1], 
                    xyz, 
                    mask_nopad, 
                    self.W, self.H, 
                    camX0_T_camX=camX0_T_camX
                    )
            else:
                # Use GT centroid and align to current position
                obj_centroid = targets['obj_centroids'][()]
                # origin_T_camX = utils.aithor.get_origin_T_camX_from_xyz_rot_batch(
                #     obj_centroid, 
                #     np.zeros(len(obj_centroid)), # zero rotation for objs
                #     np.zeros(len(obj_centroid)) # zero rotation for objs
                #     )
                # obj_centroids_camX0_T_camX = torch.bmm(camX0_T_origin.unsqueeze(0).repeat(len(obj_centroid),1,1), origin_T_camX)
                # _, trans = utils.geom.split_rt(obj_centroids_camX0_T_camX)
                # obj_centroid = trans.numpy()
                obj_centroid = utils.geom.apply_4x4(camX0_T_origin.unsqueeze(0), torch.from_numpy(obj_centroid).unsqueeze(0)).cpu().numpy().squeeze(0)

                if self.add_whole_image_mask:
                    obj_centroid = obj_centroid[:self.max_objs-1]
                    npad = ((1, self.max_objs-obj_centroid.shape[0]-1), (0, 0)) # reserve first index for full image features
                obj_centroid = np.pad(obj_centroid, pad_width=npad, mode='constant', constant_values=-1)
                obj_centroids.append(obj_centroid)

            # check if agent height is 0.900999 or 1.5759992599487305
            positions.append(targets['agent_pose'])
            
            if 'obj_instance_ids' in targets.keys():
                obj_instance_id = targets['obj_instance_ids'][()]
                if self.add_whole_image_mask:
                    obj_instance_id = obj_instance_id[:self.max_objs-1]
                    npad = ((1, self.max_objs-obj_instance_id.shape[0]-1))
                else:
                    obj_instance_id = obj_instance_id[:self.max_objs]
                    npad = ((0, self.max_objs-obj_instance_id.shape[0]))
                obj_instance_id = np.pad(obj_instance_id, pad_width=npad, mode='constant', constant_values=-1)
                obj_instance_ids.append(obj_instance_id)
        
        return images, boxes, interms, actions, positions, obj_centroids, obj_instance_ids, labels, depth_maps, valid

    # def fetch_demos(
    #     self,
    #     targets_episode=None,
    #     ):

    #     demo = {
    #         "images": [], "boxes":[], "positions": [],
    #         "depth": [], "obj_centroids": [], "keyframe_idxs": [],
    #         "actions": [], "intermediate_obj_idxs": []
    #     }
    #     if self.use_GT_tracklets:
    #         demo["obj_tracklets"] = []

    #     if self.demo_type=="same_demo":
    #         topk_keyframes = [self.image_t] # if topk_keyframes == 1, only using 1 demo, is > 1 then using multiple keyframes
    #         if targets_episode is None:
    #             assert(False) # same demo should use targets episode from loader
    #     elif self.demo_type=="augmented_same":
    #         topk_keyframes = [self.image_to_demo_mapping[self.image_t]] # if topk_keyframes == 1, only using 1 demo, is > 1 then using multiple keyframes
    #         targets_path = topk_keyframes[0].replace(ORIGINAL_IMAGES_FOLDER, TARGETS_FOLDER).replace('.png', '')
    #         targets_path, tag = os.path.split(targets_path)
    #         targets_episode = h5py.File(targets_path+'/targets.hdf5', 'r')
    #     else:
    #         raise NotImplementedError # only supports within demo currently                

    #     ###########%%%%%%%%% get retrieved demo(s) data ###########%%%%%%%%%
    #     for image_d_idx in range(len(topk_keyframes)):
    #         image_d = topk_keyframes[image_d_idx]
    #         demo_targets_path = image_d.replace(ORIGINAL_IMAGES_FOLDER, TARGETS_FOLDER).replace('.png', '')
    #         image_folder_d = os.path.split(image_d)[0]
    #         subgoal_tag = image_d.split('_')[-1].split('.')[0]
    #         demo_images_path = [d for d in glob.glob(image_folder_d+'/*') if subgoal_tag in d]

    #         history_folder_d = image_folder_d.replace(ORIGINAL_IMAGES_FOLDER, HISTORY_IMAGES_FOLDER)
    #         current_subgoal_index = int(image_d[-6:-4]) # get subgoal index
    #         demo_images_path = [d for d in glob.glob(image_folder_d+'/*')+glob.glob(history_folder_d+'/*')
    #             if (
    #                 int(d[-6:-4])==current_subgoal_index 
    #                 )]
    #         demo_images_arg_idxs = np.argsort(np.array([int(os.path.split(d)[-1].split('_')[0]) for d in demo_images_path]))
    #         demo_images_path = [demo_images_path[d_i] for d_i in list(demo_images_arg_idxs)] # this has all images for the demo subgoal
    #         keyframe_idx = int(np.where([d==image_d for d in demo_images_path])[0])
    #         if self.max_demo_history is not None or self.max_demo_future is not None:
    #             # filter demo to take max history and future
    #             # total frames is max_demo_history + 1 (keyframe) + max_demo_future
    #             start_idx = keyframe_idx-self.max_demo_history
    #             end_idx = keyframe_idx+self.max_demo_future+1
    #             pad_history, pad_future = 0, 0
    #             if start_idx<0:
    #                 start_idx = 0
    #                 pad_history = -(keyframe_idx-self.max_demo_history)
    #             if keyframe_idx+self.max_demo_future+1>len(demo_images_path):
    #                 end_idx = len(demo_images_path)
    #                 pad_future = (keyframe_idx+self.max_demo_future+1)-len(demo_images_path)
    #             demo_images_path = demo_images_path[start_idx:end_idx]
    #             # recompute index
    #             keyframe_idx = int(np.where([d==image_d for d in demo_images_path])[0])
    #         demo_images_arg_idxs = np.arange(len(demo_images_path))
    #         demo_seg_path = [d.replace(ORIGINAL_IMAGES_FOLDER, INSTANCE_MASKS_FOLDER) for d in demo_images_path]
    #         demo_seg_path = [d.replace(HISTORY_IMAGES_FOLDER, INSTANCE_MASKS_FOLDER) for d in demo_seg_path]
    #         demo_targets_path = [d.replace(ORIGINAL_IMAGES_FOLDER, TARGETS_FOLDER).replace('.png', '') for d in demo_images_path]
    #         demo_targets_path = [d.replace(HISTORY_IMAGES_FOLDER, TARGETS_FOLDER).replace('.png', '') for d in demo_targets_path]
            
    #         ###########%%%%%%%%% get history data ###########%%%%%%%%%
    #         targets_path, tag_keyframe = os.path.split(demo_targets_path[keyframe_idx])
    #         targets_to_get = ['agent_pose', 'obj_centroids']
    #         if self.use_GT_tracklets:
    #             targets_to_get.append('obj_targets')
    #         keyframe_position = self.process_agent_position(targets_episode[tag_keyframe+"_agent_pose"])
    #         # keyframe_position = keyframe_position[[0,1,2,4,6]] # extract x, y & z position, yaw, and pitch for keyframe
    #         '''
    #         Assume relative pose is current position of agent if head were at 0 pitch
    #         '''
    #         origin_T_camX0 = utils.aithor.get_origin_T_camX_from_xyz_rot(
    #             np.array([keyframe_position[0], keyframe_position[1], keyframe_position[2]]), 
    #             keyframe_position[3], 
    #             0.0, # pitch relative to zero
    #             add_camera_height=True
    #             )
    #         camX0_T_origin = utils.geom.safe_inverse_single(origin_T_camX0)
            
    #         images, boxes, interms, _, positions, obj_centroids, _, _ = self.get_history_data(
    #             demo_images_arg_idxs,
    #             demo_images_path,
    #             demo_seg_path,
    #             demo_targets_path,
    #             targets_to_get,
    #             max_steps=len(demo_images_path),
    #             camX0_T_origin=camX0_T_origin, # get history positions relative to current position
    #             # tag=tag,
    #             targets_episode=targets_episode,
    #         )

    #         # pad history
    #         full_inds = np.arange(pad_history)
    #         for idx in list(full_inds):
    #             image = np.zeros((self.W, self.H, 3))
    #             if self.roi_pool or self.roi_align:
    #                 box = np.zeros((self.max_objs, 4))
    #             images.insert(0, image)
    #             boxes.insert(0, box)
    #             obj_centroids.insert(0, np.ones((self.max_objs, 3), dtype=np.float)*-1)
    #             positions.insert(0, np.zeros(5))
    #             if 'object_targets' in targets_to_get:
    #                 interms.insert(0, np.zeros(self.max_objs, dtype=bool))

    #         # pad future
    #         full_inds = np.arange(pad_future)
    #         for idx in list(full_inds):
    #             image = np.zeros((self.W, self.H, 3))
    #             if self.roi_pool or self.roi_align:
    #                 box = np.zeros((self.max_objs, 4))
    #             images.append(image)
    #             boxes.append(box)
    #             obj_centroids.append(np.ones((self.max_objs, 3), dtype=np.float)*-1)
    #             positions.append(np.zeros(5))
    #             if 'object_targets' in targets_to_get:
    #                 interms.append(np.zeros(self.max_objs, dtype=bool))

    #         keyframe_depth_path = image_d.replace(ORIGINAL_IMAGES_FOLDER, DEPTH_IMAGES_FOLDER)
    #         depth = np.asarray(Image.open(keyframe_depth_path)) * (10/255) # convert to meters

    #         images = np.asarray(images) * 1./255
    #         images = images.transpose(0, 3, 1, 2)
    #         images = self.transform_images(images).astype(np.float32) # normalize for resnet

    #         boxes = np.stack(boxes, axis=0)
    #         positions = np.stack(positions, axis=0)
    #         obj_centroids = np.stack(obj_centroids, axis=0)

    #         # actions taken in the memory
    #         action = targets_episode[tag_keyframe+'_expert_action'][()]

    #         # intermediate object idxs
    #         intermediate_obj_idxs = targets_episode[tag_keyframe+'_obj_targets'][()]
    #         if self.add_whole_image_mask:
    #             intermediate_obj_idxs += 1 # add one since first index is full image mask
    #         intermediate_obj_idxs = intermediate_obj_idxs[intermediate_obj_idxs < self.max_objs] # anything larger than max_objs has been removed in get_masks_from_seg
    #         if intermediate_obj_idxs.shape[0] == 0:
    #             intermediate_obj_idxs = np.array([-1])
    #         demo["images"].append(images)
    #         demo["boxes"].append(boxes)
    #         demo["positions"].append(positions)
    #         demo["depth"].append(depth)
    #         demo["obj_centroids"].append(obj_centroids)
    #         demo["keyframe_idxs"].append(keyframe_idx)
    #         demo["actions"].append(action)
    #         demo["intermediate_obj_idxs"].append(intermediate_obj_idxs)
    #         if self.use_GT_tracklets:
    #             interms = np.stack(interms, axis=0)
    #             demo["obj_tracklets"].append(interms)

    #     # each of these are KxNx...
    #     demo["images"] = np.stack(demo["images"], axis=0)
    #     demo["boxes"] = np.stack(demo["boxes"], axis=0)
    #     demo["positions"] = np.stack(demo["positions"], axis=0)
    #     demo["depth"] = np.stack(demo["depth"], axis=0)
    #     demo["obj_centroids"] = np.stack(demo["obj_centroids"], axis=0)
    #     demo["keyframe_idxs"] = np.stack(demo["keyframe_idxs"], axis=0)
    #     demo["intermediate_obj_idxs"] = np.stack(demo["intermediate_obj_idxs"], axis=0)
    #     demo["actions"] = np.stack(demo["actions"], axis=0)
    #     if self.use_GT_tracklets:
    #         demo["obj_tracklets"] = np.stack(demo["obj_tracklets"], axis=0)
    #     return demo

    def get_demo_given_info(self, json_dir, subgoal_idx, steps_from_goal):
        '''
        Used for evaluation only to get demo for model
        '''
        if self.demo_type is None:
            assert(False) # no demo type given

        if len(self.demo_mapping)==0:
            self.im_count = 0
            '''
            This defines a dict for each image (epsiode name, subgoal number, steps to subgoal completion)->image path
            '''
            print("Getting demo mapping (this may take some time)...")
            for im_i in tqdm(range(len(self.images))):
                image = self.images[im_i]
                tag = image.split('/')[-3]
                subgoal = int(image[-6:-4])
                image_folder = os.path.split(image)[0]
                history_folder = image_folder.replace(ORIGINAL_IMAGES_FOLDER, HISTORY_IMAGES_FOLDER)
                subgoal_images_path = [d for d in glob.glob(image_folder+'/*')+glob.glob(history_folder+'/*')
                    if (
                        int(d[-6:-4])==subgoal 
                        )]
                subgoal_images_arg_idxs = np.argsort(np.array([int(os.path.split(d)[-1].split('_')[0]) for d in subgoal_images_path]))
                subgoal_images_path = [subgoal_images_path[d_i] for d_i in list(subgoal_images_arg_idxs)]
                for idx_s in range(1, len(subgoal_images_path)+1):
                    if subgoal_images_path[-idx_s]==image:
                        steps_from_goal_image = idx_s-1
                        break
                key = (tag, subgoal, steps_from_goal_image)
                self.demo_mapping[key] = im_i
                self.root_images = self.images[0].split(self.split)[0] + self.split
        
        # Adjust mapping in case we spawned agent at a previous subgoal index than current
        def adjust_mapping(
            steps_from_goal,
            subgoal_idx,
            json_dir,
            root_images,
            split
            ):
            image_folder = root_images + json_dir.split(split)[-1] + ORIGINAL_IMAGES_FOLDER
            history_folder = image_folder.replace(ORIGINAL_IMAGES_FOLDER, HISTORY_IMAGES_FOLDER)
            subgoal_images_path = [d for d in glob.glob(image_folder+'/*')+glob.glob(history_folder+'/*')
                        if (
                            int(d[-6:-4])==subgoal_idx 
                            )]
            steps_from_goal_copy = steps_from_goal
            count = 0
            steps_from_goal_check = 10000
            while len(subgoal_images_path)<=steps_from_goal or steps_from_goal_check==1: # also check that it is not interaction index
                count += 1
                steps_from_goal += 2 # adjust for 'done' and interaction index
                subgoal_ = subgoal_idx-count
                if subgoal_==-1:
                    break
                subgoal_images_path_ = [d for d in glob.glob(image_folder+'/*')+glob.glob(history_folder+'/*')
                        if (
                            int(d[-6:-4])==subgoal_ 
                            )]
                subgoal_images_path += subgoal_images_path_
            if count>0:
                # adjust steps_from_goal to be from previous subgoal
                steps_from_goal = steps_from_goal - (len(subgoal_images_path) - len(subgoal_images_path_))
                subgoal_idx = subgoal_idx-count

            return steps_from_goal, subgoal_idx

        steps_from_goal, subgoal_idx = adjust_mapping(
            steps_from_goal,
            subgoal_idx,
            json_dir,
            self.root_images,
            self.split
            )
        tag = json_dir.split('/')[-2]
        im_i = self.demo_mapping[(tag, subgoal_idx, steps_from_goal)]
        image_t = self.images[im_i]
        self.image_t = image_t
        targets_path = image_t.replace(ORIGINAL_IMAGES_FOLDER, TARGETS_FOLDER).replace('.png', '')

        targets_path, _ = os.path.split(targets_path)
        targets_episode = h5py.File(targets_path+'/targets.hdf5', 'r')

        demos = self.fetch_demos(targets_episode=targets_episode)

        # image = (demos['images'][0,2] * self.image_std.squeeze(0) + self.image_mean.squeeze(0)) * 255
        # image = image.astype(np.int).transpose(1,2,0)
        # import matplotlib.pyplot as plt
        # plt.figure(1); plt.clf()
        # plt.imshow(image)
        # plt.savefig(f'data/images/vis_demos/test2_{self.im_count}.png')
        # print("Demo action:", self.idx2actions[int(demos['actions'])])
        # self.im_count += 1

        return demos

    def get_goal_image_given_info(self, json_dir, subgoal_idx):
        '''
        Used for evaluation only to get demo for model
        '''
        # if self.demo_type is None:
        #     assert(False) # no demo type given

        if len(self.demo_mapping)==0:
            self.im_count = 0
            '''
            This defines a dict for each image (epsiode name, subgoal number, steps to subgoal completion)->image path
            '''
            print("Getting demo mapping (this may take some time)...")
            for im_i in tqdm(range(len(self.images))):
                image = self.images[im_i]
                tag = image.split('/')[-3]
                subgoal = int(image[-6:-4])
                image_folder = os.path.split(image)[0]
                history_folder = image_folder.replace(ORIGINAL_IMAGES_FOLDER, HISTORY_IMAGES_FOLDER)
                subgoal_images_path = [d for d in glob.glob(image_folder+'/*')+glob.glob(history_folder+'/*')
                    if (
                        int(d[-6:-4])==subgoal 
                        )]
                subgoal_images_arg_idxs = np.argsort(np.array([int(os.path.split(d)[-1].split('_')[0]) for d in subgoal_images_path]))
                subgoal_images_path = [subgoal_images_path[d_i] for d_i in list(subgoal_images_arg_idxs)]
                # for idx_s in range(1, len(subgoal_images_path)+1):
                #     if subgoal_images_path[-idx_s]==image:
                #         steps_from_goal_image = idx_s-1
                #         break
                key = (tag, subgoal)
                self.demo_mapping[key] = subgoal_images_path[-1]
            self.root_images = self.images[0].split(self.split)[0] + self.split
        
        # # Adjust mapping in case we spawned agent at a previous subgoal index than current
        # _, subgoal_idx = adjust_mapping(
        #     steps_from_goal,
        #     subgoal_idx,
        #     json_dir,
        #     self.root_images,
        #     self.split
        #     )
        tag = json_dir.split('/')[-2]
        image_goal = self.demo_mapping[(tag, subgoal_idx)]
        image = np.asarray(Image.open(image_goal))

        # self.image_t = image_t
        # targets_path = image_t.replace(ORIGINAL_IMAGES_FOLDER, TARGETS_FOLDER).replace('.png', '')

        # targets_path, _ = os.path.split(targets_path)
        # targets_episode = h5py.File(targets_path+'/targets.hdf5', 'r')

        # demos = self.fetch_demos(targets_episode=targets_episode)

        # image = (demos['images'][0,2] * self.image_std.squeeze(0) + self.image_mean.squeeze(0)) * 255
        # image = image.astype(np.int).transpose(1,2,0)
        # import matplotlib.pyplot as plt
        # plt.figure(1); plt.clf()
        # plt.imshow(image)
        # plt.savefig(f'data/images/vis_demos/test2_{self.im_count}.png')
        # print("Demo action:", self.idx2actions[int(demos['actions'])])
        # self.im_count += 1

        return image

np_str_obj_array_pattern = re.compile(r'[SaUO]')

default_collate_err_msg_format = (
    "default_collate: batch must contain tensors, numpy arrays, numbers, "
    "dicts or lists; found {}")

def my_collate(batch):
    r"""Puts each data field into a tensor with outer dimension batch size"""

    elem = batch[0]
    elem_type = type(elem)
    if isinstance(elem, torch.Tensor):
        numel = sum([x.numel() for x in batch])
        storage = elem.storage()._new_shared(numel)
        out = elem.new(storage)
        it = iter(batch)
        elem_size = len(next(it))

        if not all(len(elem) == elem_size for elem in batch):
            out_ = batch
        else:
            out_ = torch.stack(batch, 0, out=out)   
        return out_
                
    elif elem_type.__module__ == 'numpy' and elem_type.__name__ != 'str_' \
            and elem_type.__name__ != 'string_':
        if elem_type.__name__ == 'ndarray' or elem_type.__name__ == 'memmap':
            # array of string classes and object
            if np_str_obj_array_pattern.search(elem.dtype.str) is not None:
                raise TypeError(my_collate_err_msg_format.format(elem.dtype))

            return my_collate([torch.as_tensor(b) for b in batch])
        elif elem.shape == ():  # scalars
            return torch.as_tensor(batch)
    elif isinstance(elem, float):
        return torch.tensor(batch, dtype=torch.float64)
    elif isinstance(elem, int_classes):
        return torch.tensor(batch)
    elif isinstance(elem, string_classes):
        return batch
    elif isinstance(elem, container_abcs.Mapping):
        if all(e in targets_to_output for e in list(elem.keys())):
            for t in batch:
                for k in t.keys():
                    t[k] = torch.as_tensor(t[k])
            return batch # used for targets 
        return {key: my_collate([d[key] for d in batch]) for key in elem}
    elif isinstance(elem, tuple) and hasattr(elem, '_fields'):  # namedtuple
        return elem_type(*(my_collate(samples) for samples in zip(*batch)))
    elif isinstance(elem, container_abcs.Sequence):
        # check to make sure that the elements in batch have consistent size
        it = iter(batch)
        elem_size = len(next(it))
        if not all(len(elem) == elem_size for elem in it):
            pass
            transposed = batch
        else:
            transposed = zip(*batch)
        return [my_collate(samples) for samples in transposed]

    raise TypeError(my_collate_err_msg_format.format(elem_type))

