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

import torch
import re
# from torch._six import container_abcs, string_classes, int_classes
try:
    from torch._six import container_abcs, string_classes, int_classes
except:
    import collections.abc as container_abcs
    int_classes = int
    string_classes = str
from SOLQ.util import box_ops


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
        root_dir, 
        W, H, 
        max_objs,
        idx2subgoals,
        idx2actions,
        id_to_name,
        cache=False,
        start_index=None, 
        end_index=None,
        shuffle=False,
        images=None,
        name_to_mapped_name=None,
        subsample=None,
        root_dir2=None,
        root_dir3=None,
        ):
        """
        Args:
            root_dir (string): root directory with all the images, etc.
            W, H (int): width and height of frame
            max_objs (int): maximum objects per image allowable
            idx2subgoals (dict): converts subgoal idx to word
            id_to_name (dict): converts object name idx to word
        """
        
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
        if end_index is not None:
            if start_index is None:
                start_index = 0
            self.images = self.images[start_index:end_index]
        if subsample is not None:
            self.images = self.images[::subsample]

        if args.train_num_steps_away is not None:
            self.images = self.get_images_X_steps_away(args.train_num_steps_away)


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

        self.use_action_weights = args.use_action_weights

        self.idx2actions, self.idx2subgoals, self.id_to_name = idx2actions, idx2subgoals, id_to_name
        self.add_whole_image_mask = args.add_whole_image_mask # add whole image mask to object history 
        self.only_decode_intermediate = args.only_decode_intermediate # only decode intermediate/interaction objects (versus all objects in view)
        self.roi_pool = args.roi_pool
        self.roi_align = args.roi_align
        self.name_to_mapped_name = name_to_mapped_name
        self.map_names = args.map_names
        self.max_image_history = args.max_image_history
        self.use_3d_pos_enc = args.use_3d_pos_enc
        self.use_action_ghost_nodes = args.use_action_ghost_nodes
        self.use_3d_obj_centroids = args.use_3d_obj_centroids
        self.use_3d_img_pos_encodings = args.use_3d_img_pos_encodings
        self.learned_3d_pos_enc = args.learned_3d_pos_enc
        self.get_centroids_from_masks = args.get_centroids_from_masks
        if self.use_3d_pos_enc:
            targets_to_output.append('agent_pose')
            if self.use_3d_obj_centroids:
                targets_to_output.append('obj_centroids')

        if args.remove_done:
            self.images = self.remove_done_images()

    def __len__(self):
        return len(self.images)

    def get_action_weights(
        self,
    ):
        '''
        Gets proportion of each action for cross entropy loss
        This should just be called once for new data when the weights are needed
        '''

        actions = {i:0 for i in range(13)}
        for idx in tqdm(range(len(self.images))):
            image_t = self.images[idx]
            targets_path = image_t.replace(ORIGINAL_IMAGES_FOLDER, TARGETS_FOLDER).replace('.png', '')
            data = int(np.load(targets_path+f'_expert_action.npy'))
            actions[data] += 1
            
        action_counts = np.array(list(actions.values()))
        action_weights = np.median(action_counts)/action_counts
        print("Action counts:", action_counts)
        print("Action weights:", action_weights)
        assert(False)
        return action_weights

    def get_images_X_steps_away(self, X):
        
        images = []
        for idx in range(len(self.images)):
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

        return images

    def remove_done_images(self):
        
        images = []
        for idx in range(len(self.images)):
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
            if not ((len(subgoal_images_path)-1) == int(image_subgoal_index)):
                images.append(image_t)
                if args.targets_npz_format:
                    targets_path = image_t.replace(ORIGINAL_IMAGES_FOLDER, TARGETS_FOLDER).replace('.png', '')
                    targets_path, tag = os.path.split(targets_path)
                    targets_episode = np.load(targets_path+'/targets.npz', mmap_mode='r')
                    data = targets_episode[tag+'_expert_action']
                    should_be_done = self.idx2actions[int(data)]
                    targets_episode.close()
                else:
                    targets_path = image_t.replace(ORIGINAL_IMAGES_FOLDER, TARGETS_FOLDER).replace('.png', '')
                    should_be_done = self.idx2actions[int(np.load(targets_path+'_expert_action.npy'))]
                if should_be_done=="Done":
                    print(image_subgoal_index, len(subgoal_images_path)-1, should_be_done)
            else:
                if args.targets_npz_format:
                    targets_path = image_t.replace(ORIGINAL_IMAGES_FOLDER, TARGETS_FOLDER).replace('.png', '')
                    targets_path, tag = os.path.split(targets_path)
                    targets_episode = np.load(targets_path+'/targets.npz', mmap_mode='r')
                    data = targets_episode[tag+'_expert_action']
                    should_be_done = self.idx2actions[int(data)]
                    targets_episode.close()
                else:
                    targets_path = image_t.replace(ORIGINAL_IMAGES_FOLDER, TARGETS_FOLDER).replace('.png', '')
                    should_be_done = self.idx2actions[int(np.load(targets_path+'_expert_action.npy'))]
                print(should_be_done)
                if should_be_done!="Done":
                    # TODO: last subgoal does not have done? fix this in datagen
                    images.append(image_t)

        return images

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

    def __getitem__(self, idx, only_fetch_demos=False, subgoal_info=None):

        ###########%%%%%%%%% get paths of current obs and history ###########%%%%%%%%%
        image_t = self.images[idx]
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
        history_targets_path = [d.replace(ORIGINAL_IMAGES_FOLDER, TARGETS_FOLDER).replace('.png', '.npy') for d in history_images_path]
        history_targets_path = [d.replace(HISTORY_IMAGES_FOLDER, TARGETS_FOLDER).replace('.png', '.npy') for d in history_targets_path]
        targets_path = image_t.replace(ORIGINAL_IMAGES_FOLDER, TARGETS_FOLDER).replace('.png', '')
        history_idxs = np.array([int(os.path.split(d)[-1].split('_')[0]) for d in history_images_path])
        history_arg_idxs = np.argsort(history_idxs)

        if args.targets_npz_format:
            targets_path, tag = os.path.split(targets_path)
            targets_episode = np.load(targets_path+'/targets.npz', mmap_mode='r')
        else:
            tag=None
            targets_episode=None

        ###########%%%%%%%%% get history data ###########%%%%%%%%%
        targets_to_get = ['expert_action']
        if self.use_3d_pos_enc:
            if self.learned_3d_pos_enc:
                targets_to_get.append('agent_pose')
                if args.targets_npz_format:
                    current_position = targets_episode[tag+"_agent_pose"]
                else:
                    current_position = np.load(targets_path+'_agent_pose.npy')
                current_position = current_position[[0,1,2,4,6]] # extract x, y & z position, yaw, and pitch
                '''
                Assume relative pose is current position of agent if head were at 0 pitch
                '''
                origin_T_camX0 = utils.aithor.get_origin_T_camX_from_xyz_rot(
                    np.array([current_position[0], current_position[1], current_position[2]]), 
                    current_position[3], 
                    0.0, # pitch relative to zero
                    add_camera_height=True
                    )
                camX0_T_origin = utils.geom.safe_inverse_single(origin_T_camX0)
            else:
                assert(False)
        else:
            camX0_T_origin = None
            current_position = None
        
        if self.use_3d_obj_centroids:
            targets_to_get.append('obj_centroids')
        images, boxes, _, _, positions, obj_centroids, _, _ = self.get_history_data(
            history_arg_idxs,
            history_images_path,
            history_segmentations_path,
            history_targets_path,
            targets_to_get,
            max_steps=self.max_image_history-1,
            camX0_T_origin=camX0_T_origin, # get history positions relative to current position
            tag=tag,
            targets_episode=targets_episode,
        )

        targets = {}

        # ##### current observation is last index #####
        images.append(np.asarray(Image.open(image_t)))
        segm = np.asarray(Image.open(segmentation_image_path))

        if args.targets_npz_format:
            mask_colors = targets_episode[tag+"_masks"]
        else:
            mask_path = targets_path+'_masks.npy'
            mask_colors = np.load(mask_path)
        mask, mask_nopad = get_masks_from_seg(
                        segm, 
                        mask_colors,
                        self.W,self.H,
                        self.max_objs,
                        add_whole_image_mask=self.add_whole_image_mask
                        )

        targets["masks"] = mask_nopad

        # get langauge goal instruction
        if args.targets_npz_format:
            subgoal = targets_episode[tag+"_subgoal"]
        else:
            subgoal = np.load(targets_path+'_subgoal.npy')
        subgoal = self.format_subgoal(subgoal)

        for target in targets_to_output:
            if target in ['masks', 'subgoal_percent_complete', 'goal_pose']:
                continue
            if args.targets_npz_format:
                data = targets_episode[tag+f'_{target}']
            else:
                data = np.load(targets_path+f'_{target}.npy')
            if target in ['labels', 'obj_targets', 'obj_instance_ids', 'expert_action']:
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

        if self.use_3d_img_pos_encodings:
            depth_path = image_t.replace(ORIGINAL_IMAGES_FOLDER, DEPTH_IMAGES_FOLDER)
            depth = np.asarray(Image.open(depth_path)) * (10/255) # convert to meters

        if self.only_decode_intermediate:
            # filter out only the intermediate objects
            interm_idxs = targets['obj_targets']
            targets['masks'] = targets['masks'][interm_idxs]
            targets['boxes'] = targets['boxes'][interm_idxs]
            targets['labels'] = targets['labels'][interm_idxs]
            if len(interm_idxs)>0:
                targets['obj_targets'] = np.arange(len(interm_idxs))

        assert targets['masks'].shape[0]==targets['boxes'].shape[0]==targets['labels'].shape[0]

        images = np.asarray(images) * 1./255
        images = images.transpose(0, 3, 1, 2)
        images = self.transform_images(images).astype(np.float32) # normalize for resnet
        boxes = np.stack(boxes, axis=0)
        # actions = np.concatenate(actions, axis=0)
        if self.use_3d_pos_enc:
            
            if self.use_action_ghost_nodes:
                if self.learned_3d_pos_enc:
                    positions.append(np.concatenate([np.zeros(4), np.radians(current_position[-1:])],axis=0)) 

            positions = np.stack(positions, axis=0)
            if self.use_3d_obj_centroids:    
                if self.learned_3d_pos_enc:
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
                        if args.targets_npz_format:
                            obj_centroid = targets_episode[tag+'_obj_centroids']
                        else:
                            obj_centroid = np.load(targets_path+'_obj_centroids.npy')
                        origin_T_camX = utils.aithor.get_origin_T_camX_from_xyz_rot_batch(
                            obj_centroid, 
                            np.zeros(len(obj_centroid)), # zero rotation for objs
                            np.zeros(len(obj_centroid)) # zero rotation for objs
                            )
                        camX0_T_camX = torch.bmm(camX0_T_origin.unsqueeze(0).repeat(len(obj_centroid),1,1), origin_T_camX)
                        _, trans = utils.geom.split_rt(camX0_T_camX)
                        obj_centroid = trans.numpy()
                else:            
                    assert(False)

                if self.add_whole_image_mask:
                    obj_centroid = obj_centroid[:self.max_objs-1]
                    npad = ((1, self.max_objs-obj_centroid.shape[0]-1), (0, 0))
                else:
                    obj_centroid = obj_centroid[:self.max_objs]
                    npad = ((0, self.max_objs-obj_centroid.shape[0]), (0, 0))
                obj_centroid = np.pad(obj_centroid, pad_width=npad, mode='constant', constant_values=-1)
                obj_centroids.append(obj_centroid)
                obj_centroids = np.stack(obj_centroids, axis=0)

        # TODO: Add augmentations to images

        if args.targets_npz_format:
            targets_episode.close()
        
        samples = {}
        samples["images"] = images
        samples["boxes"] = boxes
        # samples["mask_idxs"] = mask_idxs
        samples["targets"] = targets
        samples["subgoal"] = subgoal
        if self.use_3d_pos_enc:
            samples["positions"] = positions
        # samples["action_history"] = actions
        if self.use_3d_img_pos_encodings:
            samples["depth"] = depth
        if self.use_3d_obj_centroids:
            samples["obj_centroids"] = obj_centroids
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
        tag=None,
        targets_episode=None,
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

        # pad observations if fewer history than max steps
        full_inds = np.arange(max_steps-len(arg_idxs))
        for idx in list(full_inds):
            image = np.zeros((self.W, self.H, 3))
            if self.roi_pool or self.roi_align:
                box = np.zeros((self.max_objs, 4))
            images.append(image)
            boxes.append(box)
            if self.learned_3d_pos_enc:
                obj_centroids.append(np.ones((self.max_objs, 3), dtype=np.float)*-1)
                positions.append(np.zeros(5))

        for idx in list(arg_idxs):
            '''
            Loop over history and demo indices. 
            Extract (1) images, (2) masks or boxes, (3) actions, (4) (if demo) intermediate object labels
            '''

            image_path = images_path[idx]
            seg_path = segs_path[idx]
            target_path = targets_path[idx]

            image = np.asarray(Image.open(image_path))
            segm = np.asarray(Image.open(seg_path))

            if self.get_centroids_from_masks:
                depth_path = image_path.replace(ORIGINAL_IMAGES_FOLDER, DEPTH_IMAGES_FOLDER)
                depth = np.asarray(Image.open(depth_path)) * (10/255) # convert to meters
                depth_ = torch.from_numpy(depth).unsqueeze(0).unsqueeze(0).float()
                xyz = utils.geom.depth2pointcloud(depth_, torch.from_numpy(self.pix_T_camX).unsqueeze(0).float(), device='cpu')
        
            # get action history and (for demo) intermediate object 
            targets = {}
            for target in targets_to_get: 
                if args.targets_npz_format:
                    data = targets_episode[tag+f'_{target}']
                else:
                    data = np.load(target_path.replace('.npy', '')+f'_{target}.npy')
                if target in ['labels', 'obj_targets', 'obj_instance_ids', 'expert_action']:
                    data = data.astype(np.int64)
                if len(data.shape)==0:
                    data = np.asarray([data])

                if target in ['agent_pose'] and camX0_T_origin is not None:
                    if self.learned_3d_pos_enc:
                        data = data[[0,1,2,4,6]] # extract x, y & z position, yaw, and pitch
                        '''
                        Assume relative pose is current position of agent if head were at 0 pitch
                        '''
                        origin_T_camX = utils.aithor.get_origin_T_camX_from_xyz_rot(
                            np.array([data[0], data[1], data[2]]), 
                            data[3], 
                            0.0, 
                            add_camera_height=True
                            )
                        camX0_T_camX = torch.matmul(camX0_T_origin, origin_T_camX)
                        rot, trans = utils.geom.split_rt_single(camX0_T_camX)
                        rx, ry, rz = utils.geom.rotm2eul(rot.unsqueeze(0))
                        data = torch.tensor([trans[0], trans[1], trans[2], ry, np.radians(data[-1])]).numpy()
                    else:
                        assert(False)

                targets[target] = data
            
            # # get object masks from segmentation image
            if self.roi_pool or self.roi_align:
                if args.targets_npz_format:
                    boxes_ = targets_episode[tag+'_boxes']
                else:
                    box_path = target_path.replace('.npy', '')+'_boxes.npy'
                    boxes_ = np.load(box_path)
                if self.add_whole_image_mask:
                    # add full image box
                    boxes_ = np.concatenate([np.array([[0.5, 0.5, 1., 1.]]), boxes_], axis=0)
                boxes_ = boxes_[:self.max_objs] # clip at max objs
                npad = ((0, self.max_objs-boxes_.shape[0]), (0, 0))
                boxes_ = np.pad(boxes_, pad_width=npad, mode='constant', constant_values=False)
                boxes.append(boxes_)
            
            if self.get_centroids_from_masks:
                if args.targets_npz_format:
                    mask_colors = targets_episode[tag+'_masks']
                else:
                    mask_path = target_path.replace('.npy', '')+'_masks.npy'
                    mask_colors = np.load(mask_path)

                filter_inds = None
                
                mask, mask_nopad = get_masks_from_seg(
                            segm, 
                            mask_colors,
                            self.W,self.H,
                            self.max_objs,
                            add_whole_image_mask=self.add_whole_image_mask,
                            filter_inds=filter_inds
                        )

            if 'obj_targets' in targets.keys():
                interm = np.zeros(self.max_objs, dtype=bool)
                interm_idxs = targets['obj_targets']
                if self.add_whole_image_mask:
                    interm_idxs += 1 # add one since first index is full image mask
                interm_idxs = interm_idxs[interm_idxs<self.max_objs] # anything larger than max_objs has been removed in get_masks_from_seg
                interm[interm_idxs] = True
                interms.append(interm)

            if 'labels' in targets.keys():
                label = targets['labels']
                if self.add_whole_image_mask:
                    label = label[:self.max_objs-1]
                    npad = ((1, self.max_objs-label.shape[0]-1))
                label = np.pad(label, pad_width=npad, mode='constant', constant_values=-1)
                labels.append(label)
            
            images.append(image)
            if self.use_3d_obj_centroids:  
                if self.learned_3d_pos_enc:
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
                        obj_centroid = targets['obj_centroids']
                        origin_T_camX = utils.aithor.get_origin_T_camX_from_xyz_rot_batch(
                            obj_centroid, 
                            np.zeros(len(obj_centroid)), # zero rotation for objs
                            np.zeros(len(obj_centroid)) # zero rotation for objs
                            )
                        camX0_T_camX = torch.bmm(camX0_T_origin.unsqueeze(0).repeat(len(obj_centroid),1,1), origin_T_camX)
                        _, trans = utils.geom.split_rt(camX0_T_camX)
                        obj_centroid = trans.numpy()
                else:
                    assert(False)

                if self.add_whole_image_mask:
                    obj_centroid = obj_centroid[:self.max_objs-1]
                    npad = ((1, self.max_objs-obj_centroid.shape[0]-1), (0, 0)) # reserve first index for full image features
                obj_centroid = np.pad(obj_centroid, pad_width=npad, mode='constant', constant_values=-1)
                obj_centroids.append(obj_centroid)

            if self.use_3d_pos_enc:
                positions.append(targets['agent_pose'])
        
        return images, boxes, interms, actions, positions, obj_centroids, obj_instance_ids, labels

np_str_obj_array_pattern = re.compile(r'[SaUO]')

default_collate_err_msg_format = (
    "default_collate: batch must contain tensors, numpy arrays, numbers, "
    "dicts or lists; found {}")

def my_collate(batch):
    r"""Puts each data field into a tensor with outer dimension batch size"""

    elem = batch[0]
    elem_type = type(elem)
    if isinstance(elem, torch.Tensor):
        # out_ = None
        # if torch.utils.data.get_worker_info() is not None:
        #     # If we're in a background process, concatenate directly into a
        #     # shared memory tensor to avoid an extra copy
        numel = sum([x.numel() for x in batch])
        storage = elem.storage()._new_shared(numel)
        out = elem.new(storage)

        # lens = [len(b) for b in batch]
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
            # raise RuntimeError('each element in list of batch should be of equal size')
            # return batch
            transposed = batch
        else:
            transposed = zip(*batch)
        return [my_collate(samples) for samples in transposed]

    raise TypeError(my_collate_err_msg_format.format(elem_type))


from torch.utils.data import Sampler, Dataset
import math
from typing import TypeVar, Optional, Iterator
import torch.distributed as dist
class DistributedWeightedSampler(Sampler):
    def __init__(self, dataset, num_replicas=None, rank=None, replacement=True):
        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = dist.get_rank()
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.num_samples = int(math.ceil(len(self.dataset) * 1.0 / self.num_replicas))
        self.total_size = self.num_samples * self.num_replicas
        self.replacement = replacement


    def calculate_weights(self, targets):
        class_sample_count = torch.tensor(
            [(targets == t).sum() for t in torch.unique(targets, sorted=True)])
        weight = 1. / class_sample_count.double()
        samples_weight = torch.tensor([weight[t] for t in targets])
        return samples_weight

    def __iter__(self):
        # deterministically shuffle based on epoch
        g = torch.Generator()
        g.manual_seed(self.epoch)
        if self.shuffle:
            indices = torch.randperm(len(self.dataset), generator=g).tolist()
        else:
            indices = list(range(len(self.dataset)))

        # add extra samples to make it evenly divisible
        indices += indices[:(self.total_size - len(indices))]
        assert len(indices) == self.total_size

        # subsample
        indices = indices[self.rank:self.total_size:self.num_replicas]
        assert len(indices) == self.num_samples

        # get targets (you can alternatively pass them in __init__, if this op is expensive)
        targets = self.dataset.targets
        targets = targets[self.rank:self.total_size:self.num_replicas]
        assert len(targets) == self.num_samples
        weights = self.calculate_weights(targets)

        return iter(torch.multinomial(weights, self.num_samples, self.replacement).tollist())

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch):
        self.epoch = epoch
