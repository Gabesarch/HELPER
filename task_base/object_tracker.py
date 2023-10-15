import numpy as np
import utils.aithor
import torch
from PIL import Image
import ipdb
st = ipdb.set_trace
from utils.ddetr_utils import check_for_detections
import numpy as np
import utils.aithor
import utils.geom
import torch
from PIL import Image
from arguments import args
import sys
import os
import cv2
import matplotlib.pyplot as plt
import ipdb
st = ipdb.set_trace
from utils.ddetr_utils import check_for_detections
from scipy.spatial import distance

import skimage

class ObjectTrack():

    def __init__(
        self, 
        name_to_id, 
        id_to_name, 
        include_classes, 
        W, H, 
        pix_T_camX=None, 
        ddetr=None, 
        use_gt_objecttrack=False, 
        controller=None, 
        navigation=None, 
        check_if_centroid_falls_within_map=False, 
        do_masks=False, 
        id_to_mapped_id=None, 
        origin_T_camX0=None,
        use_mask_rcnn_pred=False,
        use_open_set_segmenter=False,
        name_to_parsed_name=None,
        ): 
        '''
        check_if_centroid_falls_within_map: make sure centroid is within map bounds as given by navigation
        '''

        print("Initializing object tracker")

        if use_mask_rcnn_pred and use_open_set_segmenter:
            assert(False) # can't have both? 

        self.origin_T_camX0 = origin_T_camX0
        self.camX0_T_origin = utils.geom.safe_inverse_single(origin_T_camX0)

        self.controller = controller

        self.use_gt_objecttrack = use_gt_objecttrack
        self.use_mask_rcnn_pred = use_mask_rcnn_pred
        self.use_open_set_segmenter = use_open_set_segmenter

        self.include_classes = include_classes

        self.do_masks = do_masks
        # self.use_solq = use_solq
        self.nms_threshold = args.nms_threshold
        self.id_to_mapped_id = id_to_mapped_id

        self.ddetr = None
        
        if not use_gt_objecttrack and not use_mask_rcnn_pred:
            if self.use_open_set_segmenter:
                from task_base.open_set_detector import OpenSetSegmenter
                self.opensetsegmenter = OpenSetSegmenter(self.include_classes, name_to_parsed_name, name_to_id)
            elif ddetr is None:
                from nets.solq import DDETR                
                self.use_solq = True
                load_pretrained = False
                self.ddetr = DDETR(len(self.include_classes), load_pretrained).cuda()
                path = args.solq_checkpoint
                print("...found checkpoint %s"%(path))
                checkpoint = torch.load(path)
                pretrained_dict = checkpoint['model_state_dict']
                self.ddetr.load_state_dict(pretrained_dict, strict=True)
                self.ddetr.eval().cuda()
            else:
                self.use_solq = True
                self.ddetr = ddetr
                self.ddetr.eval().cuda()

        self.objects_track_dict = {}
        # define attributes and default values
        self.attributes = {"label":None, "locs":None, "holding":False, "scores":None, "can_use":True, "sliced":False, "toasted":False, "clean":False, "cooked":False}
    
        self.score_threshold = args.confidence_threshold 
        self.target_object_threshold = args.confidence_threshold_searching 
        self.dist_threshold_search = args.OT_dist_thresh_searching

        self.W = W
        self.H = H

        self.name_to_id = name_to_id
        self.id_to_name = id_to_name
        
        self.dist_threshold = args.OT_dist_thresh #1.0

        self.pix_T_camX = pix_T_camX

        self.only_one_obj_per_cat = args.only_one_obj_per_cat

        self.id_index = 0

        self.navigation = navigation
        self.check_if_centroid_falls_within_map = check_if_centroid_falls_within_map
        self.centroid_map_threshold = 1.5 # allowable distance to nearest navigable point
        self.centroid_map_threshold_in_bound = 0.75 # allowable distance to nearest non-navigable point

        self.score_boxes_name = 'pred1' # only one prediction head so same for both
        self.score_labels_name = 'pred1'

    def update(
        self, 
        rgb, 
        depth, 
        camX0_T_camX, 
        return_det_dict=False, 
        use_gt=False, 
        target_object=None, 
        only_keep_target=False,
        vis=None, 
        return_features=False,
        ):
        '''
        rgb: RGB image
        depth: depth image
        camX0_T_camX: rotation pose matrix to go from reference frame to current frame
        '''

        out = {}

        if not (depth.shape[-2]==self.W and depth.shape[-1]==self.H):
            depth = cv2.resize(depth, (self.W, self.H), interpolation=cv2.INTER_NEAREST)
        

        ####### DETECT OBJECTS ##########
        if self.use_gt_objecttrack:
            # print("Using GT Object Tracker")
            pred_scores, pred_labels, pred_boxes_or_masks, centroids_gt = self.get_objects_gt(self.controller, depth)
        else:
            if target_object is not None:
                if type(target_object)==str:
                    if target_object not in self.name_to_id.keys():
                        target_object_id = None
                    else:
                        target_object_id = self.name_to_id[target_object]
                elif type(target_object)==list:
                    target_object_id = [self.name_to_id[target_object_] for target_object_ in target_object]
                else:
                    assert(False)
            else:
                target_object_id = None
            if self.use_mask_rcnn_pred:
                
                pred_boxes_or_masks, pred_labels, pred_scores = self.get_maskrcnn_predictions()

            elif self.use_open_set_segmenter:

                pred_boxes_or_masks, pred_labels, pred_scores = self.opensetsegmenter.get_predictions(rgb)

            else:
                with torch.no_grad():
                    out = check_for_detections(
                        rgb, self.ddetr, self.W, self.H, 
                        self.score_labels_name, self.score_boxes_name, 
                        score_threshold_ddetr=self.score_threshold, do_nms=True, target_object=target_object_id, target_object_score_threshold=self.target_object_threshold,
                        solq=self.use_solq, return_masks=self.do_masks, nms_threshold=self.nms_threshold, id_to_mapped_id=self.id_to_mapped_id, return_features=return_features,
                        only_keep_target=only_keep_target,
                        )
                pred_labels = out["pred_labels"]
                pred_scores = out["pred_scores"]
                if self.do_masks:
                    pred_boxes_or_masks = out["pred_masks"] 
                else:
                    pred_boxes_or_masks = out["pred_boxes"]

        if return_det_dict:
            det_dict = out
            det_dict["centroid"] = []

        if self.check_if_centroid_falls_within_map:
            reachable = self.navigation.get_reachable_map_locations(sample=False)
            inds_i, inds_j = np.where(reachable)
            reachable_where = np.stack([inds_i, inds_j], axis=0)
        

        ########## FOR EACH DETECTION, GET OBJECT CENTROIDS + NMS ###########
        diffs_ = []
        if len(pred_scores)>0:
            if vis is not None:
                rgb_ = np.float32(rgb.copy())
            holding_label = self.get_label_of_holding()
            target_found = False
            for d in range(len(pred_scores)):
                if not self.use_gt_objecttrack:
                    label = self.id_to_name[pred_labels[d]]
                    if label in self.id_to_mapped_id.keys():
                        label = self.id_to_mapped_id[label]
                else:
                    label = pred_labels[d]
                    if label in self.id_to_mapped_id.keys():
                        label = self.id_to_mapped_id[label]
                score = pred_scores[d]
                box = pred_boxes_or_masks[d]
                if len(box)==0:
                    continue
                if self.use_gt_objecttrack and not args.use_gt_centroids:
                    centroid = utils.aithor.get_centroid_from_detection_no_controller(
                        box, depth, 
                        self.W, self.H, 
                        centroid_mode='median', 
                        pix_T_camX=self.pix_T_camX, 
                        origin_T_camX=camX0_T_camX
                        )
                elif self.use_gt_objecttrack and args.use_gt_centroids:
                    centroid = centroids_gt[d]
                else:
                    centroid = utils.aithor.get_centroid_from_detection_no_controller(
                        box, depth, 
                        self.W, self.H, 
                        centroid_mode='median', 
                        pix_T_camX=self.pix_T_camX, 
                        origin_T_camX=camX0_T_camX
                        )
                if centroid is None:
                    continue

                if self.check_if_centroid_falls_within_map:
                    obj_center_camX0_ = {'x':centroid[0], 'y':centroid[1], 'z':centroid[2]}
                    map_pos_centroid = self.navigation.get_map_pos_from_aithor_pos(obj_center_camX0_)
                    dist_to_reachable = distance.cdist(np.expand_dims(map_pos_centroid, axis=0), reachable_where.T)
                    argmin = np.argmin(dist_to_reachable)
                    map_pos_reachable_closest = [inds_i[argmin], inds_j[argmin]]
                    # map_pos_reachable_closest = self.navigation.get_clostest_reachable_map_pos(map_pos_centroid)
                    if not map_pos_centroid[0] or not map_pos_reachable_closest[1]:
                        pass
                    else:
                        dist = np.linalg.norm(map_pos_centroid - map_pos_reachable_closest) * self.navigation.explorer.resolution 
                        if dist>self.centroid_map_threshold:
                            continue
                
                if return_det_dict:
                    det_dict['centroid'].append(centroid)

                if not self.use_gt_objecttrack:
                    if label not in self.name_to_id:
                        continue
                else:
                    if label=="Floor":
                        continue

                if vis is not None and args.visualize_masks:
                    rect_th = 1
                    if label==target_object:
                        target_found = True
                        color = (0, 0, 255)
                    else:
                        color = (0, 255, 0)
                    if len(box)==4:
                        cv2.rectangle(rgb_, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])),color, rect_th)
                    else:
                        box = np.squeeze(box)
                        masked_img = np.where(box[...,None], color, rgb_)
                        rgb_ = cv2.addWeighted(rgb_, 0.8, np.float32(masked_img), 0.2,0)
                        if "pred_boxes" in out.keys():
                            box_vis = out["pred_boxes"][d]
                            cv2.rectangle(rgb_, (int(box_vis[0]), int(box_vis[1])), (int(box_vis[2]), int(box_vis[3])),color, rect_th)

                # check if detected object already exists. if it does, add only one with highest score
                locs = []
                IDs_same = []
                for id_ in self.objects_track_dict.keys():
                    if self.objects_track_dict[id_]["label"]==label and self.objects_track_dict[id_]["locs"] is not None:
                        locs.append(self.objects_track_dict[id_]["locs"])
                        IDs_same.append(id_)


                ########## NMS BASED ON CATEGORY + DISTANCE ##############
                dist_thresh_ = self.dist_threshold_search if target_object is not None else self.dist_threshold
                if self.only_one_obj_per_cat:
                    if len(locs)>0:
                        locs = np.array(locs)
                        dists = np.sqrt(np.sum((locs - np.expand_dims(centroid, axis=0))**2, axis=1))
                        dists_thresh = dists<dist_thresh_ #self.dist_threshold
                        if np.sum(dists_thresh)>0:
                            same_ind = np.where(dists_thresh)[0][0]
                            same_id = IDs_same[same_ind]
                            loc_cur = self.objects_track_dict[same_id]['locs']
                            score_cur = self.objects_track_dict[same_id]['scores']
                            holding_cur = self.objects_track_dict[same_id]['holding']
                            # add one with highest score if they are the same object
                            if not holding_cur:
                                if score>=score_cur:
                                    self.objects_track_dict[same_id]['scores'] = score
                                    self.objects_track_dict[same_id]['locs'] = centroid
                                else:
                                    self.objects_track_dict[same_id]['scores'] = score_cur
                                    self.objects_track_dict[same_id]['locs'] = loc_cur
                    else:
                        attributes = self.attributes.copy()
                        attributes["locs"] = centroid
                        attributes["label"] = label
                        attributes["scores"] = score
                        self.create_new_object_entry(attributes)
                else:
                    if len(locs)>0:
                        locs = np.array(locs)
                        dists = np.sqrt(np.sum((locs - np.expand_dims(centroid, axis=0))**2, axis=1))
                        dists_thresh = dists<dist_thresh_ #self.dist_threshold
                    else:
                        dists_thresh = 0 # no objects of this class in memory
                    if np.sum(dists_thresh)>0:
                        same_ind = np.where(dists_thresh)[0][0]
                        same_id = IDs_same[same_ind]
                        loc_cur = self.objects_track_dict[same_id]['locs']
                        score_cur = self.objects_track_dict[same_id]['scores']
                        holding_cur = self.objects_track_dict[same_id]['holding']
                        # add one with highest score if they are the same object
                        if not holding_cur:
                            if score>=score_cur:
                                self.objects_track_dict[same_id]['scores'] = score
                                self.objects_track_dict[same_id]['locs'] = centroid
                                # self.objects_track_dict[label]['holding'].append(False)
                            else:
                                self.objects_track_dict[same_id]['scores'] = score_cur
                                self.objects_track_dict[same_id]['locs'] = loc_cur
                                # self.objects_track_dict[label]['holding'].append(False)
                    else:
                        attributes = self.attributes.copy()
                        attributes["locs"] = centroid
                        attributes["label"] = label
                        attributes["scores"] = score
                        self.create_new_object_entry(attributes)

            if vis is not None:
                if target_found:
                    for _ in range(5):
                        vis.add_frame(rgb_, text="Target found!")
                else:
                    vis.add_frame(rgb_, text="Update object tracker")
        
        if return_det_dict:
            return det_dict

    def create_new_object_entry(self, attributes):
        self.objects_track_dict[self.id_index] = {}
        for attr in list(self.attributes.keys()):
            if attr=="locs":
                self.objects_track_dict[self.id_index][attr] = attributes['locs']
            elif attr=="label":
                self.objects_track_dict[self.id_index][attr] = attributes['label']
            elif attr=="holding":
                self.objects_track_dict[self.id_index][attr] = attributes['holding']
            elif attr=="scores":
                self.objects_track_dict[self.id_index][attr] = attributes['scores']
            elif attr=="can_use":
                self.objects_track_dict[self.id_index][attr] = attributes['can_use']
            elif attr=="sliced":
                self.objects_track_dict[self.id_index][attr] = attributes['sliced']
            elif attr=="toasted":
                # by default set toasted to False
                # TODO: detect this
                self.objects_track_dict[self.id_index][attr] = attributes['toasted']
            elif attr=="clean":
                # by default set clean to False
                # TODO: detect this
                self.objects_track_dict[self.id_index][attr] = attributes['clean']
            elif attr=="cooked":
                # by default set cooked to False
                # TODO: detect this
                self.objects_track_dict[self.id_index][attr] = attributes['cooked']
            else:
                print(attr)
                assert(False) # didnt add this attribute
        self.id_index += 1

    def filter_centroids_out_of_bounds(self):
        reachable = self.navigation.get_reachable_map_locations(sample=False)
        inds_i, inds_j = np.where(reachable)
        reachable_where = np.stack([inds_i, inds_j], axis=0)

        centroids, labels, IDs = self.get_centroids_and_labels(return_ids=True)

        for idx in range(len(IDs)):
            centroid = centroids[idx]
            obj_center_camX0_ = {'x':centroid[0], 'y':centroid[1], 'z':centroid[2]}
            map_pos_centroid = self.navigation.get_map_pos_from_aithor_pos(obj_center_camX0_)
            dist_to_reachable = distance.cdist(np.expand_dims(map_pos_centroid, axis=0), reachable_where.T)
            argmin = np.argmin(dist_to_reachable)
            map_pos_reachable_closest = [inds_i[argmin], inds_j[argmin]]
            if not map_pos_centroid[0] or not map_pos_reachable_closest[1]:
                pass
            else:
                dist = np.linalg.norm(map_pos_centroid - map_pos_reachable_closest) * self.navigation.explorer.resolution 
                if dist>self.centroid_map_threshold:
                    # print("centroid outside map bounds. continuing..")
                    del self.objects_track_dict[IDs[idx]]

    @torch.no_grad()
    def get_maskrcnn_predictions(self):
        self.navigation.depth_estimator.seg.get_instance_mask_seg_alfworld_both()
        segmented_dict = self.navigation.depth_estimator.seg.segmented_dict
        small_len = len(segmented_dict['small']['scores'])
        large_len = len(segmented_dict['large']['scores'])

        pred_labels = []
        pred_scores = []
        pred_boxes_or_masks = []
        for i in range(small_len):
            label = self.navigation.depth_estimator.seg.small_idx2small_object[int(segmented_dict['small']['classes'][i])]
            score = float(segmented_dict['small']['scores'][i])
            mask = segmented_dict['small']['masks'][i]
            if not (mask.shape[-2]==self.W and mask.shape[-1]==self.H):
                mask = cv2.resize(mask, (self.W, self.H), interpolation=cv2.INTER_NEAREST)
            label_mapped = self.name_to_id[label] 
            if label_mapped in self.id_to_mapped_id.keys():
                label_mapped = self.id_to_mapped_id[label_mapped]
            pred_labels.append(label_mapped)
            pred_scores.append(score)
            pred_boxes_or_masks.append(mask)

        for i in range(large_len):
            label = self.navigation.depth_estimator.seg.large_idx2large_object[int(segmented_dict['large']['classes'][i])]
            score = float(segmented_dict['large']['scores'][i])
            mask = segmented_dict['large']['masks'][i]
            if not (mask.shape[-2]==self.W and mask.shape[-1]==self.H):
                mask = cv2.resize(mask, (self.W, self.H), interpolation=cv2.INTER_NEAREST)
            label_mapped = self.name_to_id[label] 
            if label_mapped in self.id_to_mapped_id.keys():
                label_mapped = self.id_to_mapped_id[label_mapped]
            pred_labels.append(label_mapped)
            pred_scores.append(score)
            pred_boxes_or_masks.append(mask)

        if len(pred_labels)>0:
            pred_labels = np.asarray(pred_labels)
            pred_scores = np.asarray(pred_scores)
            pred_boxes_or_masks = np.stack(pred_boxes_or_masks, axis=0)
        
        return pred_boxes_or_masks, pred_labels, pred_scores


    def get_ID_of_holding(self):
        for key in list(self.objects_track_dict.keys()):
            cur_dict = self.objects_track_dict[key]
            if cur_dict['holding']:
                return key
        return None


    def get_label_of_holding(self):
        key = self.get_ID_of_holding()
        if key is not None:
            return self.objects_track_dict[key]['label']
        return None

    def is_holding(self):
        key = self.get_ID_of_holding()
        if key is not None:
            return True
        return False

    def update_centroid(self, ID, centroid):
        self.objects_track_dict[ID]["locs"] = centroid

    def set_ID_to_sliced(self, ID):
        nonsliced_to_sliced = {'Apple':'AppleSliced', 'Bread':'BreadSliced', 'Lettuce':'LettuceSliced', 'Potato':'PotatoSliced', 'Tomato':'TomatoSliced'}
        self.objects_track_dict[ID]["label"] = nonsliced_to_sliced[self.objects_track_dict[ID]["label"]]

    def set_ID_to_holding(self, ID, value=True):
        self.objects_track_dict[ID]["holding"] = value

    def set_ID_to_can_use(self, ID, value=True):
        self.objects_track_dict[ID]["can_use"] = value   

    def get_label_from_ID(self, ID):
        return self.objects_track_dict[ID]["label"]

    def get_IDs_within_distance(self, centroid, distance_threshold):
        '''
        Get all object IDs within a distance threshold to centroid(s)
        '''
        centroid = np.asarray(centroid)
        locs, labels, ids_return = self.get_centroids_and_labels(return_ids=True)
        if len(centroid.shape)<2:
            centroid = np.expand_dims(centroid, axis=0)
        ids = set()
        for idx in range(len(centroid)):
            dists = np.sqrt(np.sum((locs - centroid[idx:idx+1])**2, axis=1))
            dists_thresh = np.logical_and(dists<distance_threshold, dists!=0)
            where_threshold = np.where(dists_thresh)[0]
            ids_threshold = [ids_return[w_i] for w_i in list(where_threshold)]
            ids.update(ids_threshold)

        return list(ids)

    def get_centroids_and_labels(self, return_ids=False, object_cat=None, include_holding=False):
        '''
        get centroids and labels in memory
        object_cat: object category string or list of category strings
        '''
        # order by score 
        scores = []
        IDs = []
        for key in list(self.objects_track_dict.keys()):
            cur_dict = self.objects_track_dict[key]
            if include_holding or not cur_dict['holding']:
                scores.append(cur_dict['scores'])
                IDs.append(key)

        scores_argsort = np.argsort(-np.array(scores))
        IDs = np.asarray(IDs)[scores_argsort]

        # iterate through with highest score first
        centroids = []
        labels = []
        IDs_ = []
        for key in list(IDs):
            cur_dict = self.objects_track_dict[key]
            if object_cat is not None:
                # check for category if input
                if type(object_cat)==str:
                    cat_check = cur_dict['label']==object_cat
                elif type(object_cat)==list:
                    cat_check = cur_dict['label'] in object_cat
                else:
                    assert(False) # wrong object_cat input
            else:
                cat_check = True
            if (include_holding or not cur_dict['holding']) and cat_check and cur_dict['can_use']:
                centroids.append(cur_dict['locs'])
                labels.append(cur_dict['label'])
                IDs_.append(key)

        if return_ids:
            return np.array(centroids), labels, IDs_

        return np.array(centroids), labels

    def get_score_of_label(self, object_cat):
        # order by score 
        scores = []
        IDs = []
        for key in list(self.objects_track_dict.keys()):
            cur_dict = self.objects_track_dict[key]
            if object_cat is not None:
                # check for category if input
                if type(object_cat)==str:
                    cat_check = cur_dict['label']==object_cat
                elif type(object_cat)==list:
                    cat_check = cur_dict['label'] in object_cat
                else:
                    assert(False) # wrong object_cat input
            else:
                cat_check = True
            if cat_check:
                scores.append(cur_dict['scores'])
        return scores

    def get_2D_point(
        self, 
        camX_T_origin=None, 
        obj_center_camX0_=None, 
        object_category=None, 
        rgb=None, 
        score_threshold=0.0,
        object_id=None,
        sampling='center'
        ):
        '''
        modes: reproject_centroid, 
        '''

        if not self.use_gt_objecttrack: # first use detector? 
            if self.use_mask_rcnn_pred:
                pred_boxes_or_masks_, pred_labels_, pred_scores_ = self.get_maskrcnn_predictions()
                pred_boxes_or_masks, pred_labels, pred_scores = [], [], []
                for i in range(len(pred_labels_)):
                    if object_category is None or self.id_to_name[pred_labels_[i]]==object_category:
                        pred_boxes_or_masks.append(pred_boxes_or_masks_[i])
                        pred_labels.append(pred_labels_[i])
                        pred_scores.append(pred_scores_[i])
                if len(pred_boxes_or_masks)>0:
                    pred_boxes_or_masks = np.stack(pred_boxes_or_masks, axis=0)
                    pred_labels = np.asarray(pred_labels)
                    pred_scores = np.asarray(pred_scores)
                    if object_id is not None:
                        pred_scores_sorted = self.sort_masks_by_reprojected_from_ID(pred_boxes_or_masks, object_id, return_idxs=True)
                    else:
                        pred_scores_sorted = np.argsort(-pred_scores)
            else:
                with torch.no_grad():
                    # first see if detector has it
                    out = check_for_detections(
                            rgb, self.ddetr, self.W, self.H, 
                            self.score_labels_name, self.score_boxes_name, 
                            score_threshold_ddetr=score_threshold, do_nms=False, return_features=False,
                            solq=self.use_solq, return_masks=self.do_masks, nms_threshold=self.nms_threshold, id_to_mapped_id=self.id_to_mapped_id,
                            )
                pred_labels = out["pred_labels"]
                pred_scores = out["pred_scores"]
                if self.do_masks:
                    pred_boxes_or_masks = out["pred_masks"] 
                    if object_id is not None:
                        pred_scores_sorted = self.sort_masks_by_reprojected_from_ID(pred_boxes_or_masks, object_id, return_idxs=True)
                    else:
                        pred_scores_sorted = np.argsort(-pred_scores)
                else:
                    pred_boxes_or_masks = out["pred_boxes"] 
                    pred_scores_sorted = np.argsort(-pred_scores)
            
            center2D = []
            for d in range(len(pred_scores)):
                idx = pred_scores_sorted[d]
                label = self.id_to_name[pred_labels[idx]]
                if label==object_category:
                    if self.do_masks:
                        mask = pred_boxes_or_masks[idx]
                        where_target_i, where_target_j = np.where(mask)
                        if sampling=="center":
                            count = (mask == 1).sum()
                            x_center, y_center = np.argwhere(mask==1).sum(0)/count
                            argsort_center_i = np.argsort(np.abs(where_target_i-x_center))
                            where_center_j = where_target_j[np.where(where_target_i==where_target_i[argsort_center_i[0]])[0]]
                            argsort_center_j = np.argsort(np.abs(where_center_j-y_center))
                            where_target_i = where_target_i[argsort_center_i[0]]
                            where_target_j = where_center_j[argsort_center_j[0]]
                        elif sampling=="random":
                            sampled_idx = np.random.randint(len(where_target_i))
                            where_target_j = where_target_j[sampled_idx]
                            where_target_i = where_target_i[sampled_idx]
                        else:
                            assert NotImplementedError
                        center2D_ = [where_target_i, where_target_j]
                        center2D.append(center2D_)
                    else:
                        box = pred_boxes_or_masks[idx]
                        x_min, y_min, x_max, y_max = list(np.round(box).astype(np.int))
                        center2D_ = [(y_min+y_max)/2, (x_min+x_max)/2]
                        center2D.append(center2D_)
            if len(center2D)==0:
                pass # move to reprojected
                # return None, "DETECTED"
            else:
                print("RETURNING DETECTED 2D POINT")
                return center2D, "DETECTED"
        else:
            scores, labels, masks, centroids = self.get_objects_gt(self.controller, np.ones((self.W, self.H)))
            object_masks = []
            for label, mask in zip(labels, masks):
                if label==object_category:
                    object_masks.append(mask)
            if len(object_masks)==0:
                # return None, "DETECTED"
                pass # move to reprojected
            else:
                if object_id is not None:
                    object_masks = self.sort_masks_by_reprojected_from_ID(np.stack(object_masks), object_id)
                print("RETURNING DETECTED 2D POINT")
                center2D = []
                for object_mask in object_masks:
                    where_target_i, where_target_j = np.where(object_mask)
                    if sampling=="center":
                        count = (object_mask == 1).sum()
                        x_center, y_center = np.argwhere(object_mask==1).sum(0)/count
                        argsort_center_i = np.argsort(np.abs(where_target_i-x_center))
                        where_center_j = where_target_j[np.where(where_target_i==where_target_i[argsort_center_i[0]])[0]]
                        argsort_center_j = np.argsort(np.abs(where_center_j-y_center))
                        where_target_j = where_center_j[argsort_center_j[0]]
                        where_target_i = where_target_i[argsort_center_i[0]]
                    elif sampling=="random":
                        sampled_idx = np.random.randint(len(where_target_i))
                        where_target_j = where_target_j[sampled_idx]
                        where_target_i = where_target_i[sampled_idx]
                    else:
                        assert NotImplementedError
                    center2D_ = [where_target_i, where_target_j]
                    center2D.append(center2D_)
                return center2D, "DETECTED"

        print("RETURNING REPROJECTED 2D POINT")
        centroids, labels, IDs = self.get_centroids_and_labels(return_ids=True, object_cat=object_category)
        center2D = [self.get_reprojected_point_given_ID(ID) for ID in IDs]
        if all(v is None for v in center2D):
            return None, "REPROJECTED"

        # # Camera2Pixels
        # obj_center_camX0_ = torch.from_numpy(np.array(list(obj_center_camX0_.values())))
        # obj_center_camX0_ = torch.reshape(obj_center_camX0_, [1, 1, 3])
        # object_center_camX = utils.geom.apply_4x4(camX_T_origin.float(), obj_center_camX0_.float())
        # pix_T_cam = torch.from_numpy(self.pix_T_camX).unsqueeze(0).float()
        # center2D = np.squeeze(utils.geom.apply_pix_T_cam(pix_T_cam, object_center_camX).numpy())
        # center2D = center2D[[1,0]] # need to swap these
        # center2D = list(center2D)
        return center2D, "REPROJECTED"

    def set_new_detector_confidence(
        self,
        score_threshold,
    ):
        if self.use_mask_rcnn_pred:
            self.cached_sem_seg_threshold_small = args.sem_seg_threshold_small
            self.cached_sem_seg_threshold_large = args.sem_seg_threshold_large
            args.sem_seg_threshold_small = score_threshold
            args.sem_seg_threshold_large = score_threshold

    def reset_detector_confidence(
        self,
    ):
        if self.use_mask_rcnn_pred:
            args.sem_seg_threshold_small = self.cached_sem_seg_threshold_small
            args.sem_seg_threshold_large = self.cached_sem_seg_threshold_large

    def get_predicted_masks(
        self, 
        rgb, 
        object_category=None, 
        score_threshold=0.0,
        max_masks=3,
        ):
        '''
        modes: reproject_centroid, 
        '''
        if self.use_mask_rcnn_pred:
            pred_boxes_or_masks_, pred_labels_, pred_scores_ = self.get_maskrcnn_predictions()
            pred_boxes_or_masks, pred_labels, pred_scores = [], [], []
            for i in range(len(pred_labels_)):
                if object_category is None or self.id_to_name[pred_labels_[i]]==object_category:
                    pred_boxes_or_masks.append(pred_boxes_or_masks_[i])
                    pred_labels.append(pred_labels_[i])
                    pred_scores.append(pred_scores_[i])
            if len(pred_boxes_or_masks)>0:
                pred_boxes_or_masks = np.stack(pred_boxes_or_masks, axis=0)
                pred_labels = np.asarray(pred_labels)
                pred_scores = np.asarray(pred_scores)
        elif self.use_open_set_segmenter:
            pred_boxes_or_masks_, pred_labels_, pred_scores_ = self.opensetsegmenter.get_predictions(rgb)
            pred_boxes_or_masks, pred_labels, pred_scores = [], [], []
            for i in range(len(pred_labels_)):
                if object_category is None or self.id_to_name[pred_labels_[i]]==object_category:
                    pred_boxes_or_masks.append(pred_boxes_or_masks_[i])
                    pred_labels.append(pred_labels_[i])
                    pred_scores.append(pred_scores_[i])
            if len(pred_boxes_or_masks)>0:
                pred_boxes_or_masks = np.stack(pred_boxes_or_masks, axis=0)
                pred_labels = np.asarray(pred_labels)
                pred_scores = np.asarray(pred_scores)
        else:
            with torch.no_grad():
                # first see if detector has it
                out = check_for_detections(
                        rgb, self.ddetr, self.W, self.H, 
                        self.score_labels_name, 
                        self.score_boxes_name, 
                        score_threshold_ddetr=score_threshold, 
                        do_nms=True, 
                        return_features=False,
                        solq=self.use_solq, 
                        return_masks=self.do_masks, 
                        nms_threshold=self.nms_threshold, 
                        id_to_mapped_id=self.id_to_mapped_id,
                        target_object=self.name_to_id[object_category] if object_category is not None else None,
                        target_object_score_threshold=score_threshold,
                        )
            pred_labels = out["pred_labels"]
            pred_scores = out["pred_scores"]
            pred_boxes_or_masks = out["pred_masks"] 
        if len(pred_boxes_or_masks)==0:
            return []
        pred_scores_sorted = np.argsort(-pred_scores)
        masks_sorted = []
        masks_scores = []
        for d in range(len(pred_scores)):
            idx = pred_scores_sorted[d]
            label = self.id_to_name[pred_labels[idx]]
            if label==object_category:
                mask = pred_boxes_or_masks[idx]
                masks_sorted.append(mask)
                masks_scores.append(pred_scores[idx])
        # print(masks_scores)
        if len(masks_sorted)>0:
            masks_sorted = np.stack(masks_sorted)
        elif len(masks_sorted)>max_masks:
            masks_sorted = masks_sorted[:max_masks]
        return masks_sorted

    def get_reprojected_point_given_ID(
        self, 
        ID,
        # camX_T_origin, 
        # obj_center_camX0_, 
        ):
        '''
        reprojects centroid of ID to 2D based on current position from navigation
        '''

        camX0_T_camX = self.navigation.explorer.get_camX0_T_camX()
        camX_T_camX0 = utils.geom.safe_inverse_single(camX0_T_camX)
        obj_center_camX0_ = self.objects_track_dict[ID]['locs']
        # obj_center_camX0_ = torch.from_numpy(np.array(list(obj_center_camX0_.values())))
        if obj_center_camX0_ is None:
            return None
        obj_center_camX0_ = torch.from_numpy(obj_center_camX0_)
        obj_center_camX0_ = torch.reshape(obj_center_camX0_, [1, 1, 3])
        object_center_camX = utils.geom.apply_4x4(camX_T_camX0.float(), obj_center_camX0_.float())
        pix_T_cam = torch.from_numpy(self.pix_T_camX).unsqueeze(0).float()
        center2D = np.squeeze(utils.geom.apply_pix_T_cam(pix_T_cam, object_center_camX).numpy())
        center2D = list(center2D)

        return center2D

    def get_reprojected_mask(
        self, 
        ID,
        ):
        '''
        reprojects centroid of ID to 2D based on current position from navigation
        Then gets mask from it for manipulation
        '''

        center2D = self.get_reprojected_point_given_ID(ID)

        if center2D[0]>self.W-1 or center2D[0]<0 or center2D[1]>self.H-1 or center2D[1]<0:
            return []

        mask = np.zeros((self.W, self.H), dtype=bool)
        mask[int(center2D[0]), int(center2D[1])] = True

        selem = skimage.morphology.disk(int(5 * self.W/480))
        mask_out = skimage.morphology.binary_dilation(mask, selem) == True
        mask_out = np.expand_dims(mask_out, axis=0)

        return mask_out

    def sort_masks_by_reprojected_from_ID(self, masks, ID, return_idxs=False):
        '''
        sorts masks by which are closer to reprojected point of ID
        masks: NxWxH
        ID: object id to reproject
        '''
        center2D = self.get_reprojected_point_given_ID(ID)
        where_masks = np.where(masks)
        min_dist_masks = []
        for mask in masks:
            where_mask = np.array(np.where(mask))
            try:
                dists_2_center2D = np.linalg.norm(where_mask - np.array(center2D)[[1,0],None], axis=0)
                min_dist_masks.append(np.min(dists_2_center2D))
            except:
                min_dist_masks.append(10000000)
        idx_masks = np.argsort(min_dist_masks)
        if return_idxs:
            return idx_masks
        masks = masks[idx_masks,:,:]
        # for mask_i in range(len(masks)):
        #     plt.figure()
        #     plt.imshow(masks[mask_i])
        #     plt.plot(center2D[0], center2D[1], 'o')
        #     plt.savefig(f'output/images/test_mask{mask_i}.png')
        # st()
        return masks

    def get_objects_gt(self, controller, depth):
        '''
        Gets object info from simulator directly
        '''

        origin_T_camX = utils.aithor.get_origin_T_camX(controller.last_event, False).cuda()

        semantic = controller.last_event.instance_segmentation_frame
        object_id_to_color = controller.last_event.object_id_to_color
        color_to_object_id = controller.last_event.color_to_object_id

        obj_ids = np.unique(semantic.reshape(-1, semantic.shape[2]), axis=0)
        
        obj_metadata_IDs = []
        for obj_m in controller.last_event.metadata['objects']: #objects:
            obj_metadata_IDs.append(obj_m['objectId'])

        instance_masks = controller.last_event.instance_masks
        instance_detections2d = controller.last_event.instance_detections2D
        obj_meta_all = controller.last_event.metadata['objects']

        bboxes = []
        labels = []
        scores = []
        centroids = []

        idxs = []

        for object_id in instance_masks.keys(): #range(obj_ids.shape[0]): 

            if object_id not in obj_metadata_IDs:
                continue

            idxs.append(object_id)  

        for object_id in idxs: 

            obj_meta_index = obj_metadata_IDs.index(object_id)
            obj_meta = obj_meta_all[obj_meta_index]

            obj_category_name = obj_meta['objectType']

            if obj_category_name not in self.name_to_id:
                continue

            if self.name_to_id[obj_category_name] in self.id_to_mapped_id.keys():
                obj_category_name = self.id_to_name[self.id_to_mapped_id[self.name_to_id[obj_category_name]]]

            obj_name = obj_meta['name']

            i_mask = instance_masks[object_id]

            if np.all(depth[i_mask]<0.5):
                # filter out anything in agent's hand
                continue

            # bring to camX0 reference frame
            centroid = torch.from_numpy(np.array(list(obj_meta['position'].values()))).unsqueeze(0).cuda()
            centroid = utils.geom.apply_4x4(self.camX0_T_origin.unsqueeze(0).cuda().float(), centroid.unsqueeze(1).cuda().float()).squeeze(1)
            centroid[:,1] = -centroid[:,1]      
            centroid = centroid.squeeze().cpu().numpy()       

            bboxes.append(i_mask) #obj_bbox)
            labels.append(obj_category_name)
            scores.append(1.)
            centroids.append(centroid)

        return scores, labels, bboxes, centroids

    def get_objects_gt_from_meta(self):

        bboxes = []
        labels = []
        scores = []
        centroids = []

        objects = self.controller.last_event.metadata['objects']

        for obj_meta in objects: # skip target object


            obj_category_name = obj_meta['objectType']

            # transform to camX0 reference frame
            centroid = torch.from_numpy(np.array(list(obj_meta['axisAlignedBoundingBox']['center'].values()))).unsqueeze(0).cuda()
            centroid = utils.geom.apply_4x4(self.camX0_T_origin.unsqueeze(0).cuda().float(), centroid.unsqueeze(1).cuda().float()).squeeze(1)
            centroid[:,1] = -centroid[:,1]    
            centroid = centroid.squeeze().cpu().numpy()             

            # bboxes.append(i_mask) #obj_bbox)

            labels.append(obj_category_name)
            scores.append(1.)
            centroids.append(centroid)

        return np.array(centroids), labels