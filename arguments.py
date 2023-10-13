import argparse
import numpy as np
import os
import torch
parser = argparse.ArgumentParser()

parser.add_argument("--seed", type=int, default=39, help="Random seed")
parser.add_argument("--mode", type=str, help="mode to run, see main.py")
parser.add_argument("--verbose", action="store_true", default=False, help="print out actions + other logs during task")
parser.add_argument("--set_name", type=str, default="test00", help="experiment name")
parser.add_argument("--split", type=str, help="eval split")
parser.add_argument("--add_back_objs_progresscheck_v2", action="store_true", default=False, help="add back in objects to object tracker for progress check")
parser.add_argument("--use_openai", action="store_true", default=False, help="")
parser.add_argument("--gpt_model", type=str, default="gpt-4", help="options: gpt-3.5-turbo, text-davinci-003, gpt-4, gpt-4-32k")
parser.add_argument('--skip_if_exists', default=False, action='store_true', help='skip if file exists in teach metrics')
parser.add_argument("--dpi", type=int, default=100, help="DPI for plotting")
parser.add_argument("--max_traj_steps", type=int, default=1000, help="maximum trajectory steps")
parser.add_argument('--remove_map_vis', default=False, action='store_true', help='remove map visual from movies')
parser.add_argument('--always_check_constraints', default=False, action='store_true', help='')
parser.add_argument("--root", type=str, default="", help="root folder")
parser.add_argument("--tag", type=str, default="", help="root folder tag")
parser.add_argument("--teleport_to_objs", action="store_true", default=False, help="teleport to objects instead of navigating")
parser.add_argument("--render", action="store_true", default=False, help="render video and logs")
parser.add_argument("--use_gt_objecttrack", action="store_true", default=False, help="if navigating, use GT object masks for getting object detections + centroids?")
parser.add_argument("--use_gt_depth", action="store_true", default=False, help="if navigating, use GT depth maps? ")
parser.add_argument("--use_gt_seg", action="store_true", default=False, help="use GT segmentation?")
parser.add_argument("--use_gt_success_checker", action="store_true", default=False, help="use GT segmentation?")
parser.add_argument("--use_gt_centroids", action="store_true", default=False, help="use GT centroids?")
parser.add_argument("--do_masks", action="store_true", default=False, help="use masks?")
parser.add_argument("--use_solq", action="store_true", default=False, help="use SOLQ?")
parser.add_argument("--use_gt_subgoals", action="store_true", default=False, help="use GT subgoals?")
parser.add_argument("--sample_every_other", action="store_true", default=False, help="run every other episode in the split")
parser.add_argument("--episode_in_try_except", action="store_true", default=False, help="Continue to next episode if assertion error occurs? ")
parser.add_argument("--log_every", type=int, default=1, help="log every X episodes")
parser.add_argument("--on_aws", action="store_true", default=False, help="on AWS?")
parser.add_argument("--new_parser", action="store_true", default=False, help="on AWS?")
parser.add_argument('--load_explore', action='store_true', default=False, help="load explore for full task from path?")
parser.add_argument("--movie_dir", type=str, default="./output/movies", help="where to output rendered movies")
parser.add_argument("--precompute_map_path", default='./output/precomputed_maps', type=str, help="load trajectory list from file?")
parser.add_argument("--create_movie", action="store_true", default=False, help="create mp4 movie")
parser.add_argument("--visualize_masks", action="store_true", default=False, help="visualize masks in object tracker visuals")
parser.add_argument("--metrics_dir", type=str, default="./output/metrics", help="where to output rendered movies")
parser.add_argument("--llm_output_dir", type=str, default="./output/llm", help="where to output rendered movies")
parser.add_argument("--gpt_embedding_dir", type=str, default="./data", help="where to output rendered movies")
parser.add_argument("--run_error_correction_llm", action="store_true", default=False, help="run error correction for LLM")
parser.add_argument("--run_error_correction_basic", action="store_true", default=False, help="run error correction - manual correction")
parser.add_argument("--use_progress_check", action="store_true", default=False, help="run progress check at the end to replan")
parser.add_argument("--remove_unusable_slice", action="store_true", default=False, help="remove the unusable slice from the environment after slicing")
parser.add_argument("--add_back_objs_progresscheck", action="store_true", default=False, help="add back in objects to object tracker for progress check")
parser.add_argument("--teach_data_dir", type=str, default="./data", help="data directory where teach data is held")
parser.add_argument("--data_path", type=str, default="./data", help="data directory where teach data is held")
parser.add_argument("--dont_use_controller", action="store_true", default=False, help="dont init controller")
parser.add_argument("--use_constraint_check", action="store_true", default=False, help="use constraint check")
parser.add_argument("--num_continual_iter", type=int, default=3, help="how many continual learning iterations?")
parser.add_argument("--mod_api_continual", action="store_true", default=False, help="setting to modify the api instead of example retrieval for continual learning experiments")
parser.add_argument("--ablate_example_retrieval", action="store_true", default=False, help="ablate example retrieval for GPT?")
parser.add_argument("--increased_explore", action="store_true", default=False, help="increase explore for GT depth")
parser.add_argument("--max_episodes", type=int, default=None, help="maximum episodes to evaluate")

###########%%%%%%% agent parameters %%%%%%%###########
parser.add_argument("--start_startx", action="store_true", default=False, help="start x server upon calling main")
parser.add_argument("--server_port", type=int, default=1, help="server port for x server")
parser.add_argument("--do_headless_rendering", action="store_true", default=False, help="render in headless mode with new Ai2thor version")
parser.add_argument("--HORIZON_DT", type=int, default=30, help="pitch movement delta")
parser.add_argument("--DT", type=int, default=90, help="yaw movement delta")
parser.add_argument("--STEP_SIZE", type=int, default=0.25, help="yaw movement delta")
parser.add_argument("--pitch_range", type=list, default=[-30,60], help="pitch allowable range for the agent. positive is 'down'")
parser.add_argument("--fov", type=int, default=90, help="field of view")
parser.add_argument("--W", type=int, default=480, help="image width")
parser.add_argument("--H", type=int, default=480, help="image height")
parser.add_argument("--visibilityDistance", type=float, default=1.5, help="visibility NOTE: this will not change rearrangement visibility")
parser.add_argument('--debug', default=False, action='store_true')
parser.add_argument("--eval_split", type=str, default="test", help="evaluation mode: combined (rearrange), train, test, val")
parser.add_argument('--use_estimated_depth', action='store_true', help="use estimated depth?")
parser.add_argument('--num_search_locs_object', type=int, default=20, help='number of search locations for searching for object')
parser.add_argument("--dist_thresh", type=float, default=0.5, help="navigation distance threshold to point goal")
parser.add_argument('--use_GT_constraint_checks', default=False, action='store_true', help='use GT constraint checks?')
parser.add_argument('--use_GT_error_feedback', default=False, action='store_true', help='use GT error feedback?')
parser.add_argument("--max_api_fails", type=int, default=30, help="maximum allowable api failures")
parser.add_argument('--use_llm_search', default=False, action='store_true', help='use llm search')
parser.add_argument('--use_mask_rcnn_pred', default=False, action='store_true', help='use maskrcnn')
parser.add_argument("--episode_file", type=str, default=None, help="specify an episode file name to run")

###########%%%%%%% object tracker %%%%%%%###########
parser.add_argument("--OT_dist_thresh", type=float, default=0.5, help="distance threshold for NMS for object tracker")
parser.add_argument("--OT_dist_thresh_searching", type=float, default=0.5, help="distance threshold for NMS for object tracker")
parser.add_argument("--confidence_threshold", type=float, default=0.4, help="confidence threshold for detections [0, 0.1]")
parser.add_argument("--confidence_threshold_interm", type=float, default=0.4, help="intermediate object score threshold")
parser.add_argument("--confidence_threshold_searching", type=float, default=0.4, help="confidence threshold for detections when searching for a target object class [0, 0.1]")
parser.add_argument("--nms_threshold", type=float, default=0.5, help="NMS threshold for object tracker")
parser.add_argument("--use_GT_centroids", action="store_true", default=False, help="use GT centroids for object tracker")
parser.add_argument('--only_one_obj_per_cat', action='store_true', default=False, help="only one object per cateogry in the object tracker?")
parser.add_argument('--env_frame_width_FILM', type=int, default=300, help='Frame width (default:84)')
parser.add_argument('--env_frame_height_FILM', type=int, default=300, help='Frame height (default:84)')

# Depth network
parser.add_argument("--randomize_object_placements", action="store_true", default=False, help="")
parser.add_argument("--randomize_scene_lighting_and_material", action="store_true", default=False, help="")
parser.add_argument("--randomize_agent_pickup", action="store_true", default=False, help="")
parser.add_argument("--randomize_object_state", action="store_true", default=False, help="")
parser.add_argument("--load_model", action="store_true", default=False, help="")
parser.add_argument("--load_model_path", type=str, default="", help="load checkpoint path")
parser.add_argument("--lr_scheduler_from_scratch", action="store_true", default=False, help="")
parser.add_argument("--optimizer_from_scratch", action="store_true", default=False, help="")
parser.add_argument("--start_one", action="store_true", default=False, help="")
parser.add_argument('--max_iters', type=int, default=100000, help='max train iterations')
parser.add_argument('--log_freq', type=int, default=500, help='log frequency')
parser.add_argument('--val_freq', type=int, default=500, help='validation frequency')
parser.add_argument('--save_freq', type=int, default=2500, help='save checkpoint frequency')
parser.add_argument("--load_train_agent", action="store_true", default=False, help="")
parser.add_argument('--batch_size', default=2, type=int, help="batch size for model training")
parser.add_argument("--S", type=int, default=2, help="Number of views per trajectory")
parser.add_argument("--radius_min", type=float, default=0.0, help="radius min to spawn near target object")
parser.add_argument("--radius_max", type=float, default=7.0, help="radius max to spawn near target object")
parser.add_argument("--views_to_attempt", type=int, default=8, help="max views to attempt for getting trajectory")
parser.add_argument("--movement_mode", type=str, default="random", help="movement mode for action sampling for getting trajectory (forward_first, random); forward_first: always try to move forward")
parser.add_argument("--fail_if_no_objects", type=bool, default=True, help="fail view if no objects in view")
parser.add_argument("--torch_checkpoint_path", type=str, default="", help="torch hub checkpoint path")
parser.add_argument('--lr_scheduler_freq', type=int, default=20000, help='lr frequency for step')
parser.add_argument('--keep_latest', default=5, type=int, help="number of checkpoints to keep at one time")
parser.add_argument("--val_load_dir", type=str, default="./dataset/val", help="val load dir")
parser.add_argument("--run_val", action="store_true", default=False, help="")
parser.add_argument("--load_val_agent", action="store_true", default=False, help="")
parser.add_argument('--n_val', type=int, default=50, help='number of validation iters')
parser.add_argument("--zoedepth_checkpoint", type=str, default="./checkpoints/ZOEDEPTH-model-00015000.pth", help="zoe depth checkpoint to load for teach")
parser.add_argument("--solq_checkpoint", type=str, default="./checkpoints/SOLQ-model-00023000.pth", help="SOLQ checkpoint to load for teach")

# SOLQ hyperparams
parser.add_argument('--lr', default=2e-5, type=float) 
parser.add_argument('--lr_backbone_names', default=["backbone.0"], type=str, nargs='+')
parser.add_argument('--lr_backbone_mult', default=0.1, type=float)
parser.add_argument('--lr_linear_proj_names', default=['reference_points', 'sampling_offsets'], type=str, nargs='+')
parser.add_argument('--lr_linear_proj_mult', default=0.1, type=float)
parser.add_argument('--lr_text_encoder_mult', default=0.05, type=float)
parser.add_argument('--lr_text_encoder_names', default=["text_encoder"], type=str, nargs='+')

parser.add_argument('--weight_decay', default=1e-4, type=float)
parser.add_argument('--epochs', default=50, type=int)
parser.add_argument('--lr_drop', default=4000, type=int)
parser.add_argument('--save_period', default=10, type=int)
parser.add_argument('--lr_drop_epochs', default=None, type=int, nargs='+')
parser.add_argument('--clip_max_norm', default=1.0, type=float,
                    help='gradient clipping max norm')
parser.add_argument('--meta_arch', default='solq', type=str)
parser.add_argument('--sgd', action='store_true')
# Variants of Deformable DETR
parser.add_argument('--with_box_refine', default=True, action='store_true')
parser.add_argument('--two_stage', default=True)
# VecInst
parser.add_argument('--with_vector', default=True, action='store_true')
parser.add_argument('--n_keep', default=256, type=int,
                    help="Number of coeffs to be remained")
parser.add_argument('--gt_mask_len', default=128, type=int,
                    help="Size of target mask")
parser.add_argument('--vector_loss_coef', default=3, type=float)
parser.add_argument('--vector_hidden_dim', default=1024, type=int,
                    help="Size of the vector embeddings (dimension of the transformer)")
parser.add_argument('--no_vector_loss_norm', default=False, action='store_true')
parser.add_argument('--activation', default='relu', type=str, help="Activation function to use")
parser.add_argument('--checkpoint', default=False, action='store_true')
parser.add_argument('--vector_start_stage', default=0, type=int)
parser.add_argument('--num_machines', default=1, type=int)
parser.add_argument('--loss_type', default='l1', type=str)
parser.add_argument('--dcn', default=False, action='store_true')
# Model parameters
parser.add_argument('--frozen_weights', type=str, default=None,
                    help="Path to the pretrained model. If set, only the mask head will be trained")
parser.add_argument('--pretrained', default=None, help='resume from checkpoint')
# * Backbone
parser.add_argument('--backbone', default='resnet50', type=str,
                    help="Name of the convolutional backbone to use")
parser.add_argument('--dilation', action='store_true',
                    help="If true, we replace stride with dilation in the last convolutional block (DC5)")
parser.add_argument('--position_embedding', default='sine', type=str, choices=('sine', 'learned', 'rel'),
                    help="Type of positional embedding to use on top of the image features")
parser.add_argument('--position_embedding_scale', default=2 * np.pi, type=float,
                    help="position / size * scale")
parser.add_argument('--num_feature_levels', default=4, type=int, help='number of feature levels')
# * Transformer
parser.add_argument('--enc_layers', default=6, type=int,
                    help="Number of encoding layers in the transformer")
parser.add_argument('--dec_layers', default=6, type=int,
                    help="Number of decoding layers in the transformer")
parser.add_argument('--dim_feedforward', default=1024, type=int,
                    help="Intermediate size of the feedforward layers in the transformer blocks")
parser.add_argument('--hidden_dim', default=384, type=int,
                    help="Size of the embeddings (dimension of the transformer)")
parser.add_argument('--dropout', default=0.1, type=float,
                    help="Dropout applied in the transformer")
parser.add_argument('--nheads', default=8, type=int,
                    help="Number of attention heads inside the transformer's attentions")
parser.add_argument('--num_queries', default=300, type=int,
                    help="Number of query slots")
parser.add_argument('--dec_n_points', default=4, type=int)
parser.add_argument('--enc_n_points', default=4, type=int)
# * Segmentation
parser.add_argument('--masks', type=bool, default=True,
                    help="Train segmentation head if the flag is provided")
# Loss
parser.add_argument('--no_aux_loss', dest='aux_loss', action='store_false',
                    help="Disables auxiliary decoding losses (loss at each layer)")
# * Matcher
parser.add_argument('--set_cost_class', default=2, type=float,
                    help="Class coefficient in the matching cost")
parser.add_argument('--set_cost_bbox', default=5, type=float,
                    help="L1 box coefficient in the matching cost")
parser.add_argument('--set_cost_giou', default=2, type=float,
                    help="giou box coefficient in the matching cost")

parser.add_argument('--dataset_file', default='coco')
parser.add_argument('--coco_path', default='./data/coco', type=str)
parser.add_argument('--coco_panoptic_path', type=str)
parser.add_argument('--remove_difficult', action='store_true')
parser.add_argument('--alg', default='instformer', type=str)
parser.add_argument('--output_dir', default='',
                    help='path where to save, empty for no saving')
parser.add_argument('--device', default='cuda',
                    help='device to use for training / testing')
parser.add_argument('--resume', default='', help='resume from checkpoint')
parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                    help='start epoch')
parser.add_argument('--eval', action='store_true')
parser.add_argument('--test', action='store_true')
parser.add_argument('--num_workers', default=2, type=int)
parser.add_argument('--cache_mode', default=False, action='store_true', help='whether to cache images on memory')
# * Loss coefficients
parser.add_argument('--mask_loss_coef', default=1, type=float)
parser.add_argument('--dice_loss_coef', default=1, type=float)
parser.add_argument('--cls_loss_coef', default=2, type=float)
parser.add_argument('--bbox_loss_coef', default=5, type=float)
parser.add_argument('--giou_loss_coef', default=2, type=float)
parser.add_argument('--focal_alpha', default=0.25, type=float)

### WANDB
parser.add_argument("--group", type=str, default="default", help="group name")
parser.add_argument("--wandb_directory", type=str, default='./wandb', help="Path to wandb metadata")

args = parser.parse_args()

args.use_estimated_depth = True
if args.use_gt_depth:
   args.use_estimated_depth = False

args.metrics_dir = os.path.join(args.metrics_dir, args.set_name)
args.llm_output_dir = os.path.join(args.llm_output_dir, args.set_name)
args.movie_dir = os.path.join(args.movie_dir, args.set_name)

args.cuda = torch.device('cuda' if torch.cuda.is_available() else 'cpu')