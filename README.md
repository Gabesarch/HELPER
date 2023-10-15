<h1 align="center">
    HELPER: Instructable and Personalizable Embodied Agents with
Memory-Augmented Context-Dependent LLM Prompting
</h1>

<p align="left">
<!--     <a href="//github.com/allenai/ai2thor-rearrangement/blob/main/LICENSE">
        <!-- ai2thor-rearrangement wasn't identifiable by GitHub (on the day this was added), so using the same one as ai2thor -->
<!--         <img alt="License" src="https://img.shields.io/github/license/allenai/ai2thor.svg?color=blue">
    </a> -->
    <a href="" target="_blank">
        <img alt="Website" src="https://img.shields.io/badge/website-HELPER-orange">
    </a>
<!--     <a href="//github.com/allenai/ai2thor-rearrangement/releases">
        <img alt="GitHub release" src="https://img.shields.io/github/release/allenai/ai2thor-rearrangement.svg">
    </a> -->
    <a href="" target="_blank">
        <img src="https://img.shields.io/badge/arXiv-2207.10761-<COLOR>">
    </a>
<!--     <a href="//arxiv.org/abs/2103.16544" target="_blank">
        <img src="https://img.shields.io/badge/venue-CVPR 2021-blue">
    </a> -->
    <a href="" target="_blank">
        <img src="https://img.shields.io/badge/video-YouTube-red">
    </a>
<!--     <a href="https://join.slack.com/t/ask-prior/shared_invite/zt-oq4z9u4i-QR3kgpeeTAymEDkNpZmCcg" target="_blank">
        <img src="https://img.shields.io/badge/questions-Ask PRIOR Slack-blue">
    </a> -->
</p>

This repo contains code and data for running HELPER. 

### Contents

<div class="toc">
<ul>
<li><a href="#installation"> Installation </a></li><ul>
<li><a href="#Environment"> Environment </a></li>
<li><a href="#TEACh-Dataset"> TEACh dataset </a></li>
<li><a href="#Model-Checkpoints-and-GPT-Embeddings"> Model Checkpoints and GPT Embeddings</a></li>
</ul>
<li><a href="#Running-TEACh-benchmark"> Running HELPER on TEACh </a></li><ul>
<li><a href="#Running-the-TfD-evaluation"> Run TEACh TfD </a></li>
<li><a href="#Ablations"> Ablations </a></li>
<li><a href="#Ground-truth"> Ground truth </a></li>
<li><a href="#User-Feedback"> User Feedback </a></li>
</ul>
<li><a href="#citation"> Citation </a></li>
</ul>
</div>

## Installation 

### Environment

**(1)** Start by cloning the repository:
```bash
git clone https://github.com/Gabesarch/HELPER.git
```
**(1a)** (optional) If you are using conda, create an environment: 
```bash
conda create -n helper python=3.8
```

**(2)** Install [PyTorch](https://pytorch.org/get-started/locally/) with the CUDA version you have. For example, run the following for CUDA 11.1: 
```bash
pip install torch==1.10.0+cu111 torchvision==0.11.0+cu111 torchaudio==0.10.0 -f https://download.pytorch.org/whl/torch_stable.html
```
<!-- pip install torch==1.8.1+cu111 torchvision==0.9.1+cu111 torchaudio==0.8.1 -f https://download.pytorch.org/whl/torch_stable.html -->

**(3)** Install additional requirements: 
```bash
pip install -r requirements.txt
```

**(4)** Install [Detectron2](https://detectron2.readthedocs.io/en/latest/tutorials/install.html) (needed for SOLQ detector) with correct PyTorch and CUDA version. 
E.g. for PyTorch 1.8 & CUDA 11.1:
```bash
python -m pip install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu111/torch1.10/index.html
```
<!-- python -m pip install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu111/torch1.8/index.html -->

**(5)** Install teach: 
```bash
pip install -e teach
```

**(6)** Build SOLQ deformable attention:
```bash
cd ./SOLQ/models/ops && sh make.sh && cd ../../..
```

**(7)** Clone ZoeDepth repo
```bash
git clone https://github.com/isl-org/ZoeDepth.git
cd ZoeDepth
git checkout edb6daf45458569e24f50250ef1ed08c015f17a7
```

### TEACh Dataset
1. Download the TEACh dataset following the instructions in the [TEACh repo](https://github.com/alexa/teach)
```bash
teach_download 
```

### Model Checkpoints and GPT Embeddings
To our model on the TEACh dataset, you'll first need the GPT embeddings for example retrieval:
1. Download GPT embeddings for example retrieval: [here](https://drive.google.com/file/d/1kqZZXdglNICjDlDKygd19JyyBzkkk-UL/view?usp=sharing). Unzip it to get the gpt_embedding folder in `./data` folder (or in a desired foldered and set --gpt_embedding_dir argument). 
Alternatively, you can download the file with gdown (`pip install gdown`): 
```bash
cd data
gdown 1kqZZXdglNICjDlDKygd19JyyBzkkk-UL
unzip gpt_embeddings.zip
rm gpt_embeddings.zip
```

TO run our model with estimated depth and segmentation, download the SOLQ and ZoeDepth checkpoints:

2. Download SOLQ checkpoint: [here](https://drive.google.com/file/d/1hTCtTuygPCJnhAkGeVPzWGHiY3PHNE2j/view?usp=sharing). Place it in the `./checkpoints` folder (or anywhere you want and specify the path with `--solq_checkpoint`). 
Alternatively, you can download the file with gdown (`pip install gdown`): 
```bash
cd checkpoints
gdown 1hTCtTuygPCJnhAkGeVPzWGHiY3PHNE2j
```

3. Download ZoeDepth checkpoint: [here](https://drive.google.com/file/d/1gMe8_5PzaNKWLT5OP-9KKEYhbNxRjk9F/view?usp=drive_link). Place it in the `./checkpoints` folder (or anywhere you want and specify the path with `--zoedepth_checkpoint`). (Also make sure you clone the ZoeDepth repo: `git clone https://github.com/isl-org/ZoeDepth.git`)
Alternatively, you can download the file with gdown (`pip install gdown`): 
```bash
cd checkpoints
gdown 1gMe8_5PzaNKWLT5OP-9KKEYhbNxRjk9F
```

<!-- 2. Generate the GPT embeddings for retrieval
```bash
cd prompt
python get_embedding_examples.
``` -->

## Running TEACh benchmark

### Running the TfD evaluation
1. (if required) Start x server.
if an X server is not already running on your machine. First, open a screen with the desired node, and run the following to open an x server on that node:
```bash
python startx.py 0
```
Specify the server port number with the argument `--server_port` (default 0).

2. Set OpenAI keys.
Set Azure keys:
```bash
export AZURE_OPENAI_KEY={KEY}
export AZURE_OPENAI_ENDPOINT={ENDPOINT}
```

(If not using Azure)
Important! If using openai API, append `--use_openai` to arguments. Then set openai key:
```bash
export OPENAI_API_KEY={KEY}
```

3. Run agent.
To run the agent with all modules and estimated perception on TfD validation unseen, run the following:
```bash
python main.py \
 --mode teach_eval_tfd \
 --split valid_unseen \
 --gpt_embedding_dir ./data/gpt_embeddings \
 --teach_data_dir PATH_TO_TEACH_DATASET \
 --server_port X_SERVER_PORT_HERE \
 --episode_in_try_except \
 --use_llm_search \
 --use_constraint_check \
 --run_error_correction_llm \
 --zoedepth_checkpoint ./checkpoints/ZOEDEPTH-model-00015000.pth \
 --solq_checkpoint ./checkpoints/SOLQ-model-00023000.pth \
 --set_name HELPER_teach_tfd_validunseen
 ```
Change split to `--split valid_seen` to evaluate validation seen set. 

#### Metrics
All metrics will be saved to `./output/metrics/{set_name}`. Metrics and videos will also automatically be logged to wandb.

#### Movie generation
To create movies of the agent, append `--create_movie` to the arguments. This will by default create a movie for every episode rendered to `./output/movies`. To change the episode frequency of logging, alter `--log_every` (e.g., `--log_every 10` to render videos every 10 episodes). To remove the map visualization, append `--remove_map_vis` to the arguments. This can speed up the episode since rendering the map visual can slow down episodes.

### Ablations
The following arguments can be removed to run the ablations:
1. Remove memory augmented prompting. Add argument `--ablate_example_retrieval`.
2. Remove LLM search (locator) (only random). Remove `--use_llm_search`.
3. Remove constraint check (inspector). Remove `--use_constraint_check`.
4. Remove error correction (rectifier). Remove `--run_error_correction_llm`.
5. Change openai model type. Change `--openai_model` argument (e.g., `--openai_model gpt-3.5-turbo`).

### Ground truth
The following arguments can be added to run with ground truth:
1. GT depth `--use_gt_depth`. Reccomended to also add `--increased_explore` with estimated segmentation for best performance.
2. GT segmentation `--use_gt_seg`.
3. GT action success `--use_gt_success_checker`.
4. GT error feedback `--use_GT_error_feedback`.
5. GT constraint check using controller metadata `--use_GT_constraint_checks`.
6. Increase max API fails `--max_api_fails {MAX_FAILS}`.

### User Feedback
To run with user feedback, add `--use_progress_check`. Two additional metric files (for feedback query 1 & 2) will be saved to `./output/metrics/{set_name}`.

### Running the TfD evaluation
See the `main` branch for how to run the TfD evaluation.

<!-- ### Remote Server Setup
To run the Ai2THOR simulator on a headless machine, you must either stat an X-server or use Ai2THOR's new headless mode. 
To start an X-server with any of the scripts, you can simply append `--startx` to the arguments. You can specify the X-server port use the `--server_port` argument.
Alternatively, you can use [Ai2THOR's new headless rendering](https://ai2thor.allenai.org/ithor/documentation/#headless-setup) by appending `--do_headless_rendering` to the arguments.  -->

# Citation
If you like this paper, please cite us:
```
@proceedings{findings-2023-findings-association-linguistics-emnlp,
    title = "Findings of the Association for Computational Linguistics: EMNLP 2023",
    editor = "Sarch, Gabriel  and
      Wu, Yue  and
      Tarr, Michael and
      Fragkiadaki, Katerina",
    month = dec,
    year = "2023",
    publisher = "Association for Computational Linguistics",
}
```
