<h1 align="center">
    Open-Ended Instructable Embodied Agents with Memory-Augmented Large Language Models
</h1>

<p align="left">
    <a href="https://helper-agent-llm.github.io/" target="_blank">
        <img alt="Website" src="https://img.shields.io/badge/website-HELPER-orange">
    </a>
    <a href="https://helper-agent-llm.github.io/" target="_blank">
        <img alt="HELPER" src="https://img.shields.io/badge/paper-HELPER-blue">
    </a>
    <a href="https://helper-agent-llm.github.io/" target="_blank">
        <img alt="HELPER" src="https://img.shields.io/badge/paper-HELPERX-green">
    </a>
<!--     <a href="https://arxiv.org/abs/2310.15127" target="_blank">
        <img src="https://img.shields.io/badge/arXiv-2207.10761-<COLOR>">
    </a> -->
<!--     <a href="https://arxiv.org/abs/2404.19065" target="_blank">
        <img alt="HELEPR-X"src="https://img.shields.io/badge/arXiv-2207.10761-<COLOR>">
    </a> -->
</p>

🚀 **Exciting News!** We have newly added support for **ALFRED** and the **Tidy Task**! This major update allows users to run HELPER and HELPER-X on these additional benchmarks. See the 'ALFRED' and 'Tidy Task' branches for more information. Dialfred coming soon!

This repo contains code and data for running HELPER and HELPER-X. This branch is for running HELPER on TEACh. 

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
<li><a href="#Running-the-EDH-evaluation"> Run TEACh EDH </a></li>
</ul>
<li><a href="#citation"> Citation </a></li>
</ul>
</div>

## Running on ALFRED, Dialfred, or Tidy Task
This branch is for running HELPER and HELPER-X on TEACh. 

Please see the 'ALFRED' branch for instructions on how to run HELPER on ALFRED.

Please see the 'Dialfred' branch for instructions on how to run HELPER on Dialfred. (coming soon)

Please see the 'Tidy Task' branch for instructions on how to run HELPER on Tidy Task. (coming soon)

## Installation 

### Environment

**(1)** Start by cloning the repository:
```bash
git clone https://github.com/Gabesarch/HELPER.git
cd HELPER
```
**(1a)** (optional) If you are using conda, create an environment: 
```bash
conda create -n helper python=3.8
```

**(2)** Install [PyTorch](https://pytorch.org/get-started/locally/) with the CUDA version you have. We have tested with PyTorch 1.10 and CUDA 11.1: 
```bash
pip install torch==1.10.0+cu111 torchvision==0.11.0+cu111 torchaudio==0.10.0 -f https://download.pytorch.org/whl/torch_stable.html
```

**(3)** Install additional requirements: 
```bash
pip install setuptools==59.8.0 numpy==1.23.1 # needed for scikit-image
pip install -r requirements.txt
```

**(4)** Install [Detectron2](https://detectron2.readthedocs.io/en/latest/tutorials/install.html) (needed for SOLQ detector) with correct PyTorch and CUDA version. 
```bash
python -m pip install 'git+https://github.com/facebookresearch/detectron2.git'
```

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
<!-- To our model on the TEACh dataset, you'll first need the GPT embeddings for example retrieval:
1. Download GPT embeddings for example retrieval: [here](https://drive.google.com/file/d/1kqZZXdglNICjDlDKygd19JyyBzkkk-UL/view?usp=sharing). Unzip it to get the gpt_embedding folder in `./data` folder (or in a desired foldered and set --gpt_embedding_dir argument). 
Alternatively, you can download the file with gdown (`pip install gdown`): 
```bash
cd data
gdown 1kqZZXdglNICjDlDKygd19JyyBzkkk-UL
unzip gpt_embeddings.zip
rm gpt_embeddings.zip
``` -->

TO run our model with estimated depth and segmentation, download the SOLQ and ZoeDepth checkpoints:

1. Download SOLQ checkpoint: [here](https://drive.google.com/file/d/1hTCtTuygPCJnhAkGeVPzWGHiY3PHNE2j/view?usp=sharing). Place it in the `./checkpoints` folder (or anywhere you want and specify the path with `--solq_checkpoint`). 
Alternatively, you can download the file with gdown (`pip install gdown`): 
```bash
cd checkpoints
gdown 1hTCtTuygPCJnhAkGeVPzWGHiY3PHNE2j
```

2. Download ZoeDepth checkpoint: [here](https://drive.google.com/file/d/1gMe8_5PzaNKWLT5OP-9KKEYhbNxRjk9F/view?usp=drive_link). Place it in the `./checkpoints` folder (or anywhere you want and specify the path with `--zoedepth_checkpoint`). (Also make sure you clone the ZoeDepth repo: `git clone https://github.com/isl-org/ZoeDepth.git`)
Alternatively, you can download the file with gdown (`pip install gdown`): 
```bash
cd checkpoints
gdown 1gMe8_5PzaNKWLT5OP-9KKEYhbNxRjk9F
```

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

### Running the EDH evaluation
See the `teach_edh` branch for how to run the TEACh EDH evaluation.

<!-- ### Remote Server Setup
To run the Ai2THOR simulator on a headless machine, you must either stat an X-server or use Ai2THOR's new headless mode. 
To start an X-server with any of the scripts, you can simply append `--startx` to the arguments. You can specify the X-server port use the `--server_port` argument.
Alternatively, you can use [Ai2THOR's new headless rendering](https://ai2thor.allenai.org/ithor/documentation/#headless-setup) by appending `--do_headless_rendering` to the arguments.  -->

### References

This project utilizes several repositories and tools. Below are the references to these repositories:

- [TEACh](https://github.com/alexa/teach): TEACh (Task-driven Embodied Agents that Chat) is a dataset and benchmark for training and evaluating embodied agents in interactive environments.
- [ALFRED](https://github.com/askforalfred/alfred): ALFRED (Action Learning From Realistic Environments and Directives) is a benchmark for learning from natural language instructions in simulated environments.
- [TIDEE](https://github.com/gabesarch/TIDEE): TIDEE (Task-Informed Dialogue Embodied Environment) is a framework for training embodied agents in task-oriented dialogue settings.
- [SOLQ](https://github.com/megvii-research/SOLQ): SOLQ (Segmenting Objects by Learning Queries) is a method for object detection and segmentation.
- [ZoeDepth](https://github.com/isl-org/ZoeDepth): ZoeDepth is a repository for depth estimation models.

Please refer to these repositories for more detailed information and instructions on their usage.

# Citation
If you like this paper, please cite us:
```
@inproceedings{sarch2023helper,
                        title = "Open-Ended Instructable Embodied Agents with Memory-Augmented Large Language Models",
                        author = "Sarch, Gabriel and
                        Wu, Yue and
                        Tarr, Michael and
                        Fragkiadaki, Katerina",
                        booktitle = "Findings of the Association for Computational Linguistics: EMNLP 2023",
                        year = "2023"}
```

```
@inproceedings{sarch2024helperx,
                        title = "HELPER-X: A Unified Instructable Embodied Agent to Tackle Four Interactive Vision-Language Domains with Memory-Augmented
                        Language Models",
                        author = "Sarch, Gabriel and Somani, Sahil and Kapoor, Raghav and Tarr, Michael J and Fragkiadaki, Katerina",
                        booktitle = "ICLR 2024 LLMAgents Workshop",
                        year = "2024"}
```
