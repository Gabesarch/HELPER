<h1 align="center">
    Open-Ended Instructable Embodied Agents with Memory-Augmented Large Language Models
</h1>

<p align="left">
    <a href="https://helper-agent-llm.github.io/" target="_blank">
        <img alt="Website" src="https://img.shields.io/badge/website-HELPER-orange">
    </a>
    <a href="https://arxiv.org/abs/2310.15127" target="_blank">
        <img src="https://img.shields.io/badge/arXiv-2207.10761-<COLOR>">
    </a>
    <a href="https://arxiv.org/abs/2404.19065" target="_blank">
        <img alt="HELEPR-X"src="https://img.shields.io/badge/arXiv-2207.10761-<COLOR>">
    </a>
</p>

This repo contains code and data for running HELPER and HELPER-X. This branch is for running HELPER on ALFRED. 

## Running on TEACh, Dialfred, or Tidy Task

This branch is for running HELPER and HELPER-X on ALFRED. 

- For instructions on how to run HELPER on TEACh, please see the 'main' branch.
- For instructions on how to run HELPER on Tidy Task, please see the 'Tidy Task' branch. (coming soon)
- For instructions on how to run HELPER on Dialfred, please see the 'Dialfred' branch. (coming soon)

### Contents

<div class="toc">
<ul>
<li><a href="#installation"> Installation </a></li>
<li><a href="#dataset"> Dataset </a></li>
<li><a href="#running-full-pipeline"> Running Pipeline </a></li>
<li><a href="#citation"> Citation </a></li>
</ul>
</div>

## Installation 

**(1)** Start by cloning the repository:
```bash
git clone https://github.com/Gabesarch/HELPER.git
```
**(1a)** (optional) If you are using conda, create an environment: 
```bash
conda create -n helper_alfred python=3.8
cd HELPER
```

**(2)** Install [PyTorch](https://pytorch.org/get-started/locally/) with the CUDA version you have. We have tested with PyTorch 1.10 and CUDA 11.1: 
```bash
pip install torch==1.10.0+cu111 torchvision==0.11.0+cu111 torchaudio==0.10.0 -f https://download.pytorch.org/whl/torch_stable.html
```

**(3)** Install additional requirements: 
```bash
pip install setuptools==59.8.0 numpy==1.23.1
pip install -r requirements.txt
```

**(4)** Install [Detectron2](https://detectron2.readthedocs.io/en/latest/tutorials/install.html) (needed for SOLQ detector) with correct PyTorch and CUDA version. 
E.g. for PyTorch 1.10 & CUDA 11.1:
```bash
python -m pip install 'git+https://github.com/facebookresearch/detectron2.git'
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

### ALFRED Dataset
Download the ALFRED dataset jsons:
```bash
cd alfred/data
sh download_data.sh json
```

### Model Checkpoints

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

## Running full pipeline with GT perception
if an X server is not already running on your machine. First, open a screen with the desired node, and run the following to open an x server on that node:
```bash
python startx.py 0
```

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

Set paths and run the following script:
```bash
python main.py \
    --mode alfred_eval \
    --split tests_unseen \
    --gpt_embedding_dir ./data/gpt_embeddings \
    --alfred_data_dir ./alfred/data \
    --create_movie \
    --remove_map_vis \
    --use_llm_search \
    --run_error_correction_llm \
    --use_constraint_check \
    --wandb_directory /projects/katefgroup/embodied_llm/ \
    --group alfred_task \
    --episode_in_try_except \
    --zoedepth_checkpoint ./checkpoints/ZOEDEPTH-model-00015000.pth \
    --solq_checkpoint ./checkpoints/SOLQ-model-00023000.pth \
    --server_port X_SERVER_PORT_HERE \
    --set_name alfred_run_validunseen_TESTUNSEEN
 ```

### Remote Server Setup
To run the Ai2THOR simulator on a headless machine, you must either stat an X-server or use Ai2THOR's new headless mode. 
To start an X-server with any of the scripts, you can simply append `--startx` to the arguments. You can specify the X-server port use the `--server_port` argument.
Alternatively, you can use [Ai2THOR's new headless rendering](https://ai2thor.allenai.org/ithor/documentation/#headless-setup) by appending `--do_headless_rendering` to the arguments. 

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