<h1 align="center">
    Code for embodied LLM project
</h1>

### Contents

<div class="toc">
<ul>
<li><a href="#installation"> Installation </a></li>
<li><a href="#dataset"> Dataset </a></li>
<li><a href="#running-full-pipeline-with-GT-perception"> Running Pipeline </a></li>
<li><a href="#citation"> Citation </a></li>
</ul>
</div>

## Running on ALFRED, Dialfred, or Tidy Task
This branch is for running HELPER and HELPER-X on Tidy Task. 

Please see the 'main' branch for instructions on how to run HELPER on TEACh.

Please see the 'ALFRED' branch for instructions on how to run HELPER on ALFRED.

Please see the 'Dialfred' branch for instructions on how to run HELPER on Dialfred. (coming soon)

## Installation 

**(1)** Start by cloning the repository:
```bash
git clone https://github.com/Gabesarch/embodied-llm.git
cd HELPER
```
**(1a)** (optional) If you are using conda, create an environment: 
```bash
conda create -n helper_tidy python=3.8
```

**(2)** Install [PyTorch](https://pytorch.org/get-started/locally/) with the CUDA version you have. We have tested with PyTorch 1.10 and CUDA 11.1:  
```bash
pip install torch==1.10.0+cu111 torchvision==0.11.0+cu111 torchaudio==0.10.0 -f https://download.pytorch.org/whl/torch_stable.html
```

**(3)** Install additional requirements: 
```bash
pip install -r requirements.txt
```

**(4)** Install [Detectron2](https://detectron2.readthedocs.io/en/latest/tutorials/install.html) (needed for SOLQ detector) with correct PyTorch and CUDA version. 
```bash
python -m pip install 'git+https://github.com/facebookresearch/detectron2.git'
```

**(6)** Build SOLQ deformable attention:
```bash
cd ./SOLQ/models/ops && sh make.sh && cd ../../..
```

### Download Tidy Task data
Download messup meta data:
```bash
cd data
gdown 1KFUxxL8KU4H8dxBpjhp1SGAf3qnTtEBM
tar -xzf messup.tar.gz
rm messup.tar.gz
```

## Checkpoints and GPT Embeddings

TO run our model with estimated depth and segmentation, download the SOLQ and ZoeDepth checkpoints:

1. Download SOLQ checkpoint: [here](https://drive.google.com/file/d/1hTCtTuygPCJnhAkGeVPzWGHiY3PHNE2j/view?usp=sharing). Place it in the `./checkpoints` folder (or anywhere you want and specify the path with `--solq_checkpoint`). 
Alternatively, you can download the file with gdown (`pip install gdown`): 
```bash
cd checkpoints
gdown 1sEYLWQr-Ya2MtM4I_w_KNcMwofTju87T
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
    --mode tidy_eval \
    --split test \
    --mess_up_dir ./data/messup \
    --solq_checkpoint ./checkpoints/solq_oop-00010500.pth \
    --gpt_embedding_dir ./data/gpt_embeddings \
    --create_movie \
    --use_llm_search \
    --run_error_correction_llm \
    --use_constraint_check \
    --wandb_directory /projects/katefgroup/embodied_llm/ \
    --group tidy_task_estimated \
    --add_back_objs_progresscheck \
    --do_predict_oop \
    --server_port SERVER_PORT_HERE \
    --increased_explore \
    --episode_in_try_except \
    --set_name HELPER_tidy
 ```

### Remote Server Setup
To run the Ai2THOR simulator on a headless machine, you must either stat an X-server or use Ai2THOR's new headless mode. 
To start an X-server with any of the scripts, you can simply append `--startx` to the arguments. You can specify the X-server port use the `--server_port` argument.
Alternatively, you can use [Ai2THOR's new headless rendering](https://ai2thor.allenai.org/ithor/documentation/#headless-setup) by appending `--do_headless_rendering` to the arguments. 

<!-- # Citation
If you like this paper, please cite us:
```
``` -->
