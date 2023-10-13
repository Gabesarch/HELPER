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

## Installation 

**(1)** Start by cloning the repository:
```bash
git clone https://github.com/Gabesarch/HELPER.git
```
**(1a)** (optional) If you are using conda, create an environment: 
```bash
conda create -n helper python=3.8
```

You also will want to set CUDA paths. For example (on our tested machine with CUDA 11.1): 
```bash
export CUDA_HOME="/opt/cuda/11.1.1"
export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:$LD_LIBRARY_PATH"
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

## TEACh Dataset
1. Download the TEACh dataset following the instructions in the [TEACh repo](https://github.com/alexa/teach)
```bash
teach_download 
```

## Checkpoints and GPT Embeddings
To our model on the TEACh dataset, you'll first need the GPT embeddings for example retrieval:
1. Download GPT embeddings for example retrieval: [here](https://drive.google.com/file/d/1kqZZXdglNICjDlDKygd19JyyBzkkk-UL/view?usp=sharing). Place them in ./dataset folder (or in a desired foldered and set --gpt_embedding_dir argument).

TO run our model with estimated depth and segmentation, download the SOLQ and ZoeDepth checkpoints:

2. Download SOLQ checkpoint: [here](https://drive.google.com/file/d/1hTCtTuygPCJnhAkGeVPzWGHiY3PHNE2j/view?usp=sharing). Place it in the `./checkpoints` folder (or anywhere you want and specify the path with `--solq_checkpoint`). 

3. Download ZoeDepth checkpoint: [here](https://drive.google.com/file/d/1gMe8_5PzaNKWLT5OP-9KKEYhbNxRjk9F/view?usp=drive_link). Place it in the `./checkpoints` folder (or anywhere you want and specify the path with `--zoedepth_checkpoint`). (Also make sure you clone the ZoeDepth repo: `git clone https://github.com/isl-org/ZoeDepth.git`)

<!-- 2. Generate the GPT embeddings for retrieval
```bash
cd prompt
python get_embedding_examples.
``` -->

## Running TEACh benchmark

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
 --gpt_embedding_dir ./dataset/gpt_embeddings \
 --teach_data_dir PATH_TO_TEACH_DATASET \
 --server_port X_SERVER_PORT_HERE \
 --episode_in_try_except \
 --use_llm_search \
 --use_constraint_check \
 --run_error_correction_llm \
 --zoedepth_checkpoint ./checkpoints/model-00020000.pth \
 --solq_checkpoint ./checkpoints/model-00020000.pth \
 --set_name HELPER_teach_tfd_validunseen
 ```
Change split to `--split valid_seen` to evaluate validation seen set. 

### Movie generation:
To create movies of the agent, append `--create_movie` to the arguments. This will by default create a movie for every episode rendered to `./output/movies`. To change the episode frequency of logging, alter `--log_every` (e.g., `--log_every 10` to render videos every 10 episodes). To remove the map visualization, append `--remove_map_vis` to the arguments. This can speed up the episode since rendering the map visual can slow down episodes.

### Ground truth
The following arguments can be added to run with ground truth:
1. GT depth `--use_gt_depth`. Reccomended to also add `--increased_explore` with estimated segmentation for best performance.
2. GT segmentation `--use_gt_seg`.
3. GT action success `--use_gt_success_checker`.
4. GT error feedback `--use_GT_error_feedback`.
5. GT constraint check using controller metadata `--use_GT_constraint_checks`.
6. Increase max API fails `--max_api_fails {MAX_FAILS}`.

### Ablations
The following arguments can be removed to run the ablations:
1. Remove memory augmented prompting. Add argument `--ablate_example_retrieval`.
2. Remove LLM search (locator) (only random). Remove `--use_llm_search`.
3. Remove constraint check (inspector). Remove `--use_constraint_check`.
4. Remove error correction (rectifier). Remove `--run_error_correction_llm`.
5. Change openai model type. Change `--openai_model` argument (e.g., `--openai_model gpt-3.5-turbo`).

<!-- ### Remote Server Setup
To run the Ai2THOR simulator on a headless machine, you must either stat an X-server or use Ai2THOR's new headless mode. 
To start an X-server with any of the scripts, you can simply append `--startx` to the arguments. You can specify the X-server port use the `--server_port` argument.
Alternatively, you can use [Ai2THOR's new headless rendering](https://ai2thor.allenai.org/ithor/documentation/#headless-setup) by appending `--do_headless_rendering` to the arguments.  -->

<!-- # Citation
If you like this paper, please cite us:
```
``` -->
