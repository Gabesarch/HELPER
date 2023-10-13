from arguments import args
import torch
import numpy as np
import random
import threading
import time
import os
import sys

import ipdb
st = ipdb.set_trace

# fix the seed for reproducibility
torch.manual_seed(args.seed)
np.random.seed(args.seed)
random.seed(args.seed)

def main():
    print("Mode:", args.mode)
    print(type(args.mode))    
    if 'teach_eval_edh' in args.mode or 'teach_eval_tfd' in args.mode:
        from models.teach_eval_embodied_llm import run_teach
        run_teach()
    elif 'teach_eval_custom' in args.mode:
        from models.teach_eval_custom import run_custom
        run_custom()
    elif 'teach_eval_continual' in args.mode:
        if args.mod_api_continual:
            from models.teach_eval_continual_modapi import run_continual
        else:
            from models.teach_eval_continual import run_continual
        run_continual()
    elif 'teach_train_depth' in args.mode:
        from models.teach_train_depth import Ai2Thor as Ai2Thor_DEPTH
        aithor_depth = Ai2Thor_DEPTH()
        aithor_depth.run_episodes()
    else:
        raise NotImplementedError

    print("main finished.")

if __name__ == '__main__':
    main()
