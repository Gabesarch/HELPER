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
    if 'alfred_eval' in args.mode:
        from models.alfred_eval_embodied_llm import run_alfred
        run_alfred()
    elif 'teach_train_depth' in args.mode:
        from models.teach_train_depth import Ai2Thor as Ai2Thor_DEPTH
        aithor_depth = Ai2Thor_DEPTH()
        aithor_depth.run_episodes()
    else:
        raise NotImplementedError

    print("main finished.")

if __name__ == '__main__':
    main()
