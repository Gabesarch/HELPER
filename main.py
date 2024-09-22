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
    if 'tidy_eval' in args.mode or 'tidy_examples' in args.mode:
        from models.tidy_eval_embodied_llm import run_tidy
        run_tidy()
    else:
        raise NotImplementedError

    print("main finished.")

if __name__ == '__main__':
    main()
