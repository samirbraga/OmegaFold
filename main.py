# -*- coding: utf-8 -*-
import os
import torch_xla
from omegafold import pipeline

from omegafold.__main__ import _main

if __name__ == '__main__':
    args, state_dict, forward_config = pipeline.get_args()
    # create the output directory
    os.makedirs(args.output_dir, exist_ok=True)
    torch_xla.launch(_main, args=(args, state_dict, forward_config, args.input_file))