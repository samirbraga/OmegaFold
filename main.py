# -*- coding: utf-8 -*-
import os
import torch_xla
from omegafold import pipeline

from omegafold.__main__ import _main

if __name__ == '__main__':
    args = pipeline.get_args()

    # create the output directory
    os.makedirs(args.output_dir, exist_ok=True)

    inputs = pipeline.fasta2inputs(
        args.input_file,
        num_pseudo_msa=args.num_pseudo_msa,
        output_dir=args.output_dir,
        mask_rate=args.pseudo_msa_mask_rate,
        num_cycle=args.num_cycle,
    )
    inputs = list(inputs)
    print(f"Total inputs: {len(inputs)}")

    torch_xla.launch(_main, args=(args, inputs))