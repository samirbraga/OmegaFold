# -*- coding: utf-8 -*-
import os
import torch_xla
from omegafold import pipeline

from omegafold.__main__ import _main

if __name__ == '__main__':
    args, state_dict, forward_config = pipeline.get_args()
    # create the output directory
    os.makedirs(args.output_dir, exist_ok=True)
    input_files = [
        "../proteinmpnn_residues/rec_prot_40_sample_0_proteinmpnn_residues_0.fasta",
        "../proteinmpnn_residues/rec_prot_43_sample_31_proteinmpnn_residues_7.fasta",
        "../proteinmpnn_residues/rec_prot_49_sample_82_proteinmpnn_residues_1.fasta",
        "../proteinmpnn_residues/rec_prot_98_sample_122_proteinmpnn_residues_2.fasta"
    ]
    torch_xla.launch(_main, args=(args, state_dict, forward_config, input_files))