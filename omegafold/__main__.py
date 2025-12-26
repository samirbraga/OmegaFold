# -*- coding: utf-8 -*-
# =============================================================================
# Copyright 2022 HeliXon Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =============================================================================
"""
The main function to run the prediction
"""
# =============================================================================
# Imports
# =============================================================================
import gc
import logging
import os
import sys
import time
import torch_xla
import torch
import torch_xla.runtime as xr
import torch_xla.core.xla_model as xm

import omegafold as of
from . import pipeline


# =============================================================================
# Functions
# =============================================================================


def _main(rank, args, state_dict, forward_config, input_files):
    device = torch_xla.device()
    n_chunks = xr.world_size()
    chunk_size = (len(input_files) + n_chunks - 1) // n_chunks
    input_files = input_files[rank * chunk_size:(rank + 1) * chunk_size]
    with torch.no_grad():
        xm.master_print(f"Constructing OmegaFold")
        model = of.OmegaFold(of.make_config(args.model))
        if state_dict is None:
            xm.master_print("Inferencing without loading weight")
        else:
            if "model" in state_dict:
                state_dict = state_dict.pop("model")
            model.load_state_dict(state_dict)
        model.eval()
        model.to(device)

        for input_file in input_files:
            xm.master_print(f"Reading {input_file}")
            for i, (input_data, save_path) in enumerate(
                pipeline.fasta2inputs(
                    input_file,
                    num_pseudo_msa=args.num_pseudo_msa,
                    output_dir=args.output_dir,
                    device=device,
                    mask_rate=args.pseudo_msa_mask_rate,
                    num_cycle=args.num_cycle,
                )
            ):
                xm.master_print(f"Predicting {i + 1}th chain in {input_file}")
                xm.master_print(
                    f"{len(input_data[0]['p_msa'][0])} residues in this chain."
                )
                ts = time.time()
                try:
                    output = model(
                        input_data,
                        predict_with_confidence=True,
                        fwd_cfg=forward_config
                    )                
                except RuntimeError as e:
                    xm.master_print(f"Failed to generate {save_path} due to {e}")
                    xm.master_print(f"Skipping...")
                    continue
                xm.master_print(f"Finished prediction in {time.time() - ts:.2f} seconds.")

                xm.master_print(f"Saving prediction to {save_path}")
                pipeline.save_pdb(
                    pos14=output["final_atom_positions"],
                    b_factors=output["confidence"] * 100,
                    sequence=input_data[0]["p_msa"][0],
                    mask=input_data[0]["p_msa_mask"][0],
                    save_path=save_path,
                    model=0
                )
                xm.master_print(f"Saved")
                del output
                gc.collect()
        xm.master_print("Done!")
