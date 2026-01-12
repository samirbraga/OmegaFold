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
import torch.nn.functional as F
import torch_xla.runtime as xr
import torch_xla.core.xla_model as xm

import omegafold as of
from omegafold import utils

from . import pipeline


# =============================================================================
# Functions
# =============================================================================

def _get_max_fasta_len(fasta_path: str) -> int:
    max_len = 0
    curr_len = 0
    with open(fasta_path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if curr_len > max_len:
                    max_len = curr_len
                curr_len = 0
            else:
                curr_len += len(line)
    if curr_len > max_len:
        max_len = curr_len
    return max_len


def _pad_inputs(
        inputs, pad_len: int, pad_value: int = 20
):
    if pad_len is None:
        return inputs
    padded = []
    for cycle_data in inputs:
        p_msa = cycle_data["p_msa"]
        p_msa_mask = cycle_data["p_msa_mask"]
        seq_len = p_msa.shape[-1]
        if seq_len > pad_len:
            raise ValueError(
                f"Sequence length {seq_len} exceeds padding length {pad_len}."
            )
        if seq_len < pad_len:
            pad_amount = pad_len - seq_len
            p_msa = F.pad(p_msa, (0, pad_amount), value=pad_value)
            p_msa_mask = F.pad(p_msa_mask, (0, pad_amount), value=0)
        padded.append({"p_msa": p_msa, "p_msa_mask": p_msa_mask})
    return padded


def _mp_fn(rank, args, inputs):
    state_dict, forward_config = pipeline.get_models(args)
    device = torch_xla.device()
    n_chunks = xr.world_size()
    pad_len = 128

    chunk_size = (len(inputs) + n_chunks - 1) // n_chunks
    inputs = inputs[rank * chunk_size:(rank + 1) * chunk_size]
    
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

        xm.master_print(f"Reading {args.input_file}")
        for i, (input_data, save_path) in enumerate(inputs):
            xm.master_print(f"Predicting {i + 1}th chain in {args.input_file}")
            input_data = utils.recursive_to(input_data, device=device)

            input_data = _pad_inputs(input_data, pad_len)
            xm.master_print(
                f"{int(input_data[0]['p_msa_mask'][0].sum().item())} residues in this chain."
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

def main():
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

    torch_xla.launch(_mp_fn, args=(args, inputs))

if __name__ == '__main__':
    main()