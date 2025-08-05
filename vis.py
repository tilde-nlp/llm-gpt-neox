#!/usr/bin/env python
"""
Inspect a Megatron IndexedDataset.

For first 100 documents it prints
  •  token IDs / pieces / detokenised text`
"""

import sys
from typing import List
from tqdm import tqdm
import numpy as np
import sentencepiece as spm

from megatron.data.indexed_dataset_hacked import MMapIndexedDataset


# --- helpers ----
def load_indexed_dset(path_to_bin: str) -> MMapIndexedDataset:
    """Open <prefix>.idx / <prefix>.bin as a Megatron IndexedDataset."""
    return MMapIndexedDataset(path_to_bin.replace(".bin", ""))


def show(ids: List[int], label: str, doc_idx: int, sp: any):
    print(f"[doc {doc_idx}] {label} IDs   : {ids}")
    print(f"[doc {doc_idx}] {label} pieces: {[sp.id_to_piece(i) for i in ids]}")
    print(f"[doc {doc_idx}] {label} text  : {sp.decode(ids)}")


def main(input_bin, sp_model):
    special_tokens = []

    # --- SentencePiece --------
    sp = spm.SentencePieceProcessor(model_file=sp_model)
    print(f"Loaded SentencePiece model ({sp.get_piece_size()} pieces) from {sp_model!r}")

    dset = load_indexed_dset(input_bin)

    for doc_idx in tqdm(range(100), desc="docs"):
        ids: np.ndarray = dset.get(doc_idx)

        pretty_ids = ids.tolist()

        show(pretty_ids, "full document", doc_idx, sp)


# --- main ----------------------------------------------------------
if __name__ == "__main__":
    if len(sys.argv) < 3:
        print(f"Usage: {sys.argv[0]} /path/to/file.bin /path/to/tokenizer_dir")
        sys.exit(1)

    inp_bin = sys.argv[1]
    tokenizer_dir = sys.argv[2]

    main(inp_bin, tokenizer_dir)
