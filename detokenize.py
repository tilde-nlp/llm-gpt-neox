import sys
from typing import List
import numpy as np
import sentencepiece as spm
from megatron.data.indexed_dataset_hacked import MMapIndexedDataset
import argparse
import json

def parse_args():
  parser = argparse.ArgumentParser("Detokenizes some amount of tokens from .bin file provided.")
  
  parser.add_argument(
    "--input",
    type=str,
    required=True,
    help="Path to bin file with or without extension."
  )
  
  parser.add_argument(
    "--tokens",
    type=int,
    required=True,
    help="Number of tokens to aim for, for detokenization."
  )
  
  parser.add_argument(
    "--output",
    type=str,
    required=True,
    help="Output location. Will output a jsonl there."
  )
  
  parser.add_argument(
    "--tokenizer",
    type=str,
    required=True,
    help="Path to sentencepiece model."
  )
  
  args = parser.parse_args()
  
  return args

def load_indexed_dset(path_to_bin: str) -> MMapIndexedDataset:
    """Open <prefix>.idx / <prefix>.bin as a Megatron IndexedDataset."""
    if path_to_bin[-4:] == ".bin":
      return MMapIndexedDataset(path_to_bin[:-4])
    else:
      return MMapIndexedDataset(path_to_bin)
    

def main():
  args = parse_args()
  
  model = spm.SentencePieceProcessor(model_file=args.tokenizer)
  print(f"Loaded SentencePiece model ({model.get_piece_size()} pieces) from {args.tokenizer!r}")
  ds = load_indexed_dset(args.input)
  
  collected = 0
  with open(args.output, "w", encoding="utf-8") as f:
    chunk = -1
    chunk_step = 0.025 * args.tokens
    for doc_idx in range(len(ds)):
      sample = ds.get(doc_idx)
      text = model.decode(sample.tolist())
      obj = {"text": text}
      print(json.dumps(obj, ensure_ascii=False), file = f)
      collected+= len(sample)
      
      new_chunk = collected // chunk_step
      if new_chunk != chunk:
        chunk = new_chunk
        print(collected, "/", args.tokens, "detokenized.")
      
      if collected >= args.tokens:
        break

  print("Detokenized", collected, "tokens.")


if __name__ == "__main__":
  main()
