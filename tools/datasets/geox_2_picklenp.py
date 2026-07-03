import argparse
import os
import sys

sys.path.append(
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), os.path.pardir, os.path.pardir)
    )
)

import tqdm
from megatron.data.indexed_dataset_hacked import MMapIndexedDataset, MMapIndexedDatasetBuilder
import pickle as pkl

def parse_args():
  parser = argparse.ArgumentParser("Turn a GPT-NeoX dataset to a pickle of numpy arrays.")

  parser.add_argument(
    "--input-prefix",
    type=str,
    required=True,
    help="Path to GeoX dataset without .bin/.idx extension"
  )

  parser.add_argument(
    "--output",
    type=str,
    required=True,
    help="Path where pickle file will be saved."
  )

  return parser.parse_args()


def main(args):
  input_ds = MMapIndexedDataset(args.input_prefix)

  output_file = open(args.output, "wb")

  sample_count = len(input_ds)

  for idx in tqdm.tqdm(range(sample_count)):
    sample = input_ds.get(idx)
    pkl.dump(sample, output_file)

  output_file.close()

if __name__ == "__main__":
  main(parse_args())
