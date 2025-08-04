import os
import sys
import yaml
from megatron.data.indexed_dataset_hacked import MMapIndexedDataset, MMapIndexedDatasetBuilder

import logging

# load idx/bin into megatron IndexedDataset class
def load_indexed_dset(path_to_bin):
    # needs dataset prefix, adds ".idx" extension automatically
    return MMapIndexedDataset(path_to_bin.replace(".bin", ""))

# init a builder for output
# FIXME: datatype?
def make_indexed_dset_builder(path_to_bin, dtype=69):
    # needs path to bin file
    return MMapIndexedDatasetBuilder(path_to_bin, dtype=dtype)

def normalise_bin(max_toukens: int, indexed_dset: MMapIndexedDataset,
              path_to_out_bin: str):
    """
    Slices the input index dataset on document level, while trying to match desired token count. Outputs a new .bin/.idx file.
    :param token_count: number of desired tokens
    :param indexed_dset: input IndexedDataset
    :param path_to_out_bin: path to output .bin file
    :return:
    """

    # log some stats
    normalised = 0
    total_tokens = 0

    # init output bin dataset builder
    is_builder_set = False

    # get total number of documents available in the dataset
    max_idx = indexed_dset.__len__()


    for current_idx in range(max_idx):

        # get a document
        temp_tokens = indexed_dset.get(current_idx)  # numpy array

        # calc length
        temp_len = temp_tokens.size

        did_normalise = False

        if not is_builder_set:
            # infer data type
            dtype = temp_tokens.dtype

            # init builder with correct datatype
            indexed_dset_builder = make_indexed_dset_builder(path_to_out_bin, dtype=dtype)

            is_builder_set = True


        # split if exceeding
        while temp_len > max_toukens:

            # slice off
            temp_sliced_tokens = temp_tokens[:max_toukens]

            # add to slice
            indexed_dset_builder.add_item(temp_sliced_tokens)  # this already writes to disk
            indexed_dset_builder.end_document()  # each sequence is a separate document

            total_tokens += temp_sliced_tokens.size

            # what's left
            temp_tokens = temp_tokens[max_toukens:]
            temp_len = temp_tokens.size

            did_normalise = True

        if did_normalise:
            normalised += 1
        # add to the left-overs to slice
        assert temp_len <= max_toukens
        total_tokens += temp_len
        indexed_dset_builder.add_item(temp_tokens)  # this already writes to disk
        indexed_dset_builder.end_document()  # each sequence is a separate document

    # finalize the slice, create idx file for the bin
    indexed_dset_builder.finalize(path_to_out_bin.replace(".bin", ".idx"))

    # some more logging
    logging.info(f"Normalised {normalised}/{max_idx} samples. [{round(100*normalised/max_idx,2)}%] ")
    logging.info(f"Total tokens: {total_tokens} tokens")
    logging.info("Normalised dataset written to %s(.bin/.idx)" % path_to_out_bin.replace(".bin", ""))



def main(args):

    # Configure logging to log both to file and console
    logging.basicConfig(
        level=logging.INFO,  # Set logging level
        format="%(asctime)s - %(levelname)s - %(message)s",  # Log format
        handlers=[
            logging.StreamHandler()  # Log to console
        ]
    )


    indexed_dataset = load_indexed_dset(args.input)  # load this once

    # construct output
    output_bin = args.out_dir + "/" + args.input.replace(".bin", "_normalised.bin")

    normalise_bin(args.max_toukens, indexed_dataset, output_bin)


    logging.info("Done.")


if __name__ == "__main__":
    import argparse

    # Argument parsing happens here, inside main()
    parser = argparse.ArgumentParser(description="A simple argument parser example.")

    parser.add_argument("--input", type=str, required=True,
                        help=f"Path to the input bin file.")
    parser.add_argument("--out_dir", type=str, required=True,
                        help=f"Path to output dir.")
    parser.add_argument("--max_tokens", type=int, required=True,
                        help=f"Maximum allowed sample length in toukens.")

    args = parser.parse_args()

    # Call main with parsed arguments


    main(args)