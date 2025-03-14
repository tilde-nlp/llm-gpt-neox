import os
import sys
import yaml
from megatron.data.indexed_dataset_hacked import MMapIndexedDataset, MMapIndexedDatasetBuilder

import logging
# set logging level
log_filename = "slicer.log"

# Configure logging to log both to file and console
logging.basicConfig(
    level=logging.INFO,  # Set logging level
    format="%(asctime)s - %(levelname)s - %(message)s",  # Log format
    handlers=[
        logging.FileHandler(log_filename, mode='w'),  # Log to a file
        logging.StreamHandler()  # Log to console
    ]
)

# checks if all .bin files exist
def validate_state(state_dict):
    missing_files = []

    for lang, info in state_dict.items():
        datapath = info.get("datapath", "")
        if not os.path.exists(datapath):  # Check if the file exists
            missing_files.append((lang, datapath))

    if missing_files:
        logging.error("The following datapaths do not exist:")
        for lang, path in missing_files:
            logging.error(f" - {lang}: {path}")
        sys.exit(1)  # Exit with error code 1

    logging.info("Successfully validated state")


# load idx/bin into megatron IndexedDataset class
def load_indexed_dset(path_to_bin):
    # needs dataset prefix, adds ".idx" extension automatically
    return MMapIndexedDataset(path_to_bin.replace(".bin", ""))


# init a builder for output
# FIXME: datatype?
def make_indexed_dset_builder(path_to_bin, dtype=69):
    # needs path to bin file
    return MMapIndexedDatasetBuilder(path_to_bin, dtype=dtype)


# loads state file
def load_state(path_to_state):
    with open(path_to_state, "r", encoding="utf-8") as file:
        state_data = yaml.safe_load(file)

    return state_data


# writes state to a new state file
def write_state(state, path_to_og_state):
    # get current state counter
    # state filename is of the form state.{num}.yaml
    path_to_og_state = path_to_og_state.split(".")
    path_to_og_state[-2] = str(int(path_to_og_state[-2]) + 1)

    logging.info("Saving state to %s" % ".".join(path_to_og_state))
    with open(".".join(path_to_og_state), "w", encoding="utf-8") as yaml_file:
        yaml.dump(state, yaml_file, default_flow_style=False, sort_keys=False)


# yes this is just a copy of the function below
# load yaml containing token amounts
def load_token_counts(path_to_tokens):
    # Load YAML file
    with open(path_to_tokens, "r", encoding="utf-8") as file:
        token_counts = yaml.safe_load(file)  # Safe parsing

    return token_counts


# yes this is just a copy of the function above
# load yaml containing slice names
def load_slice_names(path_to_slices):
    # Load YAML file
    with open(path_to_slices, "r", encoding="utf-8") as file:
        slice_names = yaml.safe_load(file)  # Safe parsing

    return slice_names


def slice_bin(index_offset: int, token_count: int, indexed_dset: MMapIndexedDataset,
              path_to_out_bin: str) -> int:
    """
    Slices the input index dataset on document level, while trying to match desired token count. Outputs a new .bin/.idx file.
    :param index_offset: i.e. start index for the slice from
    :param token_count: number of desired tokens
    :param indexed_dset: input IndexedDataset
    :param path_to_out_bin: path to output .bin file
    :return:
    """

    # init dataset builder
    is_builder_set = False
    # indexed_dset_builder = make_indexed_dset_builder(path_to_out_bin)

    # get total number of documents available in the dataset
    max_idx = indexed_dset.__len__()

    # count sampled tokens
    current_tokens = 0

    current_idx = index_offset

    while current_idx < max_idx:

        # get a document
        temp_tokens = indexed_dset.get(current_idx)  # numpy array

        if not is_builder_set:
            # infer data type
            dtype = temp_tokens.dtype

            # init builder with correct datatype
            indexed_dset_builder = make_indexed_dset_builder(path_to_out_bin, dtype=dtype)

            is_builder_set = True

        # calc length
        temp_len = temp_tokens.size

        current_tokens += temp_len

        # if we haven't exceeded the desired token count yet, add this document to the slice
        if current_tokens <= token_count:

            # add to slice
            indexed_dset_builder.add_item(temp_tokens)  # this already writes to disk
            indexed_dset_builder.end_document()  # each sequence is a separate document

        else:

            # add to slice
            indexed_dset_builder.add_item(temp_tokens)  # this already writes to disk
            indexed_dset_builder.end_document()  # each sequence is a separate document

            # finalize the slice, create idx file for the bin
            indexed_dset_builder.finalize(path_to_out_bin.replace(".bin", ".idx"))

            # some logging
            logging.info("Sampled %s out of %s [%s%s] tokens" % (
                current_tokens, token_count, round(100 * current_tokens / token_count, 2), "%"))

            logging.info("Sliced dataset written to %s[.bin/.idx]" % path_to_out_bin.replace(".bin", ""))

            # current index is consumed (we are always oversampling) and future offset is incremented by 1
            logging.info("Index offset for next slice set to %s" % current_idx + 1)
            return current_idx + 1

        # increment index
        current_idx += 1

        # IMPORTANT: allowed wrapping around
        if current_idx >= max_idx:
            # some logging
            logging.info("Ran out of documents to sample from ...")
            logging.info("Sampled so far: %s out of %s [%s%s] tokens" % (
                current_tokens, token_count, round(100 * current_tokens / token_count, 2), "%"))
            logging.info("Re-setting index to 0 and wrapping around ...")

            current_idx = 0

    # some more logging
    logging.info("Sliced dataset written to %s(.bin/.idx)" % path_to_out_bin.replace(".bin", ""))
    logging.info("Index offset for next slice set to %s" % current_idx)
    return 0


def main(args):
    # load the state
    state = load_state(args.state)
    validate_state(state)

    # load the token counts
    token_counts = load_token_counts(args.tokens)

    # load slice names
    logging.info("Slice YAML file has been provided. Loading")
    slice_names = load_slice_names(args.slice)
    logging.info("Slices: %s" % slice_names)
    assert (sorted(slice_names.keys()) == sorted(token_counts.keys()))
    for key in slice_names.keys():
        assert (len(slice_names[key]) == len(token_counts[key]))

    # some meme sanity
    logging.info(list(state.keys()))
    logging.info(list(sorted(token_counts.keys())))
    logging.info(list(sorted(state.keys())))
    assert (sorted(list(token_counts.keys())) == sorted(list(state.keys())))

    # loop trough languages and slice if possible
    # log the new idx position

    for lang in token_counts.keys():

        # skip languages with no requested tokens
        assert len(token_counts[lang]) > 0

        logging.info("--- Processing %s [%s tokens requested]" % (lang, sum(token_counts[lang])))
        # get the source indexed dataset for this lang
        path_to_inp_bin = state[lang]["datapath"]
        indexed_dataset = load_indexed_dset(path_to_inp_bin)  # load this once

        # grab the current index offset
        idx_offset = state[lang]["idxs"][-1]

        # loop trough requested slices
        for n, slice in enumerate(token_counts[lang]):

            # convert to integer if not already
            slice = int(slice)

            # grab corresponding slice name if it exists
            slice_folder = "/" + slice_names[lang][n] + "_" + str(n)
            os.makedirs(args.out_dir + slice_folder, exist_ok=True)  # a bit hacky and redundant

            # slice the dataset, write output to input_filename_slice{n}_{idx_offset}_{token_count}.bin/.idx
            if slice == 0:
                logging.info(
                    f"Skipping {path_to_inp_bin}, phase {slice_names[lang][n]}, requested {slice} tokens")
                continue

            logging.info(f"Slicing {path_to_inp_bin}, phase {slice_names[lang][n]} for {slice} tokens")
            new_idx_offset = slice_bin(idx_offset, slice, indexed_dataset, args.out_dir + slice_folder + "/" +
                                       os.path.basename(
                                           path_to_inp_bin.replace(".bin", f"_slice{n}_{idx_offset}_{slice}.bin")))

            # update state
            state[lang]["idxs"].append(new_idx_offset)

            # set the next idx_offset as the current new
            idx_offset = new_idx_offset

            # Important: allowed wrapping
            # if idx_offset == 0:
            #     break

    # write the new state
    write_state(state, args.state)

    logging.info("Done.")


if __name__ == "__main__":
    import argparse

    state_yaml_example = """\
    Example state YAML format:
    ----------------------
    lv:
      datapath: /home/user/lv.bin
      idxs: [0]
    en:
      datapath: /home/user/en.bin
      idxs: [0]
    """

    tokens_yaml_example = """\
    Example tokens YAML format:
    ----------------------
    lv: [420,1337]
    en: [69,1488]
    """

    slices_yaml_example = """\
    Example slices YAML format:
    ----------------------
    lv: ['warmup','U1', 'N']
    en: ['warmup', 'U1', 'N']
    """

    # Argument parsing happens here, inside main()
    parser = argparse.ArgumentParser(description="A simple argument parser example.")

    parser.add_argument("--tokens", type=str, required=True,
                        help=f"Path to the YAML tokens file. \n\n{tokens_yaml_example}")
    parser.add_argument("--state", type=str, required=True,
                        help=f"Path to the YAML state file. \n\n{state_yaml_example}")
    parser.add_argument("--slice", type=str, required=True,
                        help=f"Path to the YAML slice names file. \n\n{slices_yaml_example}")
    parser.add_argument("--out_dir", type=str, required=False, default=".", help=f"Path to output directory")

    args = parser.parse_args()

    # Call main with parsed arguments
    main(args)
