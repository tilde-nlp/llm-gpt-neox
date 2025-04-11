import sys
import os
from tqdm import tqdm
import numpy as np
from megatron.data.indexed_dataset_hacked import MMapIndexedDataset, MMapIndexedDatasetBuilder


# load idx/bin into megatron IndexedDataset class
def load_indexed_dset(path_to_bin):
    # needs dataset prefix, adds ".idx" extension automatically
    return MMapIndexedDataset(path_to_bin.replace(".bin", ""))


# init a builder for output
def make_indexed_dset_builder(path_to_bin, dtype=69):
    # needs path to bin file
    return MMapIndexedDatasetBuilder(path_to_bin, dtype=dtype)


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} /path/to/input_file.bin")
        sys.exit(1)

    # Get input path from command line argument
    path_to_inp_bin = sys.argv[1]

    # Construct output file path: prepend "eod_" to the input file's base name
    dir_name = os.path.dirname(path_to_inp_bin)
    base_name = os.path.basename(path_to_inp_bin)
    output_base_name = "eod_" + base_name
    path_to_out_bin = os.path.join(dir_name, output_base_name)

    # load the dataset
    indexed_dset = load_indexed_dset(path_to_inp_bin)

    is_builder_set = False

    # iterate through the documents
    for current_idx in tqdm(range(0, indexed_dset.__len__())):
        # get a document's tokens
        temp_tokens = indexed_dset.get(current_idx)
        # infer data type from current document tokens
        dtype = temp_tokens.dtype

        if not is_builder_set:
            # initialize builder with correct datatype
            indexed_dset_builder = make_indexed_dset_builder(path_to_out_bin, dtype=dtype)
            is_builder_set = True

        # Ensure tokens are in numpy array format with proper dtype
        temp_tokens = np.array(temp_tokens, dtype=dtype)
        # Define EOD token as an array (the example uses token 48)
        tokens_eod = np.array([48], dtype=dtype)
        # Concatenate document tokens with EOD token
        tokens_output = np.concatenate((temp_tokens, tokens_eod))

        # Add item (document) to builder and mark the document ending
        indexed_dset_builder.add_item(tokens_output)  # this writes to disk immediately
        indexed_dset_builder.end_document()

    # Finalize and create the corresponding index file (".idx")
    indexed_dset_builder.finalize(path_to_out_bin.replace(".bin", ".idx"))
