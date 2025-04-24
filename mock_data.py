import sys
import os
from tqdm import tqdm
import numpy as np
from megatron.data.indexed_dataset_hacked import MMapIndexedDataset, MMapIndexedDatasetBuilder

# init a builder for output
def make_indexed_dset_builder(path_to_bin, dtype=69):
    # needs path to bin file
    return MMapIndexedDatasetBuilder(path_to_bin, dtype=dtype)


def generate_mock_documents(n, m):
    documents = []
    token_seed = 1  # starting point for token sequences

    for _ in range(n):
        doc_length = random.randint(2, m)  # minimum 2: start + end
        start_token = token_seed
        middle_tokens = list(range(start_token + 1, start_token + doc_length - 1))
        document = [start_token] + middle_tokens + [start_token]
        documents.append(document)
        token_seed += doc_length  # ensure next doc starts fresh and avoids overlaps

    return documents

# load idx/bin into megatron IndexedDataset class
def load_indexed_dset(path_to_bin):
    # needs dataset prefix, adds ".idx" extension automatically
    return MMapIndexedDataset(path_to_bin.replace(".bin", ""))

if __name__ == '__main__':

    import random


    if len(sys.argv) < 4:
        print(f"Usage: {sys.argv[0]} /path/to/output_file.bin num_docs [int] max_doc_len [int]")
        sys.exit(1)

    # Get input path from command line argument
    path_to_out_bin = sys.argv[1]
    num_docs = int(sys.argv[2])
    max_doc_len = int(sys.argv[3])

    # initialize builder with correct datatype
    # dtype = np.int64

    # apparently int64 doesn't cut it
    # so try to hardcode infer
    path_to_inp_bin = "/scratch/project_465001281/tokenized/final_data_sliced/warmup_0/ltg_slice0_0_342338.bin"
    indexed_dset = load_indexed_dset(path_to_inp_bin)
    # get a document's tokens
    temp_tokens = indexed_dset.get(1)
    # infer data type from current document tokens
    dtype = temp_tokens.dtype
    print(f"dtype: {dtype}")

    del indexed_dset

    indexed_dset_builder = make_indexed_dset_builder(path_to_out_bin, dtype=dtype)

    # generate mock docs
    mock_docs = generate_mock_documents(num_docs, max_doc_len)

    # instruction tokens
    tokens_instruct = [111111, 1337, 420, 1337, 1337, 111111]

    # iterate through the documents
    for temp_tokens in tqdm(mock_docs):

        # prepend mock instructions
        temp_tokens = tokens_instruct + temp_tokens
        # Ensure tokens are in numpy array format with proper dtype
        tokens_output = np.array(temp_tokens, dtype=dtype)

        # Add item (document) to builder and mark the document ending
        indexed_dset_builder.add_item(tokens_output)  # this writes to disk immediately
        indexed_dset_builder.end_document()

    # Finalize and create the corresponding index file (".idx")
    indexed_dset_builder.finalize(path_to_out_bin.replace(".bin", ".idx"))
