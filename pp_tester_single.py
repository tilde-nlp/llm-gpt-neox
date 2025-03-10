"""
Doesnt sniff, just benches a single ckpt

Script assumes you're using a sentencepiece tokenizer model.
"""

GPT_NEOX_PATH = "/project/project_465001281/llm-gpt-neox"

import yaml
import argparse
import os
import shutil
import subprocess
import string
import json
import csv
import math
import time
import random

from transformers import GPTNeoXForCausalLM, LlamaTokenizer, LlamaForCausalLM
import torch
import torch.nn.functional as f
from accelerate import init_empty_weights, load_checkpoint_and_dispatch

# Doing magic so prints flush immediately.
import builtins

original_print = builtins.print
builtins.print = lambda *args, **kwargs: original_print(*args, flush=True, **kwargs)


def parse_args():
    """
    Parses commandline arguments and returns them.
    """

    parser = argparse.ArgumentParser(
        description="A script that processes some data with optional parameters."
    )

    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Paths to .yml file being used for gptneox training.py"
    )

    parser.add_argument(
        "--architecture",
        type=str,
        default='neox',
        help="Huggingface model to convert to. Options are 'neox' or 'llama' for NeoxForCausalLM and LlamaForCausalLM respectively. " +
             "Defaults to 'neox'."
    )

    parser.add_argument(
        "--ckpt_path",
        type=str,
        required=True,
        help="Path to checkpoint."
    )

    parser.add_argument(
        "--test-folder",
        type=str,
        required=True,
        help="Path to folder with .jsonl test data files. " +
             "Each line should be a dict with a 'text' key, which should contain the text to test on."
    )

    parser.add_argument(
        "--log-file",
        type=str,
        required=True,
        help="Path to .csv file to log to. Appends to existing, if it already exists."
    )

    parser.add_argument(
        "--tmp-path",
        type=str,
        required=True,
        help="Path to 'root' temporary folder for hf checkpoints."
    )

    args = parser.parse_args()

    # Since they're parsed now, let's validate them.
    # Todo.
    pass

    return args


def load_jsonl(path: str) -> list[str]:
    """
    Loads contents of a datset.

    :path: Path to a jsonl file.
           The file should contain dictionaries.
           Each dictionary should be of the format {"text": "This is a test sample."}
    :return: The test samples.
    """
    print("Opening", path)
    with open(path, "r") as file:
        data = []

        # Parse samples line by line.
        for line in file:
            sample = json.loads(line)
            data.append(sample["text"])

    print("File contained", len(data), "samples.")

    return data


def load_datasets(test_folder: str) -> tuple[list[str], list[list[str]]]:
    """
    Loads datasets.
    Function currently looks into test_folder for .jsonl files, and returns their parsed contents.

    :test_folder:
    :return: Tuple of:
               list of dataset names,
               list [dataset] of list of samples.
    """
    # Filter the jsonl files.
    paths_unfiltered: list[str] = os.listdir(test_folder)
    paths: list[str] = []  # Will contain only the .jsonl files.
    for path in paths_unfiltered:
        if path[-6:] == ".jsonl":
            paths.append(path)
    print("Detected", len(paths), "different test files in test folder.")

    # Get the dataset names.
    names: list[str] = []
    for path in paths:
        # We'll use the filename without extensions as the names.
        file_name = os.path.basename(path)
        name = os.path.splitext(path)[0]
        names.append(name)

    # Load the files.
    print("Loading test datasets.")
    datasets: list[list[str]] = []
    for path in paths:
        data = load_jsonl(os.path.join(test_folder, path))
        datasets.append(data)

    print("Finished loading test datasets.")

    return names, datasets


def tokenize_dataset(data: list[str], tokenizer) -> list[list[int]]:
    """
    Tokenizes the dataset.
    """
    tokens = [tokenizer.encode(sample) for sample in data]
    return tokens


def open_log_file(path: str, names: list[str]):
    """
    Gets an opened log file.
    Makes a new one if it doesn't exist.
    """
    print("Opening log file.")
    if os.path.isfile(path):
        print("Log file already exists. Assuming I just need to append to it.")
        return open(path, "a")
    else:
        print("Log file doesn't exist. Will just create a new one.")
        dir_path = os.path.dirname(path) or "."
        os.makedirs(dir_path, exist_ok=True)
        file = open(path, "w")

        # Gotta create the header before returning to user.
        writer = csv.writer(file)
        columns = ["time", "step"]
        for name in names:
            columns.append(name + "_TotalCE")
            columns.append(name + "_CE")
            columns.append(name + "_PPL")
        writer.writerow(columns)

        return file


def convert_checkpoint(step_path: str, output_path: str, config: dict, model_type: str = "Neox") -> None:
    """
    Converts gpt-neox checkpoint to huggingface checkpoint.

    :param step_path: Path to a checkpoint, including the step number.
    :param configs: Condensed config from all the .yml files that were used during training.
    :param output_path: Path to save converted checkpoint to.
    :param model_type: "Llama" or "Neox" Huggingface side model to convert to.
    """

    print("Converting checkpoint.")
    command = ["python", os.path.join(GPT_NEOX_PATH, "tools/ckpts/convert_neox_to_hf.py")]
    command += ["--input_dir", step_path]
    command += ["--output_dir", output_path]

    os.makedirs("./tmp/", exist_ok=True)
    with open("./tmp/conf.yml", "w") as file:
        yaml.dump(config, file)
    command += ["--config_file", "./tmp/conf.yml"]

    if model_type.upper() == "NEOX":
        command += ["--architecture", "neox"]
    elif model_type.upper() == "LLAMA":
        command += ["--architecture", "llama"]

    print("Running command: ", " ".join(command))

    result = subprocess.run(command)

    if result.returncode != 0:
        raise Exception("Checkpoint conversion failed.")

    print("Converted the checkpoint.")


def get_cross_entropy(model, dataset: list[torch.Tensor]) -> tuple[float, float]:
    """
    :model: Huggingface model.
    :tokens: Tokenized data.
    """
    # if True:
    with torch.no_grad():
        token_count = 0
        cummulative_CE = 0
        for sample in dataset:
            result = model.forward(sample.view(1, -1))
            sample_len = sample.shape[0] - 1
            CE = f.cross_entropy(result.logits.permute(0, 2, 1)[:, :, :-1], sample[1:].view(1, -1))
            token_count += sample_len
            cummulative_CE += CE * sample_len
        return cummulative_CE.cpu().item(), (cummulative_CE / token_count).cpu().item()


def get_config(config_file_path: str) -> dict:
    """
    Loads gpt neox train config yaml into python dict.
    """
    print("Reading configuration file %s." % config_file_path)
    config = {}  # Will merge all configs into this.
    with open(config_file_path, "r") as conf_file:
        conf = yaml.safe_load(conf_file)
        for key in conf:
            if key in config:
                raise ValueError("Duplicate key in config! The bad key is: " + str(key))
            config[key] = conf[key]

    return config


def create_temp_subfolder(root_temp_dir):
    # Generate a unique folder name using processor time, three random letters, and three random digits
    letters = ''.join(random.choices(string.ascii_lowercase, k=3))  # 3 random lowercase letters
    numbers = ''.join(random.choices(string.digits, k=3))  # 3 random digits
    unique_id = f"{int(time.process_time() * 1e6)}_{letters}{numbers}"

    temp_subfolder = os.path.join(root_temp_dir, unique_id)

    # Create the directory if it doesn't exist
    os.makedirs(temp_subfolder, exist_ok=True)

    return temp_subfolder


def main():
    args = parse_args()

    # Snag configs
    config: dict = get_config(args.config)

    # Validate config files.
    if "save" not in config:
        raise ValueError(
            "%s didn't contain the 'save' key, so code can't infer where the checkpoint folder is." % args.config)

    # Validate tokenizer
    pass

    names, datasets = load_datasets(args.test_folder)

    print("Loading tokenizer from %s " % config["vocab_file"])
    print(" ---------- REMOVING BOS ------------ ")
    tokenizer = LlamaTokenizer.from_pretrained(config["vocab_file"], add_bos_token=False)
    print("Finished loading tokenizer.")

    # Tokenize data.
    print("Tokenizing test datasets.")
    datasets = [tokenize_dataset(dataset, tokenizer) for dataset in datasets]
    print("Finished tokenizing test datasets.")

    # Turn datasets into tensors.
    # We'll have a list [dataset] of list [sample] of tensors.
    datasets_ = []
    for dataset in datasets:
        dataset_ = []
        for sample in dataset:
            # FIXME: might cause issues if model doesnt fit into single gpu
            dataset_.append(
                torch.tensor(sample, dtype=torch.long, device="cuda:0"))  # Not really sure how to choose the cuda here.
        datasets_.append(dataset_)
    datasets = datasets_

    # Open our logging CSV file.
    log_file = open_log_file(args.log_file, names)
    log_file_writer = csv.writer(log_file)

    # create a unique temp folder in the root temp folder
    tmp_path = create_temp_subfolder(args.tmp_path)

    # Look for untested checkpoints
    ckpt = args.ckpt_path

    # create a subfolder for the specific ckpt
    tmp_path = os.path.join(tmp_path, os.path.basename(ckpt))
    os.makedirs(tmp_path, exist_ok=True)

    print(" ----- New checkpoint detected: %s" % ckpt)

    # Convert to hugginface checkpoint format.
    convert_checkpoint(ckpt, tmp_path, config, args.architecture)

    # Load model.
    with init_empty_weights():
        if args.architecture.upper() == "NEOX":
            model = GPTNeoXForCausalLM.from_pretrained(tmp_path, device_map="auto")
        elif args.architecture.upper() == "LLAMA":
            # FIXME: hardcoded, i think this might not work from models that dont fit into one gpu
            model = LlamaForCausalLM.from_pretrained(tmp_path, device_map={"": "cuda:0"})
        else:
            raise ValueError("Huggingface --architecture " + str(args.architecture) + " not recgonized.")
    # model = load_checkpoint_and_dispatch(model, tmp_path, device_map = "auto", no_split_module_classes=['Block'])
    print(model.hf_device_map)

    # Evaluate checkpoint.
    print("Performing testing.")
    results = []  # Will contain our testing results.
    results.append(time.time())
    results.append(os.path.basename(ckpt))
    for test_index in range(len(datasets)):
        totalCE, averageCE = get_cross_entropy(model, datasets[test_index])
        results.append(totalCE)
        results.append(averageCE)
        results.append(math.e ** results[-1])
        print(names[test_index], ": Perplexity", results[-1], "; Total crossentropy", results[-3])
    print("Finished evaluating testing.")

    del model

    # Log.
    log_file_writer.writerow(results)
    log_file.flush()

    # Clean up.
    print("Deleting temporary HF checkpoint from %s " % tmp_path)
    shutil.rmtree(tmp_path)


if __name__ == "__main__":
    main()
