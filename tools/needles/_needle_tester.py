"""
Doesnt sniff, just benches a single ckpt

Script assumes you're using a sentencepiece tokenizer model.
"""

import yaml
import argparse
import os
import shutil
import string
import json
import time
import random
import numpy as np
import sys
import subprocess

from transformers import GPTNeoXForCausalLM, LlamaTokenizer, LlamaForCausalLM
import torch
from accelerate import init_empty_weights, load_checkpoint_and_dispatch

# Doing magic so prints flush immediately.
import builtins

original_print = builtins.print
builtins.print = lambda *args, **kwargs: original_print(*args, flush=True, **kwargs)

import niah


def parse_args():
    """
    Parses commandline arguments and returns them.
    """

    parser = argparse.ArgumentParser(
        description="Script that performs needle tests on a gpt-neox checkpoint."
    )

    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Paths to .yml file being used for gptneox training.py"
    )

    # This arg exists only for conevnience and is not used by the script.
    parser.add_argument(
        "--container",
        type=str,
        default=".",
        help=argparse.SUPPRESS
    )

    parser.add_argument(
        "--architecture",
        type=str,
        default='llama',
        help="Huggingface model to convert to. Options are 'neox' or 'llama' for NeoxForCausalLM and LlamaForCausalLM respectively. " +
             "Defaults to 'llama'. 'neox' probably doesn't work anymore anyway."
    )

    parser.add_argument(
        "--ckpt-path",
        type=str,
        required=True,
        help="Path to checkpoint (global_stepX folder)."
    )

    parser.add_argument(
        "--hay-path",
        type=str,
        required=True,
        help='Path to .jsonl file that contains hay samples. Hay samples should be dictionaries with a "text" key. The potential needle insertion locations should be marked with "<<|needle|>>" in the text.'
    )
    
    parser.add_argument(
        "--needle-path",
        type=str,
        required=True,
        help='Path to .json file with needles. Should contain a list of needles. e.g. [{"needle_raw": "George Smith is Marcus Smith\'s brother. ", "answer_prompt_raw": "Marcus Smith\'s brother is ", "answer_raw": "George"}]. needle_raw will be inserted in the hay; answer_prompt_raw will be appended to the hay; the code will look for answer_raw in the generated text.'
    )

    parser.add_argument(
        "--max-context-length",
        type=int,
        default=8192,
        help="Code will cut hay to this size or undershoot. (excludes needle texts)."
    )

    parser.add_argument(
        "--context-length-test-count",
        type=int,
        default=10,
        help="How many different context lengths to test. Will perform a linspace with this many steps (excluding 0)."
    )

    parser.add_argument(
        "--depth-test-count",
        type=int,
        default=11,
        help="How many percentages to insert the needle at. Will do a linspace with this many steps from 0%% to 100%%."
    )

    parser.add_argument(
        "--log-folder",
        type=str,
        required=True,
        help="Path to folder where results will be written. Will output results.json and debug.json. Will create the folder if it doesn't exist already."
    )

    parser.add_argument(
        "--tmp-path",
        type=str,
        required=True,
        help="Path to 'root' temporary folder for hf checkpoints."
    )

    parser.add_argument(
        "--neox-path",
        type=str,
        required=True,
        help="Path to llm-gpt-neox folder."
    )

    args = parser.parse_args()

    must_exist = {
        "config": args.config,
        "ckpt-path": args.ckpt_path,
        "hay-path": args.hay_path,
        "needle-path": args.needle_path,
        "neox-path": args.neox_path,
    }

    for name, path in must_exist.items():
        if not os.path.exists(path):
            print(f"Error: --{name} '{path}' doesn't exist.")
            sys.exit(1)

    if not os.path.exists(args.log_folder):
        os.makedirs(args.log_folder)

    return args


def convert_checkpoint(step_path: str, output_path: str, config: dict, GPT_NEOX_PATH: str, model_type: str = "Neox") -> None:
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
    # get repo
    GPT_NEOX_PATH = args.neox_path

    
    # Convert model to HF.
    tmp_path_root = create_temp_subfolder(args.tmp_path)
    ckpt = args.ckpt_path
    tmp_path = os.path.join(tmp_path_root, os.path.basename(ckpt))
    os.makedirs(tmp_path, exist_ok=True)
    print(" ----- Converting checkpoint: %s" % ckpt)
    convert_checkpoint(ckpt, tmp_path, config, GPT_NEOX_PATH, args.architecture)

    
    # Load HF model.
    if args.architecture.upper() == "NEOX":
        model = GPTNeoXForCausalLM.from_pretrained(tmp_path, device_map="auto")
    elif args.architecture.upper() == "LLAMA":
        # FIXME: hardcoded, i think this might not work from models that dont fit into one gpu
        model = LlamaForCausalLM.from_pretrained(tmp_path, device_map="auto")
    else:
        raise ValueError("Huggingface --architecture " + str(args.architecture) + " not recgonized.")
    # model = load_checkpoint_and_dispatch(model, tmp_path, device_map = "auto", no_split_module_classes=['Block'])
    print(model.hf_device_map)
    print("First param dtype:", next(model.parameters()).dtype)
    model = model.to(torch.bfloat16)
    print("First param after conversion dtype:", next(model.parameters()).dtype)

    
    # Load tokenizer.
    print("Loading tokenizer from %s " % config["vocab_file"])
    print(" ---------- REMOVING BOS ------------ ")
    tokenizer = LlamaTokenizer.from_pretrained(config["vocab_file"], add_bos_token=False)
    print("Finished loading tokenizer.")

    
    # Load needles.
    needles = niah.load_needles(args.needle_path, tokenizer)


    # Prepare tests.
    tests = [] # List(context length)[List(insertion depth)[List(hay sample)[List(needle)[List[int]]]]]
    ctx_step_size = args.max_context_length / args.context_length_test_count
    depth_step_size = 1 / (args.depth_test_count - 1)
    target_contexts = [(i + 1) * ctx_step_size for i in range(args.context_length_test_count)]
    target_depths = [i * depth_step_size for i in range(args.depth_test_count)]
    for ctx_step in range(args.context_length_test_count):
        # Load hay.
        hay_cap = ctx_step_size * (ctx_step + 1)
        haystacks = niah.load_hay(args.hay_path, tokenizer, hay_cap)
        
        tmp3 = []
        for depth_step in range(args.depth_test_count):
            depth = depth_step_size * depth_step
            tmp2 = []
            for hay in haystacks:
                tmp = []
                for needle in needles: 
                    tmp.append(niah.get_sample(hay, needle, depth))
                tmp2.append(tmp)
            tmp3.append(tmp2)
        tests.append(tmp3)
    
    debug = np.zeros([args.context_length_test_count, args.depth_test_count, len(haystacks), len(needles)]).tolist()
    # Make tests tensors on correct device device.
    for idx1, obj1 in enumerate(tests):
        for idx2, obj2 in enumerate(obj1):
            for idx3, obj3 in enumerate(obj2):
                for idx4, obj4 in enumerate(obj3):
                    tests[idx1][idx2][idx3][idx4] = (
                        torch.tensor([obj4[0]], dtype=torch.long, device=next(model.parameters()).device), 
                        obj4[1]
                    )
                    debug[idx1][idx2][idx3][idx4] = obj4[1]

    
    # Evaluate checkpoint.
    scores = np.zeros([args.context_length_test_count, args.depth_test_count, len(haystacks), len(needles)])
    for idx1, obj1 in enumerate(tests):
        print("Needle testing progress: ", idx1, "/", args.context_length_test_count)
        for idx2, obj2 in enumerate(obj1):
            for idx3, obj3 in enumerate(obj2):
                for idx4, obj4 in enumerate(obj3):
                    print(idx1, idx2, idx3, idx4)
                    with torch.no_grad():
                        output_ids = model.generate(obj4[0], max_new_tokens=15, temperature=0, do_sample=False)[0]
                    output_text = tokenizer.decode(output_ids[len(obj4[0][0]):])
                    #print(output_text)
                    if needles[idx4]["answer_raw"] in output_text:
                        scores[idx1, idx2, idx3, idx4] = 1.0
                    else:
                        scores[idx1, idx2, idx3, idx4] = 0.0
                    #print(scores[idx1, idx2, idx3, idx4])
                    #print()
    # Log needle results
    os.makedirs(args.log_folder, exist_ok=True)
    results_path = os.path.join(args.log_folder, "results.json")
    log_needles = []
    for needle in needles:
        log_needles.append({"needle": needle["needle_raw"], "answer_prompt": needle["answer_prompt_raw"], "answer": needle["answer_raw"]})
    with open(results_path, "w") as f:
        result = {
            "scores": scores.tolist(),
            "target_depths": target_depths,
            "target_contexts": target_contexts,
            "needles": log_needles
        }
        print(json.dumps(result), file=f)
    
    # Log debug stuff
    debug_path = os.path.join(args.log_folder, "debug.json")
    with open(debug_path, "w") as f:
        print(json.dumps(debug), file=f)

    del model

    # Clean up.
    print("Deleting temporary HF checkpoint from %s " % tmp_path)
    shutil.rmtree(tmp_path_root)


if __name__ == "__main__":
    main()
