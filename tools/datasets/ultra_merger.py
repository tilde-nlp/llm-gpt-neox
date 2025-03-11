#!/usr/bin/env python3

neoxpath = "/project/project_465001281/IP/llm-gpt-neox/"

import re
import argparse
import os
import subprocess


def parse_args():
    parser = argparse.ArgumentParser(description="Merge .bin and .idx files from an input folder, optionally only merges whitelist. Assumes all the files are of the structure ..._<slice index>_<key>_document.<idx/bin>")
    parser.add_argument(
        "--input",
        required=True,
        help="Path to the input folder containing files to be merged."
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Path to the output folder where the merged files will be created."
    )
    parser.add_argument(
        "--whitelist",
        nargs='*',
        default=None,
        help="List of dataset names to restrict the merging. If omitted, all files will be considered."
    )
    return parser.parse_args()


def get_files(folder_path):
    """
    Scans folder and returns dict of list like:

    {"lv": ["lv_0_text_document", "lv_1_text_document", "lv_2_text_documen"],
     "en": ["en_0_text_document"],
     "ru": ["ru_0_text_document", "ru_1_text_document"]}

    Returns full paths though not just the file name.
    """
    pattern = re.compile(r"^(.*)_(\d+)_[^_]+_document\.(bin)$")

    prefixes = {}

    for filename in os.listdir(folder_path):
        match = pattern.match(filename)
        if not match:
            continue

        dataset_name, slice_str, extension = match.groups()
        slice_num = int(slice_str)
        full_path = os.path.join(folder_path, filename)

        if dataset_name not in prefixes:
            prefixes[dataset_name] = []
        prefixes[dataset_name].append(full_path[:-4])

    # Sort lists
    for dataset in prefixes:
        prefixes[dataset].sort(key = lambda x: x[0])

    return prefixes


def main():
    args = parse_args()

    print("Mergingin .bins from", args.input, "to", args.output, " with whitelist", args.whitelist)

    datasets = [] # List of dataset of list of slice.

    print("Listing all the relevant files")
    files = get_files(args.input)

    if args.whitelist is not None:
        print("Filtering out only stuff in the whitelist.")
        tmp = {}
        for dataset in files:
            if dataset in args.whitelist:
                tmp[dataset] = files[dataset]
        files = tmp

    processes = []

    #subprocess.run(["module", "purge"])
    #subprocess.run(["module", "use", "/appl/local/training/modules/AI-20240529/"])
    #subprocess.run(["module", "load", "singularity-userfilesystems"])
    for dataset in files:
        print("Merging ", dataset)
        cmd = ["srun", "--account=project_465001281", "--partition=small-g", "--gpus-per-node=1", "--ntasks-per-node=1", "--cpus-per-task=7", "--mem-per-gpu=60G", "--time=1:00:00", "--nodes=1"]
        cmd+= ["singularity", "exec", "-B", "/scratch:/scratch", "-B", "/project:/project", "/scratch/project_465001281/containers/rocm603_flash.sif"]
        #print(files[dataset])
        cmd+= ["bash", "-c", "$WITH_CONDA ; python " + neoxpath + "tools/datasets/cursed_merge_datasets.py --input " + args.input + " --output-prefix " + os.path.join(args.output, dataset) + " --file-list " + " ".join(files[dataset])]
        #print(cmd)

        processes.append((dataset, subprocess.Popen(cmd)))

    for p in processes:
        exit_code = p[1].wait()
        if exit_code != 0:
            print("Merging", p[0], "process crashed :(")
        else:
            print("Mergining", p[0], "sucessful :)")

    print("Done merging.")


if __name__ == "__main__":
    main()
