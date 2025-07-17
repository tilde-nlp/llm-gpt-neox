import sys
import argparse
import subprocess
import os


# Fetch path of needle folder
file_path = os.path.abspath(__file__)
filename_len = len(file_path.split("/")[-1])
folder_path = file_path[:-filename_len]


def parse_args():
    """
    Parses commandline arguments and returns them.
    """

    parser = argparse.ArgumentParser(
        description="Script that performs needle tests on a gpt-neox checkpoint. (includes launching slurm job and singularity container)"
    )

    parser.add_argument(
        "--container",
        type=str,
        default="/scratch/project_465001281/IP/sbox_rocm603_inference",
        help="Singularity container to run job in. default: /scratch/project_465001281/IP/sbox_rocm603_inference"
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
        default=folder_path + "hay/paulgram.jsonl",
        help='Path to .jsonl file that contains hay samples. Hay samples should be dictionaries with a "text" key. The potential needle insertion locations should be marked with "<<|needle|>>" in the text. Default: %(default)s'
    )

    parser.add_argument(
        "--needle-path",
        type=str,
        default=folder_path + "needles/the_needle.json",
        help='Path to .json file with needles. Should contain a list of needles. e.g. [{"needle_raw": "George Smith is Marcus Smith\'s brother. ", "answer_prompt_raw": "Marcus Smith\'s brother is ", "answer_raw": "George"}]. needle_raw will be inserted in the hay; answer_prompt_raw will be appended to the hay; the code will look for answer_raw in the generated text. Default: %(default)s'
    )

    parser.add_argument(
        "--max-context-length",
        type=int,
        default=8192,
        help="Code will cut hay to this size or undershoot. (excludes needle texts). Default: %(default)s"
    )

    parser.add_argument(
        "--context-length-test-count",
        type=int,
        default=10,
        help="How many different context lengths to test. Will perform a linspace with this many steps (excluding 0). Default: %(default)s"
    )

    parser.add_argument(
        "--depth-test-count",
        type=int,
        default=11,
        help="How many percentages to insert the needle at. Will do a linspace with this many steps from 0%% to 100%%. Default: %(default)s"
    )

    parser.add_argument(
        "--log-folder",
        type=str,
        required="./needle_log/",
        help="Path to folder where results will be written. Will output results.json and debug.json. Will create the folder if it doesn't exist already. Default: %(default)s"
    )

    parser.add_argument(
        "--tmp-path",
        type=str,
        default="./needle_temp/",
        help="Path to 'root' temporary folder for hf checkpoints. Default: %(default)s"
    )

    parser.add_argument(
        "--neox-path",
        type=str,
        default="/project/project_465001281/IP/testing2-llm-gpt-neox/llm-gpt-neox/",
        help="Path to llm-gpt-neox folder For checkpoint converting purpouses. Default: %(default)s"
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


def main():
    args = parse_args()

    container_path = args.container

    #Fetch arg string.
    arg_string = ""
    argdict = vars(args)
    for k in argdict:
      arg_string+= " --" + k.replace("_", "-") + " " + str(argdict[k])

    # Command for running actual needle tester script.
    needle_command = "python3 -u " + folder_path + "_needle_tester.py" + arg_string
    # Since singularity runs a single command but we want to activate conda, have to wrap in bash to activate the conda environement.
    bash_command = ["bash", "-c", "$WITH_CONDA ; " + needle_command]
    # Wrap command in singularity
    singularity_command = ["singularity", "exec",
                             "-B", "/scratch",
                             "-B", "/project",
                             "-B", "/appl",
                             "-B", "/projappl",
                             "-B", "/pfs",
                             container_path] + bash_command
    # Wrap command in srun
    srun_command = ["srun",
                      "--job-name=needle_test",
                      "--account=project_465001281",
                      "--partition=dev-g",
                      "--gpus-per-node=8",
                      "--ntasks-per-node=1",
                      "--cpus-per-task=56",
                      "--mem=0",
                      "--time=3:00:00",
                      "--nodes=1"] + singularity_command

    print(srun_command)
    subprocess.run(srun_command)


if __name__ == "__main__":
    main()
