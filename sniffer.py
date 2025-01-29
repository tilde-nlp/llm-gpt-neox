"""
This code just evaluates models during training.
It sniffs for new checkpoint, evaluates them, and logs the results.

Script assumes you're using a sentencepiece tokenizer model.
"""

GPT_NEOX_PATH = "/project/project_465001281/IP/llm-gpt-neox/"

import yaml
import argparse
import time
import os
import shutil
import re
import subprocess
import json
import csv
import math

from transformers import GPTNeoXForCausalLM, LlamaTokenizer, LlamaForCausalLM
import torch
import torch.nn.functional as f
from accelerate import init_empty_weights, load_checkpoint_and_dispatch

#Doing magic so prints flush immediately.
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
    "--configs",
    type = str,
    nargs = "+",
    required = True,
    help = "Paths to .yml files being used for gptneox training.py"
  )

  parser.add_argument(
    "--architecture",
    type = str,
    default = 'neox',
    help = "Huggingface model to convert to. Options are 'neox' or 'llama' for NeoxForCausalLM and LlamaForCausalLM respectively. " +
           "Defaults to 'neox'."
  )

  parser.add_argument(
    "--test-folder",
    type = str,
    required = True,
    help = "Path to folder with .jsonl test data files. " +
           "Each line should be a dict with a 'text' key, which should contain the text to test on."
  )

  parser.add_argument(
    "--log-file",
    type = str,
    default = "logs.csv",
    help = "Path to .csv file to log to. Appends to existing, if it already exists."
  )

  parser.add_argument(
    "--tmp-path",
    type = str,
    default = "./tmp/hf_checkpoint/",
    help = "Path to save temporary hf checkpoints to."
  )

  args = parser.parse_args()

  #Since they're parsed now, let's validate them.
  #Todo.
  pass
  
  return args


def get_merged_config(config_file_paths: list[str]) -> dict:
  """
  Reads list of gpt neox config yml files and merges them into one python dict.
  """
  print("Reading and merging .yml configs.")
  config = {} #Will merge all configs into this.
  for conf_path in config_file_paths:
    print("Opening " + conf_path)
    with open(conf_path, "r") as conf_file:
      conf = yaml.safe_load(conf_file)
      for key in conf:
        if key in config:
          raise ValueError("Duplicate key in configs! The bad key is: " + str(key))
        config[key] = conf[key]
  print("Done with the configs.")
  
  return config


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
    
    #Parse samples line by line.
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
  #Filter the jsonl files.
  paths_unfiltered: list[str] = os.listdir(test_folder)
  paths: list[str] = [] #Will contain only the .jsonl files.
  for path in paths_unfiltered:
    if path[-6:] == ".jsonl":
      paths.append(path)
  print("Detected", len(paths), "different test files in test folder.")
  
  #Get the dataset names.
  names: list[str] = []
  for path in paths: 
    #We'll use the filename without extensions as the names.
    file_name = os.path.basename(path)
    name = os.path.splitext(path)[0]
    names.append(name)
    
  #Load the files.
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
    os.makedirs(os.path.dirname(path), exist_ok = True)
    file = open(path, "w")
    
    #Gotta create the header before returning to user.
    writer = csv.writer(file)
    columns = ["time", "step"]
    for name in names:
      columns.append(name + "_TotalCE")
      columns.append(name + "_CE")
      columns.append(name + "_PPL")
    writer.writerow(columns)
    
    return file


def convert_checkpoint(step_path: str, output_path: str, config: dict, model_type : str = "Neox") -> None:
  """
  Converts gpt-neox checkpoint to huggingface checkpoint.

  :param step_path: Path to a checkpoint, including the step number.
  :param configs: Condensed config from all the .yml files that were used during training.
  :param output_path: Path to save converted checkpoint to.
  :param model_type: "Llama" or "Neox" Huggingface side model to convert to.
  """

  print("Converting checkpoint.")
  command = ["python", os.path.join(GPT_NEOX_PATH, "tools/ckpts/convert_neox_to_hf.py")]
  command+= ["--input_dir", step_path]
  command+= ["--output_dir", output_path]

  os.makedirs("./tmp/", exist_ok = True)
  with open("./tmp/conf.yml", "w") as file:
    yaml.dump(config, file)
  command+= ["--config_file", "./tmp/conf.yml"]

  if model_type.upper() == "NEOX":
    command+= ["--architecture", "neox"]
  elif model_type.upper() == "LLAMA":
    command+= ["--architecture", "llama"]

  print("Running command: ", " ".join(command))

  result = subprocess.run(command)

  if result.returncode != 0:
    raise Exception("Checkpoint conversion failed.")

  print("Converted the checkpoint.")


def get_latest_tested(cp_path: str) -> int:
  """
  Gets the step number of the most recently evaluated checkpoint.

  :param cp_path: Path to checkpoint folder (including folder itself).
  :note: If none have been tested, returns -1.
  """

  file_path = os.path.join(cp_path, "latest_tested.txt")
  if os.path.isfile(file_path):
    with open(file_path, "r") as file:
      return int(file.read())
  else:
    return -1


def set_latest_tested(cp_path: str, step: int):
  """
  Saves the latest tested checkpoint step number.

  :param cp_path: Path to checkpoint folder (including folder itself).
  :param step: Step number to write to file.
  """
  file_path = os.path.join(cp_path, "latest_tested.txt")
  with open(file_path, "w") as file:
    print(step, file = file)


def get_latest_checkpoint(cp_path) -> int:
  """
  Returns step number of latest checkpoint.

  :param cp_path: Path to checkpoint folder (including folder itself).
  :note: If there's no checkpoints, just returns -1.
  """
  #Just in case check if the folder even exists.
  if not os.path.isdir(cp_path):
    return -1

  #Get all folder/file names in cp_path.
  files = os.listdir(cp_path)

  #Find max checkpoint step number.
  mx = -1
  number_suffix_pattern = re.compile(r'\d+$')
  for file in files:
    match = number_suffix_pattern.search(file)
    if match:
      mx = max(mx, int(match.group()))

  return mx


def new_checkpoint(cp_path) -> tuple[bool, int]:
  """
  Checks whether there's a new checkpoint.

  :param cp_path: Path to checkpoint folder (including folder itself).
  :return: Tuple where first elements is True/False for whether there's a checkpoint.
           Second number is just the latest checkpoint or -1 if there is none.
  """
  latest_checkpoint = get_latest_checkpoint(cp_path)
  if latest_checkpoint != get_latest_tested(cp_path):
    return (True, latest_checkpoint)
  else:
    return (False, latest_checkpoint)

args = None


def get_cross_entropy(model, dataset: list[torch.Tensor]) -> tuple[float, float]:
  """
  :model: Huggingface model.
  :tokens: Tokenized data.
  """
  #if True:
  with torch.no_grad():
    token_count = 0
    cummulative_CE = 0
    for sample in dataset:
      result = model.forward(sample.view(1, -1))
      sample_len = sample.shape[0] - 1
      CE = f.cross_entropy(result.logits.permute(0, 2, 1)[:, :, :-1], sample[1:].view(1, -1))
      token_count+= sample_len
      cummulative_CE += CE * sample_len
    return cummulative_CE.cpu().item(), (cummulative_CE / token_count).cpu().item()


def main():
  args = parse_args()

  
  #Snag configs
  config: dict = get_merged_config(args.configs)
  
  #Validate config files.
  if "save" not in config:
    raise ValueError(".yml Configs didn't contain the 'save' key, so code can't infer where the checkpoint folder is.")

  #Validate tokenizer
  pass

  names, datasets = load_datasets(args.test_folder)

  print("Loading tokenizer.")
  tokenizer = LlamaTokenizer.from_pretrained(config["vocab_file"])
  print("Finish loading tokenizer.")
  
  #Tokenize data.
  print("Tokenizing test datasets.")
  datasets = [tokenize_dataset(dataset, tokenizer) for dataset in datasets]
  print("Finished tokenizing test datasets.")

  #Turn datasets into tensors.
  #We'll have a list [dataset] of list [sample] of tensors.
  datasets_ = []
  for dataset in datasets:
    dataset_ = []
    for sample in dataset:
      dataset_.append(torch.tensor(sample, dtype = torch.long, device = "cuda:7")) #Not really sure how to choose the cuda here.
    datasets_.append(dataset_)
  datasets = datasets_

  #Open our logging CSV file.
  log_file = open_log_file(args.log_file, names)
  log_file_writer = csv.writer(log_file)

  #Sniff for checkpoints
  cp_path = config["save"] #This should be the folder that contains all the checkpoints.
  
  while True:
    #Detect whether there's a new checkpoint.
    have_new, step_number = new_checkpoint(cp_path)
    if have_new:
      print("New checkpoint detected at step " + str(step_number) + ".")
      
      #Convert to hugginface checkpoint format.
      convert_checkpoint(cp_path + "/global_step" + str(step_number), args.tmp_path, config, args.architecture)
      
      #Load model.
      with init_empty_weights():
        if args.architecture.upper() == "NEOX":
          model = GPTNeoXForCausalLM.from_pretrained(args.tmp_path, device_map="auto")
        elif args.architecture.upper() == "LLAMA":
          model = LlamaForCausalLM.from_pretrained(args.tmp_path, device_map="auto")
        else:
          raise ValueError("Huggingface --architecture " + str(args.architecture) + " not recgonized.")
      #model = load_checkpoint_and_dispatch(model, args.tmp_path, device_map = "auto", no_split_module_classes=['Block'])
      print(model.hf_device_map)


      #Evaluate checkpoint.
      print("Performing testing.")
      results: list[float] = [] #Will contain our testing results.
      results.append(time.time())
      results.append(step_number)
      for test_index in range(len(datasets)):
        totalCE, averageCE = get_cross_entropy(model, datasets[test_index])
        results.append(totalCE)
        results.append(averageCE)
        results.append(math.e ** results[-1])
        print(names[test_index], ": Perplexity", results[-1], "; Total crossentropy", results[-3])
      print("Finished evaluating testing.")

      del model

      #Log.
      log_file_writer.writerow(results)
      log_file.flush()
        
      #Clean up.
      print("Deleting checkpoint.")
      shutil.rmtree(args.tmp_path)
      
      set_latest_tested(cp_path, step_number)

    print("Sleeping")
    time.sleep(30)


if __name__ == "__main__":
    print("I'm not dead")
    main()

