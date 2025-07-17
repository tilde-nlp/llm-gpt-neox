import os
import json
import re


def crop_haystack(hs, spm, max_tokens = 8192):
  result = []
  total_tokens = 0
  for sentence in hs:
    if total_tokens + len(sentence) > max_tokens:
      if len(result) != 0:
        result[-1]+= spm.encode("...")
      else:
        result+= spm.encode("...")
      break
    else:
      result.append(sentence)
      total_tokens+= len(sentence)
  return result

def load_hay(file_path, spm, max_tokens = 8192):
  """
  :return: list(file)[list(sentence)[list(token_index)[int]]]
  """
  pattern = re.escape("<<|needle|>>")
  haystacks_raw = []
  with open(file_path, "r") as f:
    for line in f:
      haystacks_raw.append(json.loads(line)["text"])

  haystacks = []
  for haystack in haystacks_raw:
    haystack = [spm.encode(chunk) for chunk in re.split(pattern, haystack) if chunk]
    haystack = crop_haystack(haystack, spm, max_tokens)
    haystacks.append(haystack)
  
  return haystacks

  
def load_needles(path, spm):
  """
  Let's just assume a simple json of list of dict["needle_raw", "answer_prompt_raw", "answer_raw"]
  """
  with open(path, "r") as f:
    needles = json.loads(f.read())
  for needle in needles:
    needle["needle"] = spm.encode(needle["needle_raw"])
    needle["answer_prompt"] = spm.encode(needle["answer_prompt_raw"])
    needle["answer"] = spm.encode(needle["answer_raw"])
    
  return needles

def get_sample(hs, needle, position = 0.5):
  """
  :param position: Token percentage of prompt length where code will try to insert the needle. [0.0 .. 1.0]
                   Will overshoot rather than undershoot.
  :return: Returns list of spm tokens for prompt as well as a dict of useful info.
  """
  meta = {}
  result = []
  
  meta["correct_answer"] = needle["answer_raw"]

  needle_length = len(needle["needle"]) + len(needle["answer_prompt"])
  prompt_length = sum([len(sentence) for sentence in hs]) + needle_length
  meta["prompt_length"] = prompt_length
  meta["total_length"] = len(needle["answer"])

  target_token = prompt_length * position

  current_length = 0
  # Hay up to needle
  for idx, sentence in enumerate(hs):
    current_length+= len(sentence)
    result+= sentence
    if current_length >= target_token:
      break
  
  # Needle
  meta["needle_position"] = current_length

  # Hay after needle
  result+= needle["needle"]
  for idx in range(idx + 1, len(hs)):
    result+= hs[idx]
  
  # Answer prefix
  result+= needle["answer_prompt"]
  
  return result, meta
