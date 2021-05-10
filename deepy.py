#!/usr/bin/env python
# Copyright (c) 2021, EleutherAI contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import os

import deepspeed
import requests
from deepspeed.launcher.runner import main


logging.basicConfig(level=os.environ.get("LOGLEVEL", "INFO"))

from megatron.neox_arguments import NeoXArgs
from megatron.utils import get_wandb_api_key



neox_args = NeoXArgs.consume_deepy_args()
if neox_args.wandb_group is not None:
    # concat the wandb group name with a uid to make sure it's unique
    import wandb
    neox_args.wandb_group += "_" + wandb.util.generate_id()
neox_args.print()
deepspeed_main_args = neox_args.get_deepspeed_main_args()

# Extract wandb API key and inject into worker environments
wandb_token = get_wandb_api_key(neox_args=neox_args)
if wandb_token is not None:
    deepspeed.launcher.runner.EXPORT_ENVS.append('WANDB_API_KEY')
    os.environ['WANDB_API_KEY'] = wandb_token

if __name__ == '__main__':
    main(deepspeed_main_args)
