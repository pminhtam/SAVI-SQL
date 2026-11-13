# Copyright 2020-2025 The HuggingFace Team. All rights reserved.
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

"""
Supervised fine-tuning script for decoder language models.

Usage:

# One 1 node of 8 x H100s
accelerate launch --config_file recipes/accelerate_configs/zero3.yaml scripts/sft.py \
    --model_name_or_path Qwen/Qwen2.5-1.5B-Instruct \
    --dataset_name trl-lib/Capybara \
    --learning_rate 2.0e-5 \
    --num_train_epochs 1 \
    --packing \
    --max_seq_length 4096 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 8 \
    --gradient_checkpointing \
    --bf16 true \
    --logging_steps 5 \
    --eval_strategy steps \
    --eval_steps 100 \
    --output_dir data/Qwen2.5-1.5B-SFT
"""
import json
import logging
import os
import sys
import shutil

import datasets
import transformers
from transformers import set_seed
from transformers.trainer_utils import get_last_checkpoint

from alignment import ScriptArguments, SFTConfig, get_dataset, get_model, get_tokenizer
from trl import ModelConfig, TrlParser, get_peft_config, setup_chat_format
from trl import SFTTrainer
# from alignment.trainer.sft_trainer import SFTTrainer_new as SFTTrainer
from datasets import disable_caching
# disable_caching()

logger = logging.getLogger(__name__)


def main(script_args, training_args, model_args):
    # Set seed for reproducibility
    set_seed(training_args.seed)

    ###############
    # Setup logging
    ###############
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    logger.info(f"Model parameters {model_args}")
    logger.info(f"Script parameters {script_args}")
    logger.info(f"Training parameters {training_args}")

    # Check for last checkpoint
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir):
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
    if last_checkpoint is not None and training_args.resume_from_checkpoint is None:
        logger.info(f"Checkpoint detected, resuming training at {last_checkpoint=}.")

    ################
    # Load datasets
    ################
    dataset = get_dataset(script_args)
    ################
    # Load tokenizer
    ################
    tokenizer = get_tokenizer(model_args, training_args)
    ############
    # Load model
    ############
    logger.info("*** Loading model ***")
    model = get_model(model_args, training_args)

    if tokenizer.chat_template is None:
        logger.info("No chat template provided, using ChatML.")
        model, tokenizer = setup_chat_format(model, tokenizer, format="chatml")

    ############################
    # Create branch on the hub if needed
    ############################
    # 1. Define your repo and target branch
    hub_model_id = training_args.hub_model_id if training_args.push_to_hub else None
    hub_revision = training_args.hub_revision if training_args.hub_revision else "main"

    if hub_model_id is not None and hub_revision != "main":
        from huggingface_hub import HfApi
        # 2. Programmatically create the branch if it doesn't exist
        api = HfApi()

        try:
            # Ensure the repository itself exists first
            api.create_repo(repo_id=hub_model_id, exist_ok=True)

            # Create the branch (if it already exists, this will raise an HTTPError)
            api.create_branch(repo_id=hub_model_id, branch=hub_revision, exist_ok=True)
            print(f"Branch '{hub_revision}' is ready.")
        except Exception as e:
            print(f"Branch status: {e}")

    ############################
    # Initialize the SFT Trainer
    ############################
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset[script_args.dataset_train_split],
        eval_dataset=(dataset[script_args.dataset_test_split] if training_args.eval_strategy != "no" else None),
        processing_class=tokenizer,
        peft_config=get_peft_config(model_args),
    )

    ###############
    # Training loop
    ###############
    logger.info("*** Train ***")
    checkpoint = None
    if training_args.resume_from_checkpoint is not None:
        checkpoint = training_args.resume_from_checkpoint
    elif last_checkpoint is not None:
        checkpoint = last_checkpoint
    train_result = trainer.train(resume_from_checkpoint=checkpoint)
    metrics = train_result.metrics
    metrics["train_samples"] = len(dataset[script_args.dataset_train_split])
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()

    ##################################
    # Save model and create model card
    ##################################
    logger.info("*** Save model ***")
    # Align the model's generation config with the tokenizer's eos token
    # to avoid unbounded generation in the transformers `pipeline()` function
    trainer.model.generation_config.eos_token_id = tokenizer.eos_token_id
    trainer.model.config.eos_token_id = tokenizer.eos_token_id
    trainer.save_model(training_args.output_dir)
    logger.info(f"Model saved to {training_args.output_dir}")

    # Save everything else on main process
    kwargs = {
        "model_name": training_args.hub_model_id if training_args.push_to_hub else None,
        "dataset_name": script_args.dataset_name,
        "tags": ["alignment-handbook"],
    }
    if trainer.accelerator.is_main_process:
        trainer.create_model_card(**kwargs)
        # Restore k,v cache for fast inference
        trainer.model.config.use_cache = True
        trainer.model.config.save_pretrained(training_args.output_dir)

    ##########
    # Evaluate
    ##########
    if training_args.do_eval:
        logger.info("*** Evaluate ***")
        metrics = trainer.evaluate()
        metrics["eval_samples"] = len(dataset[script_args.dataset_test_split])
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    #############
    # push to hub
    #############
    if training_args.push_to_hub:
        logger.info("Pushing to hub...")
        # push to specific branch https://github.com/huggingface/transformers/blob/02d8fb9784e8f14a1251e4c992cd82a5762417c6/src/transformers/trainer.py#L4253
        target_branch = training_args.hub_revision if training_args.hub_revision else "main"
        trainer.push_to_hub(revision=target_branch, **kwargs)

class MyEncoder(json.JSONEncoder):
    """
    Add 15/05/2026
    """
    def default(self, obj):
        if hasattr(obj, '__dict__'):
            return obj.__dict__
        # Handle other non-serializable types like sets if needed
        if isinstance(obj, set):
            return list(obj)
        return super().default(obj)

def copy_config_to_output_dir(script_args, training_args, model_args):
    """
    Copy the script arguments, training arguments, and model arguments to the output directory.
    https://github.com/huggingface/trl/blob/v0.20.0/trl/scripts/utils.py
    """

    os.makedirs(training_args.output_dir, exist_ok=True)
    args = sys.argv[1:]
    if "--config" in args:
        # Get the config file path from
        config_index = args.index("--config")
        args.pop(config_index)  # remove the --config flag
        config_path = args.pop(config_index)  # get the path to the config file
        # Copy the config file to the output directory
        try:
            shutil.copy(config_path, os.path.join(training_args.output_dir, "config.yaml"))
        except shutil.SameFileError:
            print("Source and destination are the same file.")
    # import pdb; pdb.set_trace()
    with open(os.path.join(training_args.output_dir, "script_args.json"), "w") as f:
        # json.dump(script_args.__dict__, f, indent=4)
        json_data = json.dumps(script_args, cls=MyEncoder)
        json.dump(json_data, f, indent=4)
    with open(os.path.join(training_args.output_dir, "training_args.json"), "w") as f:
        # json.dump(training_args.__dict__, f, indent=4)
        f.write(str(training_args.__dict__))  # Use str() to avoid serialization issues with TrainingArguments
    with open(os.path.join(training_args.output_dir, "model_args.json"), "w") as f:
        json.dump(model_args.__dict__, f, indent=4)

if __name__ == "__main__":
    parser = TrlParser((ScriptArguments, SFTConfig, ModelConfig))
    # import pdb; pdb.set_trace()
    script_args, training_args, model_args = parser.parse_args_and_config()
    copy_config_to_output_dir(script_args, training_args, model_args)
    main(script_args, training_args, model_args)
