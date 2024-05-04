# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
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
Run the KTO training script with the following command with some example arguments:

python examples/scripts/kto.py \
    --model_name_or_path "gpt2" \
    --per_device_train_batch_size 8 \
    --gradient_accumulation_steps 2 \
    --learning_rate 1e-4 \
    --max_steps 1000 \
    --report_to "wandb" \
    --gradient_checkpointing True  \
    --output_dir="./test" \
    --use_peft True \
    --lora_r 64 \
    --lora_alpha 16 \
    --evaluation_strategy "steps" \
    --logging_first_step True \
    --logging_steps 10 \
    --eval_steps 500


# 1 GPU test
WANDB_PROJECT=gritkto python kto.py --model_name_or_path openaccess-ai-collective/tiny-mistral --output_dir test --report_to "wandb" --per_device_train_batch_size 1 --gradient_accumulation_steps 1
# 1 GPU test accelerate
WANDB_PROJECT=gritkto accelerate launch --config_file=config_1gpus_m7.yml kto.py --model_name_or_path openaccess-ai-collective/tiny-mistral --output_dir test --report_to "wandb" --per_device_train_batch_size 1 --gradient_accumulation_steps 1
# 8 GPUs test -> times out
WANDB_PROJECT=gritkto accelerate launch --config_file=config_8gpusfsdp_m7.yml kto.py --model_name_or_path openaccess-ai-collective/tiny-mistral --output_dir test --report_to "wandb" --per_device_train_batch_size 1 --gradient_accumulation_steps 1
# 8 GPUs test with SHARD_GRAD_OP -> increase bs (1->4)
WANDB_PROJECT=gritkto accelerate launch --config_file=config_8gpusdp_m7.yml kto.py --model_name_or_path openaccess-ai-collective/tiny-mistral --output_dir test --report_to "wandb" --per_device_train_batch_size 4 --gradient_accumulation_steps 1 --logging_steps 1
# total bs 32 -> fails with _saved_grad_shard error 
WANDB_PROJECT=gritkto accelerate launch --config_file=config_8gpusfsdp_m7.yml kto.py --model_name_or_path HuggingFaceH4/mistral-7b-sft-beta --output_dir test --report_to "wandb" --per_device_train_batch_size 1 --gradient_accumulation_steps 8 --optim rmsprop --learning_rate 5e-07 --beta 0.1

# Works but lots of nans
WANDB_PROJECT=gritkto accelerate launch --config_file=config_8gpusfsdp_m7.yml kto.py --model_name_or_path HuggingFaceH4/mistral-7b-sft-beta --output_dir test --report_to "wandb" --per_device_train_batch_size 4 --gradient_accumulation_steps 2 --optim rmsprop --learning_rate 5e-07 --beta 0.1 --logging_steps 1

# 8 GPUs test deepspeed -> Fails with https://github.com/microsoft/DeepSpeed/issues/1960
WANDB_PROJECT=gritkto accelerate launch --config_file=config_8gpusds_m7.yml kto.py --model_name_or_path openaccess-ai-collective/tiny-mistral --output_dir test --report_to "wandb" --per_device_train_batch_size 4 --gradient_accumulation_steps 1 --bf16 --logging_steps 1
# ZeRO2 -> times out -> w/ bigger bs works (1->4)
WANDB_PROJECT=gritkto accelerate launch --config_file=config_8gpusdsz2_m7.yml kto.py --model_name_or_path openaccess-ai-collective/tiny-mistral --output_dir test --report_to "wandb" --per_device_train_batch_size 4 --gradient_accumulation_steps 1 --bf16 --logging_steps 1
# Real -> Works
WANDB_PROJECT=gritkto accelerate launch --config_file=config_8gpusdsz2_m7.yml kto.py --model_name_or_path HuggingFaceH4/mistral-7b-sft-beta --output_dir test --report_to "wandb" --per_device_train_batch_size 6 --gradient_accumulation_steps 1 --optim rmsprop --learning_rate 5e-07 --beta 0.1 --logging_steps 1 --bf16

WANDB_PROJECT=gritkto accelerate launch --config_file=config_8gpusds_m7.yml kto.py --model_name_or_path HuggingFaceH4/mistral-7b-sft-beta --output_dir test --report_to "wandb" --per_device_train_batch_size 6 --gradient_accumulation_steps 1 --optim rmsprop --learning_rate 5e-07 --beta 0.1 --logging_steps 1 --bf16
CUDA_VISIBLE_DEVICES=0,1,2,3 WANDB_PROJECT=gritkto accelerate launch --config_file=config_8gpusdsgas2.yml kto.py --model_name_or_path HuggingFaceH4/mistral-7b-sft-beta --output_dir test --report_to "wandb" --per_device_train_batch_size 4 --gradient_accumulation_steps 2 --optim rmsprop --learning_rate 5e-07 --beta 0.1 --logging_steps 1 --bf16

# Works but eventually becomes nan
WANDB_PROJECT=gritkto accelerate launch --config_file=config_8gpusds_m7.yml kto.py --model_name_or_path HuggingFaceH4/mistral-7b-sft-beta --output_dir test --report_to "wandb" --per_device_train_batch_size 6 --gradient_accumulation_steps 1 --optim rmsprop --learning_rate 5e-07 --beta 0.1 --logging_steps 1 --bf16 --sanity_check False
# Try w/ peft
WANDB_PROJECT=gritkto accelerate launch --config_file=config_8gpusds_m7.yml kto.py --model_name_or_path HuggingFaceH4/mistral-7b-sft-beta --output_dir test --report_to "wandb" --per_device_train_batch_size 6 --gradient_accumulation_steps 1 --optim rmsprop --learning_rate 5e-07 --beta 0.1 --logging_steps 1 --bf16 --sanity_check False --use_peft True --lora_r 64 --lora_alpha 16

# Same as above but dpo
WANDB_PROJECT=gritkto accelerate launch --config_file=config_8gpusds_m7.yml dpo.py --model_name_or_path HuggingFaceH4/mistral-7b-sft-beta --output_dir test --report_to "wandb" --per_device_train_batch_size 6 --gradient_accumulation_steps 1 --optim rmsprop --learning_rate 5e-07 --logging_steps 1 --bf16 --sanity_check False --use_peft True --lora_r 64 --lora_alpha 16

# Fix
WANDB_PROJECT=gritkto accelerate launch --config_file=config_8gpusds_m7.yml kto.py --model_name_or_path HuggingFaceH4/mistral-7b-sft-beta --output_dir test --report_to "wandb" --per_device_train_batch_size 4 --gradient_accumulation_steps 1 --optim rmsprop --learning_rate 5e-07 --beta 0.1 --logging_steps 1 --bf16 --sanity_check False --num_train_epochs 1
WANDB_PROJECT=gritkto accelerate launch --config_file=config_8gpusds_m7.yml kto.py --model_name_or_path HuggingFaceH4/mistral-7b-sft-beta --output_dir test --report_to "wandb" --per_device_train_batch_size 6 --gradient_accumulation_steps 1 --optim rmsprop --learning_rate 5e-07 --beta 0.1 --logging_steps 1 --bf16 --sanity_check False --use_peft True --lora_r 64 --lora_alpha 16 --num_train_epochs 1
WANDB_PROJECT=gritkto accelerate launch --config_file=config_8gpusdsz2_m7.yml kto.py --model_name_or_path openaccess-ai-collective/tiny-mistral --output_dir test --report_to "wandb" --per_device_train_batch_size 2 --learning_rate 5e-07 --beta 0.1 --logging_steps 1 --bf16

WANDB_PROJECT=gritkto accelerate launch --config_file=config_8gpusds_m7.yml kto.py --model_name_or_path HuggingFaceH4/mistral-7b-sft-beta --output_dir m7-1ep-kto --report_to "wandb" --per_device_train_batch_size 4 --gradient_accumulation_steps 1 --optim rmsprop --learning_rate 5e-07 --beta 0.1 --logging_steps 1 --bf16 --sanity_check False --num_train_epochs 1
WANDB_PROJECT=gritkto accelerate launch --config_file=config_8gpusdsz2_m7.yml kto.py --model_name_or_path HuggingFaceH4/mistral-7b-sft-beta --output_dir m7-1ep-kto --report_to "wandb" --per_device_train_batch_size 4 --gradient_accumulation_steps 1 --optim rmsprop --learning_rate 5e-07 --beta 0.1 --logging_steps 1 --bf16 --sanity_check False --num_train_epochs 1

### KTO ###
WANDB_PROJECT=gritkto accelerate launch --config_file=config_8gpusdsz2_m7.yml kto.py --model_name_or_path HuggingFaceH4/mistral-7b-sft-beta --output_dir /data/niklas/m7-1ep-kto-v3 --report_to "wandb" --per_device_train_batch_size 4 --gradient_accumulation_steps 1 --optim rmsprop --learning_rate 5e-07 --beta 0.1 --logging_steps 1 --bf16 --sanity_check False --num_train_epochs 1
### DPO ###
WANDB_PROJECT=gritkto accelerate launch --config_file=config_8gpusdsz2_m7.yml dpo.py --model_name_or_path HuggingFaceH4/mistral-7b-sft-beta --output_dir /data/niklas/m7-1ep-dpo-v3 --report_to "wandb" --per_device_train_batch_size 4 --gradient_accumulation_steps 1 --optim rmsprop --learning_rate 5e-07 --logging_steps 1 --bf16 --sanity_check False --num_train_epochs 1

### GRIT KTO ###
# M7
WANDB_PROJECT=gritkto accelerate launch --config_file=config_8gpusdsz2_m7.yml kto.py --model_name_or_path GritLM/GritLM-7B --output_dir /data/niklas/GritLM-7B-KTO --report_to "wandb" --per_device_train_batch_size 4 --gradient_accumulation_steps 1 --optim rmsprop --learning_rate 5e-07 --beta 0.1 --logging_steps 1 --bf16 --sanity_check True --num_train_epochs 1
# M7 PEFT
WANDB_PROJECT=gritkto accelerate launch --config_file=config_8gpusdsz2_m7.yml kto.py --model_name_or_path GritLM/GritLM-7B --output_dir /data/niklas/GritLM-7B-KTO-LoRA --report_to "wandb" --per_device_train_batch_size 4 --gradient_accumulation_steps 1 --optim rmsprop --learning_rate 5e-07 --beta 0.1 --logging_steps 1 --bf16 --sanity_check False --use_peft True --lora_r 64 --lora_alpha 16
# M8x7
# ZeRO2
WANDB_PROJECT=gritkto accelerate launch --config_file=config_8gpusdsz2_m7.yml kto.py --model_name_or_path GritLM/GritLM-8x7B --output_dir /data/niklas/GritLM-8x7B-KTO --report_to "wandb" --per_device_train_batch_size 1 --gradient_accumulation_steps 4 --optim rmsprop --learning_rate 5e-07 --beta 0.1 --logging_steps 1 --bf16 --sanity_check True --num_train_epochs 1
# ZeRO2 - 2 machines
WANDB_PROJECT=gritkto accelerate launch --config_file=config_16gpusdsz2_m7_re.yml kto.py --model_name_or_path GritLM/GritLM-8x7B --output_dir /data/niklas/GritLM-8x7B-KTO --report_to "wandb" --per_device_train_batch_size 1 --gradient_accumulation_steps 1 --optim rmsprop --learning_rate 5e-07 --beta 0.1 --logging_steps 1 --bf16 --sanity_check True --num_train_epochs 1
WANDB_PROJECT=gritkto accelerate launch --config_file=config_16gpusdsz2_m7_re.yml --machine_rank=1 kto.py --model_name_or_path GritLM/GritLM-8x7B --output_dir /data/niklas/GritLM-8x7B-KTO --report_to "wandb" --per_device_train_batch_size 1 --gradient_accumulation_steps 1 --optim rmsprop --learning_rate 5e-07 --beta 0.1 --logging_steps 1 --bf16 --sanity_check True --num_train_epochs 1

# ZeRO3
WANDB_PROJECT=gritkto accelerate launch --config_file=config_8gpusds_m7.yml kto.py --model_name_or_path GritLM/GritLM-8x7B --output_dir /data/niklas/GritLM-8x7B-KTO --report_to "wandb" --per_device_train_batch_size 1 --gradient_accumulation_steps 1 --optim rmsprop --learning_rate 5e-07 --beta 0.1 --logging_steps 1 --bf16 --sanity_check True --num_train_epochs 1
# DPO
# Z2
WANDB_PROJECT=gritkto accelerate launch --config_file=config_8gpusdsz2_m7.yml dpo.py --model_name_or_path GritLM/GritLM-8x7B --output_dir /data/niklas/GritLM-8x7B-DPO --report_to "wandb" --per_device_train_batch_size 1 --gradient_accumulation_steps 1 --optim rmsprop --learning_rate 5e-07 --logging_steps 1 --bf16 --sanity_check True --num_train_epochs 1
# Z3
WANDB_PROJECT=gritkto accelerate launch --config_file=config_8gpusds_m7.yml dpo.py --model_name_or_path GritLM/GritLM-8x7B --output_dir /data/niklas/GritLM-8x7B-DPO --report_to "wandb" --per_device_train_batch_size 1 --gradient_accumulation_steps 1 --optim rmsprop --learning_rate 5e-07 --logging_steps 1 --bf16 --sanity_check True --num_train_epochs 1
"""
from dataclasses import dataclass, field
from typing import Optional

from datasets import Dataset, load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, HfArgumentParser

from trl import KTOConfig, KTOTrainer, ModelConfig, get_peft_config


# Define and parse arguments.
@dataclass
class ScriptArguments:
    """
    The arguments for the KTO training script.
    """

    # debugging
    sanity_check: Optional[bool] = field(default=True, metadata={"help": "only train on 1000 samples"})


def extract_anthropic_prompt(prompt_and_response):
    """Extract the anthropic prompt from a prompt and response pair."""
    search_term = "\n\nAssistant:"
    search_term_idx = prompt_and_response.rfind(search_term)
    assert search_term_idx != -1, f"Prompt and response does not contain '{search_term}'"
    return prompt_and_response[: search_term_idx + len(search_term)]


def get_hh(split: str, sanity_check: bool = False, silent: bool = False, cache_dir: str = None) -> Dataset:
    """Load the Anthropic Helpful-Harmless dataset from Hugging Face and convert it to the necessary format.

    The dataset is converted to a dictionary with the following structure:
    {
        'prompt': List[str],
        'completion': List[str],
        'label': List[bool],
    }

    Prompts should be structured as follows:
      \n\nHuman: <prompt>\n\nAssistant:
    Multiple turns are allowed, but the prompt should always start with \n\nHuman: and end with \n\nAssistant:.
    """
    dataset = load_dataset("Anthropic/hh-rlhf", split=split, cache_dir=cache_dir)
    if sanity_check:
        dataset = dataset.select(range(min(len(dataset), 1000)))

    flat_data = {
        "prompt": [],
        "completion": [],
        "label": [],
    }
    for sample in dataset:
        prompt = extract_anthropic_prompt(sample["chosen"])
        #flat_data["prompt"].append(prompt)
        flat_data["prompt"].append(f"<|user|>\n{prompt}\n<|assistant|>\n")
        flat_data["completion"].append(sample["chosen"][len(prompt) :])
        flat_data["label"].append(True)
        #flat_data["prompt"].append(prompt)
        flat_data["prompt"].append(f"<|user|>\n{prompt}\n<|assistant|>\n")
        flat_data["completion"].append(sample["rejected"][len(prompt) :])
        flat_data["label"].append(False)

    return dataset.from_dict(flat_data)


def get_ultrabin(split: str, sanity_check: bool = False, silent: bool = False, cache_dir: str = None) -> Dataset:
    dataset = load_dataset("HuggingFaceH4/ultrafeedback_binarized", split=split, cache_dir=cache_dir)
    if sanity_check:
        dataset = dataset.select(range(min(len(dataset), 1000)))
    
    flat_data = {
        "prompt": [],
        "completion": [],
        "label": [],
    }
    for sample in dataset:
        prompt = sample["prompt"]
        if len(sample["chosen"][1]["content"].strip()) > 0:
            flat_data["prompt"].append(f"<|user|>\n{prompt}\n<|assistant|>\n")
            flat_data["completion"].append(sample["chosen"][1]["content"])
            flat_data["label"].append(True)
        if len(sample["rejected"][1]["content"].strip()) > 0:
            flat_data["prompt"].append(f"<|user|>\n{prompt}\n<|assistant|>\n")
            flat_data["completion"].append(sample["rejected"][1]["content"])
            flat_data["label"].append(False)

    return dataset.from_dict(flat_data)


if __name__ == "__main__":
    parser = HfArgumentParser((ScriptArguments, KTOConfig, ModelConfig))
    script_args, kto_args, model_args = parser.parse_args_into_dataclasses()

    # 1. load a pretrained model
    model = AutoModelForCausalLM.from_pretrained(model_args.model_name_or_path, trust_remote_code=True, attn_implementation="sdpa", torch_dtype="auto")
    model_ref = AutoModelForCausalLM.from_pretrained(model_args.model_name_or_path, trust_remote_code=True, attn_implementation="sdpa", torch_dtype="auto")

    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 2. Load the Anthropic Helpful-Harmless dataset
    #train_dataset = get_hh("train", sanity_check=script_args.sanity_check)
    train_dataset = get_ultrabin("train_prefs", sanity_check=script_args.sanity_check)

    # 3. Load evaluation dataset
    #eval_dataset = get_hh("test", sanity_check=script_args.sanity_check)
    eval_dataset = get_ultrabin("test_prefs", sanity_check=script_args.sanity_check)

    # 4. initialize the KTO trainer
    trainer = KTOTrainer(
        model,
        model_ref,
        args=kto_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        peft_config=get_peft_config(model_args),
    )

    """
    # https://github.com/huggingface/trl/issues/1147#issuecomment-1896206757
    prepared_model = trainer._wrap_model(
        trainer.model, training=True, dataloader=None
    )
    if hasattr(trainer.lr_scheduler, "step"):
        prepared_model, trainer.optimizer = trainer.accelerator.prepare(
            prepared_model, trainer.optimizer
        )
    else:
        (
            prepared_model,
            trainer.optimizer,
            trainer.lr_scheduler,
        ) = trainer.accelerator.prepare(
            prepared_model, trainer.optimizer, trainer.lr_scheduler
        )
    trainer.model_wrapped = prepared_model
    if trainer.is_fsdp_enabled:
        trainer.model = prepared_model
    if trainer.ref_model is not None:
        trainer.ref_model = trainer.accelerator.prepare_model(trainer.ref_model)

    trainer.accelerator.prepare_model = lambda model, *args, **kwargs: model # Monkey-patch prepare_model a no-op , since we have manually prepared the models
    """
    # 5. train
    trainer.train()