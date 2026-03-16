import os
import sys
import json
import pathlib
from PIL import Image
from typing import Optional
from dataclasses import dataclass, field

import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence

from transformers import (
    Trainer,
    TrainingArguments,
    LlavaNextProcessor,
    LlavaNextForConditionalGeneration,
    BitsAndBytesConfig,
    HfArgumentParser
)

from peft import (
    LoraConfig,
    prepare_model_for_kbit_training,
    get_peft_model
)


@dataclass
class ModelArguments:
    model_id: str = field(default=None)
    max_length: int = field(default=4096)
    use_lora: bool = field(default=True)
    use_qlora: bool = field(default=False)
    lora_r: int = field(default=32)
    lora_alpha: int = field(default=64)
    lora_dropout: float = field(default=0.1)


@dataclass
class DataArguments:
    dataset_path: str = field(default='../input/annotations/GroundSet_TRAIN_it.jsonl')
    images_path: str = field(default='../input/images')


@dataclass
class ProjectTrainingArguments(TrainingArguments):
    output_dir: str = field(default='../output/checkpoints/llava-next-vicuna-lora')
    deepspeed: Optional[str] = field(default=None)
    num_train_epochs: float = field(default=1)
    per_device_train_batch_size: int = field(default=4)
    gradient_accumulation_steps: int = field(default=1)
    save_strategy: str = field(default="steps")
    save_steps: int = field(default=200)
    logging_steps: int = field(default=10)
    learning_rate: float = field(default=2e-4)
    vision_learning_rate: float = field(default=2e-5)
    weight_decay: float = field(default=0.0)
    warmup_ratio: float = field(default=0.03)
    lr_scheduler_type: str = field(default="cosine")
    tf32: bool = field(default=True)
    bf16: bool = field(default=True)
    gradient_checkpointing: bool = field(default=False)
    dataloader_num_workers: int = field(default=8)
    report_to: str = field(default="wandb")
    remove_unused_columns: bool = field(default=False)
    unfreeze_vision_tower: bool = field(default=False)
    freeze_language_model: bool = field(default=False)
    low_res: bool = field(default=False)


def format_bytes(size):
    billion = 10 ** 9
    million = 10 ** 6

    if size >= billion:
        return f"{size / billion:.2f}B"
    elif size >= million:
        return f"{size / million:.2f}M"
    else:
        return f"{size} bytes"


def find_all_linear_names(model, freeze_language_model=False):
    cls = torch.nn.Linear
    lora_module_names = set()
    multimodal_keywords = ['multi_modal_projector', 'vision_tower']
    for name, module in model.named_modules():
        if any(mm_keyword in name for mm_keyword in multimodal_keywords):
            continue

        if freeze_language_model and 'language_model' in name:
            continue

        if isinstance(module, cls):
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])

    if 'lm_head' in lora_module_names:
        lora_module_names.remove('lm_head')
    return list(lora_module_names)


def print_rank0(*args):
    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if local_rank in [0, -1]:
        print(*args)


def param_breakdown(model):
    param_counts = {}
    for name, param in model.named_parameters():
        if param.requires_grad:
            if "multi_modal_projector" in name:
                group = "Multi-Modal Projector"
            elif "vision_tower" in name:
                group = "Vision Tower"
            elif "embed_tokens" in name:
                group = "Embeddings"
            elif "lm_head" in name:
                group = "LM Head"
            else:
                group = "LLM Layers"

            if group not in param_counts:
                param_counts[group] = 0
            param_counts[group] += param.numel()

    print_rank0("\n>> Trainable Parameter Breakdown:")
    for group, count in param_counts.items():
        print_rank0(f"   - {group}: {format_bytes(count)}")

    print_rank0("=========================================================\n")


class LlavaDataset(Dataset):
    def __init__(self, dataset_name_or_path, img_dir, processor, max_length, low_res=False):
        super().__init__()
        self.img_dir = img_dir
        self.processor = processor
        self.max_length = max_length
        self.low_res = low_res

        self.dataset = []
        if 'jsonl' in dataset_name_or_path:
            with open(dataset_name_or_path, 'r') as in_file:
                for line in in_file:
                    self.dataset.append(json.loads(line))
        else:
            with open(dataset_name_or_path, 'r') as json_file:
                self.dataset = json.load(json_file)

        self.dataset_length = len(self.dataset)

    def __len__(self) -> int:
        return self.dataset_length

    def __getitem__(self, idx: int):
        sample = self.dataset[idx]
        image_name = sample.get("image", None)
        question = sample['human']
        answer = sample['gpt']

        is_missing = False
        is_text_only = (image_name is None or str(image_name).lower() == "null" or image_name == "")

        if not is_text_only:
            img_path = os.path.join(self.img_dir, image_name)
            if not os.path.exists(img_path):
                print(f"WARNING: Image missing at {img_path}.")
                is_missing = True

        if is_text_only or is_missing:
            image = Image.new('RGB', (672, 672), color=(0, 0, 0))
        else:
            try:
                image = Image.open(img_path).convert('RGB')
            except Exception as e:
                print(f"WARNING: Could not open image {img_path}: {e}.")
                image = Image.new('RGB', (672, 672), color=(0, 0, 0))

        if self.low_res:
            image = image.resize((336, 336))

        prompt = (
            "A chat between a curious human and an artificial intelligence assistant. "
            "The assistant gives helpful, detailed, and polite answers to the human's questions. "
            f"USER: <image>\n{question} ASSISTANT:"
        )

        full_prompt = f"{prompt} {answer}{self.processor.tokenizer.eos_token}"
        inputs = self.processor(images=image, text=full_prompt, return_tensors="pt",
                                truncation=True, max_length=self.max_length)

        input_ids = inputs["input_ids"].squeeze(0)
        pixel_values = inputs["pixel_values"].squeeze(0)
        image_sizes = inputs["image_sizes"].squeeze(0)

        patches = 3 if self.low_res else 5
        assert inputs["pixel_values"].shape[1] == patches, \
            f"Unexpected patch count: {inputs['pixel_values'].shape}"

        prompt_inputs = self.processor(images=image, text=prompt, return_tensors="pt",
                                       truncation=True, max_length=self.max_length)

        prompt_len = prompt_inputs["input_ids"].shape[1]
        if prompt_inputs["input_ids"][0, -1] == self.processor.tokenizer.eos_token_id:
            prompt_len -= 1

        start_index = min(prompt_len, input_ids.shape[0])
        if input_ids.shape[0] > self.max_length:
            input_ids = input_ids[:self.max_length]

        return input_ids, pixel_values, image_sizes, start_index


class LlavaDataCollator:
    def __init__(self, pad_token_id):
        self.pad_token_id = pad_token_id

    def __call__(self, examples):
        examples = [ex for ex in examples if ex is not None]

        if len(examples) == 0:
            raise ValueError("All examples in the batch were missing images or corrupted!")

        input_ids_list = [example[0] for example in examples]
        pixel_values_list = [example[1] for example in examples]
        image_sizes_list = [example[2] for example in examples]
        start_index_list = [example[3] for example in examples]

        input_ids = pad_sequence(input_ids_list, batch_first=True, padding_value=self.pad_token_id)
        attention_mask = input_ids.ne(self.pad_token_id)
        pixel_values = torch.stack(pixel_values_list, dim=0)
        image_sizes = torch.stack(image_sizes_list, dim=0)

        labels = input_ids.clone()
        labels[labels == self.pad_token_id] = -100

        for i, start_idx in enumerate(start_index_list):
            labels[i, :start_idx] = -100

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "pixel_values": pixel_values,
            "image_sizes": image_sizes,
            "labels": labels
        }


class LayerwiseTrainer(Trainer):
    def create_optimizer(self):
        opt_model = self.model
        if self.optimizer is None:
            decay_parameters = []
            decay_parameters_name = []
            no_decay_parameters = []
            no_decay_parameters_name = []
            vision_parameters = []
            vision_parameters_name = []

            for name, param in opt_model.named_parameters():
                if not param.requires_grad:
                    continue

                if "vision_tower" in name:
                    vision_parameters.append(param)
                    vision_parameters_name.append(name)
                    continue

                if name.endswith(".bias") or "norm" in name:
                    no_decay_parameters.append(param)
                    no_decay_parameters_name.append(name)
                else:
                    decay_parameters.append(param)
                    decay_parameters_name.append(name)

            optimizer_grouped_parameters = [
                {
                    "params": decay_parameters,
                    "weight_decay": self.args.weight_decay,
                    "lr": self.args.learning_rate,
                },
                {
                    "params": no_decay_parameters,
                    "weight_decay": 0.0,
                    "lr": self.args.learning_rate,
                },
                {
                    "params": vision_parameters,
                    "weight_decay": self.args.weight_decay,
                    "lr": self.args.vision_learning_rate,
                },
            ]

            optimizer_cls, optimizer_kwargs = Trainer.get_optimizer_cls_and_kwargs(self.args)
            self.optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)

        return self.optimizer


def run():
    # Arguments
    parser = HfArgumentParser((ModelArguments, DataArguments, ProjectTrainingArguments))

    if len(sys.argv) == 1:
        print_rank0("No arguments passed, using hardcoded defaults for demonstration.")
        model_args = ModelArguments()
        data_args = DataArguments()
        training_args = ProjectTrainingArguments()
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Processor
    print_rank0(f"Loading processor for {model_args.model_id}...")
    processor = LlavaNextProcessor.from_pretrained(model_args.model_id)
    processor.tokenizer.padding_side = "right"
    pad_id = processor.tokenizer.pad_token_id

    # Dataset
    print_rank0("Loading Dataset...")
    train_dataset = LlavaDataset(data_args.dataset_path, data_args.images_path, processor,
                                 model_args.max_length, training_args.low_res)

    # Model
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    device_map = {"": local_rank}
    if training_args.deepspeed:
        try:
            with open(training_args.deepspeed, 'r') as zero_config:
                deepspeed_config = json.load(zero_config)
                if deepspeed_config.get("zero_optimization", {}).get("stage") == 3:
                    device_map = None
                    print_rank0(">> Detected DeepSpeed ZeRO-3: Setting device_map to None")
                    if model_args.use_qlora:
                        print_rank0("WARNING: QLoRA (4-bit) + ZeRO-3 is not stable.")
        except Exception as e:
            print_rank0(f"Error reading DeepSpeed config: {e}")

    bnb_config = None
    if model_args.use_qlora:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )

    print_rank0(f"Loading Model {model_args.model_id} on local rank {local_rank}...")
    model = LlavaNextForConditionalGeneration.from_pretrained(
        model_args.model_id,
        torch_dtype=torch.bfloat16,
        quantization_config=bnb_config,
        device_map=device_map,
        attn_implementation="flash_attention_2"
    )

    # Model settings
    model.config.use_cache = False
    model.config.text_config.use_cache = False

    if training_args.freeze_language_model:
        print_rank0(">> Freezing Language Model...")
        for param in model.language_model.parameters():
            param.requires_grad = False

    if training_args.unfreeze_vision_tower:
        print_rank0(">> Unfreezing Vision Tower...")
        for param in model.vision_tower.parameters():
            param.requires_grad = True
    else:
        for param in model.vision_tower.parameters():
            param.requires_grad = False

    print_rank0(">> Unfreezing Multi-Modal Projector...")
    for name, param in model.named_parameters():
        if "multi_modal_projector" in name:
            param.requires_grad = True

    # Gradient Checkpointing
    if training_args.gradient_checkpointing:
        print_rank0(">> Enabling Gradient Checkpointing...")
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()
        model.language_model.enable_input_require_grads()

        if hasattr(model, 'vision_tower'):
            model.vision_tower.enable_input_require_grads()

        if hasattr(model, 'get_input_embeddings'):
            model.get_input_embeddings().requires_grad_(True)

    # LoRA
    if model_args.use_lora or model_args.use_qlora:
        target_modules = find_all_linear_names(model, freeze_language_model=training_args.freeze_language_model)

        modules_to_save = ["multi_modal_projector"]
        if training_args.unfreeze_vision_tower:
            modules_to_save.append("vision_tower")

        if len(target_modules) > 0:
            lora_config = LoraConfig(
                r=model_args.lora_r,
                lora_alpha=model_args.lora_alpha,
                lora_dropout=model_args.lora_dropout,
                target_modules=target_modules,
                init_lora_weights="gaussian",
                modules_to_save=modules_to_save
            )

            if model_args.use_qlora:
                model = prepare_model_for_kbit_training(model,
                                                        use_gradient_checkpointing=training_args.gradient_checkpointing)

            model = get_peft_model(model, lora_config)
            print_rank0(f">> PEFT enabled. Target modules: {target_modules}")
        else:
            print_rank0(">> No linear modules targeted for LoRA")

        if training_args.gradient_checkpointing:
            model.enable_input_require_grads()

    # Float32 stability
    for name, param in model.named_parameters():
        if "norm" in name:
            param.data = param.data.to(torch.float32)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print_rank0(f"Total parameters: {format_bytes(total_params)}")
    print_rank0(f"Trainable parameters: {format_bytes(trainable_params)}")
    param_breakdown(model)

    # Trainer
    trainer = LayerwiseTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=LlavaDataCollator(pad_id),
    )

    print_rank0("Starting Training...")
    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()

    # Saving
    print_rank0("Saving Model...")
    trainer.save_state()
    trainer.save_model(training_args.output_dir)

    if training_args.local_rank in [0, -1]:
        processor.save_pretrained(training_args.output_dir)
        print_rank0(f"Done! Model and tokenizer saved to {training_args.output_dir}")


if __name__ == '__main__':
    run()
