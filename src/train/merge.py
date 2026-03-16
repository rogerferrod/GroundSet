import os
import argparse
import torch
from transformers import LlavaNextForConditionalGeneration, LlavaNextProcessor
from peft import PeftModel


def merge(args):
    print("Loading base model...")
    base_model = LlavaNextForConditionalGeneration.from_pretrained(
        args.base,
        torch_dtype=torch.float16,
        device_map="cpu",
        low_cpu_mem_usage=True
    )

    print("Loading LoRA adapter...")
    model = PeftModel.from_pretrained(base_model, args.input)

    print("Merging weights...")
    model = model.merge_and_unload()

    print(f"Saving merged model to {args.output}...")
    model.save_pretrained(args.output)
    processor = LlavaNextProcessor.from_pretrained(args.base)
    processor.save_pretrained(args.output)
    print("Done!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', type=str, default='../output/checkpoints/llava-next-vicuna-lora')
    parser.add_argument('-b', '--base', type=str, default='llava-hf/llava-v1.6-vicuna-7b-hf')
    parser.add_argument('-o', '--output', type=str, default='../output/checkpoints/llava-next-vicuna-lora-merged')

    args = parser.parse_args()
    os.makedirs(args.output, exist_ok=True)
    merge(args)
