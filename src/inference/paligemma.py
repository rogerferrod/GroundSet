import os
import argparse
from tqdm import tqdm
import torch
import re
from PIL import Image

from torch.utils.data import Dataset, DataLoader
from transformers.utils import is_flash_attn_2_available

from utils import Config, load_data, collect_done, write_output
from utils import collate_llava

from accelerate import Accelerator

from transformers import (
    AutoProcessor,
    PaliGemmaForConditionalGeneration,
    BitsAndBytesConfig,
)

accelerator = Accelerator()


def extract_classes(txt):
    entities = re.findall(r"'(.*?)'", txt)
    qs = ' ; '.join(entities)
    return qs.strip()


class EvalDataset(Dataset):
    def __init__(self, annotations, img_dir, processor, task, done):
        self.data = load_data(annotations, done)
        self.img_dir = img_dir
        self.processor = processor
        self.task = task

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        elem = self.data[idx]
        img_path = os.path.join(self.img_dir, elem['image'])
        image = Image.open(img_path).convert("RGB").resize((448, 448))
        question = elem['question']
        answer = elem['answer']
        img_id = elem['id']
        cls = elem['class'] if 'class' in elem.keys() else 'null'
        noise = str(elem['noise']) if 'noise' in elem.keys() else 'false'

        if 'caption' in self.task:
            prompt = 'caption en '
        elif 'detect' in self.task:
            prompt = 'detect ' + cls
        elif 'phrase' in self.task:
            prompt = 'detect ' + extract_classes(question)
        elif 'segment' in self.task:
            prompt = 'segment ' + cls
        elif 'vqa' in self.task:
            prompt = 'answer en ' + question
        else:
            prompt = question

        # TODO <image> token
        tmp = self.processor(images=[image], text=prompt, return_tensors="pt")
        input_ids = tmp['input_ids'].squeeze()
        pixel_values = tmp['pixel_values'].squeeze()

        return input_ids, pixel_values, answer, cls, noise, img_id


def run(args, config):
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    attn_implementation = 'flash_attention_2' if is_flash_attn_2_available() else None

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=dtype,
        bnb_4bit_use_double_quant=True
    )

    processor = AutoProcessor.from_pretrained(args.model, use_fast=True)
    pad_id = processor.tokenizer.pad_token_id
    model = PaliGemmaForConditionalGeneration.from_pretrained(
        args.model,
        device_map={"": accelerator.process_index},
        torch_dtype=dtype,
        quantization_config=bnb_config,
        trust_remote_code=True,
        attn_implementation=attn_implementation
    ).eval()

    model = torch.compile(model)
    accelerator.wait_for_everyone()

    done = collect_done(args.output, config.task)
    dataset = EvalDataset(args.dataset, args.images, processor, config.task, done)
    ids_range = list(range(len(dataset)))
    with accelerator.split_between_processes(ids_range) as indexes:
        out_path = os.path.join(args.output, f'{config.task}_{accelerator.process_index}.csv')
        is_new = not os.path.exists(out_path)
        with open(out_path, 'a') as out:
            if is_new:
                out.write('ID\tPREDICTION\tGROUND_TRUTH\tCLASS\tNOISE\n')
                out.flush()

            local_dataset = torch.utils.data.Subset(dataset, indexes)
            dataloader = DataLoader(local_dataset, shuffle=False, batch_size=args.batch,
                                    collate_fn=lambda x: collate_llava(x, pad_id))

            for inputs, len_prompt, answers, classes, noise, ids in tqdm(dataloader,
                                                                         disable=(not accelerator.is_main_process)):
                inputs = inputs.to(model.device, dtype)
                output_ids = model.generate(
                    **inputs,
                    do_sample=config.sample,
                    temperature=config.temperature,
                    top_k=config.top_k,
                    top_p=config.top_p,
                    max_new_tokens=config.max_tok,
                    eos_token_id=processor.tokenizer.eos_token_id
                )

                preds = []
                for i in range(len(output_ids)):
                    pred = processor.decode(output_ids[i, len_prompt:], skip_special_tokens=False)
                    pred = pred.replace(processor.tokenizer.eos_token, "")
                    pred = pred.replace(processor.tokenizer.pad_token, "")

                    preds.append(pred)

                write_output(preds, answers, classes, noise, ids, out)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--images', type=str, default='../input/VRSBench/Images_val')
    parser.add_argument('-d', '--dataset', type=str,
                        default='../output/annotations/VRSBench/VRSBench_EVAL_caption.jsonl')
    parser.add_argument('-m', '--model', type=str, default='google/paligemma2-10b-mix-448')
    parser.add_argument('-b', '--batch', type=int, default=1)
    parser.add_argument('-o', '--output', type=str, default='../output/eval/vrs_paligemma')

    args = parser.parse_args()
    task = args.dataset.split('_EVAL_')[-1].split('.')[0]
    config = Config(task)
    if accelerator.is_main_process:
        print(task)
        if not os.path.exists(args.output):
            os.mkdir(args.output)

    run(args, config)


if __name__ == '__main__':
    main()
