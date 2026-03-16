import os
import argparse
from tqdm import tqdm
import torch
from PIL import Image

from torch.utils.data import Dataset, DataLoader
from transformers.utils import is_flash_attn_2_available

from accelerate import Accelerator

from transformers import LlavaNextProcessor as Processor
from transformers import LlavaNextForConditionalGeneration as Model
from transformers import BitsAndBytesConfig

from utils import Config, load_data, collect_done, write_output
from torch.nn.utils.rnn import pad_sequence
from transformers.feature_extraction_utils import BatchFeature

accelerator = Accelerator()


def collate_llava_next(data, pad_id):
    input_ids, pixel_values, image_sizes, answer, cls, noise, ids = zip(*data)
    input_ids = pad_sequence(input_ids, batch_first=True, padding_value=pad_id, padding_side='left')
    attention_mask = input_ids.ne(pad_id)
    pixel_values = torch.stack(pixel_values, dim=0)
    image_sizes = torch.stack(image_sizes, dim=0)
    answers = list(answer)
    classes = list(cls)
    noise = list(noise)
    ids = list(ids)
    len_prompt = input_ids.shape[1]

    inputs = BatchFeature({'input_ids': input_ids, 'attention_mask': attention_mask, 'pixel_values': pixel_values,
                           'image_sizes': image_sizes})
    return inputs, len_prompt, answers, classes, noise, ids


class EvalDataset(Dataset):
    def __init__(self, annotations, img_dir, processor, done):
        self.data = load_data(annotations, done)
        self.img_dir = img_dir
        self.processor = processor

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        elem = self.data[idx]
        img_path = os.path.join(self.img_dir, elem['image'])
        image = Image.open(img_path)
        question = elem['question']
        answer = elem['answer']
        img_id = elem['id']
        cls = elem['class'] if 'class' in elem.keys() else 'null'
        noise = str(elem['noise']) if 'noise' in elem.keys() else 'false'

        msgs = [
            {"role": "user", "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": question}
            ]},
        ]

        prompt = self.processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        tmp = self.processor(images=image, text=prompt, return_tensors="pt")
        input_ids = tmp['input_ids'].squeeze()
        pixel_values = tmp['pixel_values'].squeeze()
        image_sizes = tmp['image_sizes'].squeeze()

        return input_ids, pixel_values, image_sizes, answer, cls, noise, img_id


def run(args, config):
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    attn_implementation = 'flash_attention_2' if is_flash_attn_2_available() else None

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=dtype,
        bnb_4bit_use_double_quant=True
    )

    processor = Processor.from_pretrained(args.model, use_fast=True)
    pad_id = processor.tokenizer.pad_token_id
    model = Model.from_pretrained(
        args.model,
        quantization_config=bnb_config,
        device_map={"": accelerator.process_index},
        torch_dtype=dtype,
        trust_remote_code=True,
        attn_implementation=attn_implementation,
    ).eval()

    model = torch.compile(model)
    accelerator.wait_for_everyone()

    done = collect_done(args.output, config.task)
    dataset = EvalDataset(args.dataset, args.images, processor, done)
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
                                    collate_fn=lambda x: collate_llava_next(x, pad_id))

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
                    pred = processor.decode(output_ids[i, len_prompt:],
                                            skip_special_tokens=True, clean_up_tokenization_spaces=False)
                    preds.append(pred)

                write_output(preds, answers, classes, noise, ids, out)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--images', type=str, default='../input/VRSBench/Images_val')
    parser.add_argument('-d', '--dataset', type=str,
                        default='../output/annotations/VRSBench/VRSBench_EVAL_caption.jsonl')
    parser.add_argument('-m', '--model', type=str, default='llava-hf/llava-v1.6-vicuna-7b-hf')
    parser.add_argument('-b', '--batch', type=int, default=1)
    parser.add_argument('-o', '--output', type=str, default='../output/eval/vrs_llava')

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
