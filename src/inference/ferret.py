import argparse
from tqdm import tqdm
from PIL import Image

import torch
import os
import re

from torch.utils.data import Dataset, DataLoader
from accelerate import Accelerator

from ferret.mm_utils import tokenizer_image_token, get_model_name_from_path
from ferret.model.builder import load_pretrained_model
from ferret.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from ferret.conversation import conv_templates
from ferret.utils import disable_torch_init

from utils import Config, load_data, collect_done, write_output
from utils import collate_llava

accelerator = Accelerator()


def normalize(txt):
    replacements = []
    for num in re.findall(r'(-?\d+(?:\.\d+)?)', txt):
        rep = int(float(num) * 1000)
        replacements.append((num, str(rep)))

    for match, rep in replacements:
        txt = txt.replace(match, rep)

    return txt


class EvalDataset(Dataset):
    def __init__(self, annotations, img_dir, tokenizer, image_processor, done):
        self.data = load_data(annotations, done)
        self.img_dir = img_dir
        self.tokenizer = tokenizer
        self.image_processor = image_processor

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        elem = self.data[idx]
        img_path = os.path.join(self.img_dir, elem['image'])
        image = Image.open(img_path)
        question = normalize(elem['question'])
        answer = elem['answer']
        img_id = elem['id']
        cls = elem['class'] if 'class' in elem.keys() else 'null'
        noise = str(elem['noise']) if 'noise' in elem.keys() else 'false'

        qs = DEFAULT_IMAGE_TOKEN + '\n' + question

        conv = conv_templates['ferret_v1'].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt')
        pixel_values = self.image_processor.preprocess([image], return_tensors='pt', do_resize=True,
                                                       do_center_crop=False, size=[336, 336])['pixel_values']

        return input_ids, pixel_values, answer, cls, noise, img_id


def run(args, config):
    dtype = torch.float16

    disable_torch_init()
    model_path = os.path.expanduser(args.model)
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, None, model_name,
                                                                           load_4bit=True,
                                                                           device_map={"": accelerator.process_index})

    pad_id = tokenizer.pad_token_id
    model = torch.compile(model.eval())
    accelerator.wait_for_everyone()

    done = collect_done(args.output, config.task)
    dataset = EvalDataset(args.dataset, args.images, tokenizer, image_processor, done)
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
                input_ids = inputs['input_ids'].to(model.device)
                image_tensor = inputs['pixel_values'].to(model.device, dtype)
                with torch.inference_mode():
                    output_ids = model.generate(input_ids, images=image_tensor,
                                                do_sample=config.sample,
                                                temperature=config.temperature,
                                                top_k=config.top_k,
                                                top_p=config.top_p,
                                                max_new_tokens=config.max_tok,
                                                eos_token_id=tokenizer.eos_token_id)

                preds = []
                for i in range(len(output_ids)):
                    pred = tokenizer.decode(output_ids[i, len_prompt:],
                                            skip_special_tokens=True, clean_up_tokenization_spaces=False)
                    preds.append(pred)

                write_output(preds, answers, classes, noise, ids, out)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--images', type=str, default='../input/VRSBench/Images_val')
    parser.add_argument('-d', '--dataset', type=str,
                        default='../output/annotations/VRSBench/VRSBench_EVAL_caption.jsonl')
    parser.add_argument('-m', '--model', type=str, default='../../input/models/ferret-7b-v1-3')
    parser.add_argument('-b', '--batch', type=int, default=1)
    parser.add_argument('-o', '--output', type=str, default='../output/eval/vrs_ferret')

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
