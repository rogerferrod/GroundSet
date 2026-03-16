import os
import argparse
from tqdm import tqdm
import torch
import re
from PIL import Image

from torch.utils.data import Dataset, DataLoader

from accelerate import Accelerator

from geochat.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from geochat.conversation import conv_templates, SeparatorStyle, Conversation
from geochat.model.builder import load_pretrained_model
from geochat.utils import disable_torch_init
from geochat.mm_utils import tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria

from utils import Config, load_data, collect_done, write_output
from utils import collate_llava

accelerator = Accelerator()


def normalize(txt):
    replacements = []
    for num in re.findall(r'(-?\d+(?:\.\d+)?)', txt):
        rep = int(float(num) * 100)
        replacements.append((num, str(rep)))

    for match, rep in replacements:
        txt = txt.replace(match, rep)

    return txt


class EvalDataset(Dataset):
    def __init__(self, annotations, img_dir, tokenizer, image_processor, task, done):
        self.data = load_data(annotations, done)
        self.img_dir = img_dir
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.task = task

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        elem = self.data[idx]
        img_path = os.path.join(self.img_dir, elem['image'])
        image = Image.open(img_path)

        question = elem['question']
        question = normalize(question)
        search = re.findall(r'(\[\s*-?\d+(?:\.\d+)?(?:\s*,\s*-?\d+(?:\.\d+)?){3}\s*\])', question)
        replacements = []
        for match in search:
            nums = re.findall(r'(-?\d+(?:\.\d+)?)', match)
            rbox = '{' + f'<{nums[0]}><{nums[1]}><{nums[2]}><{nums[3]}>|<90>' + '}'
            replacements.append((match, rbox))

        for match, rep in replacements:
            question = question.replace(match, rep)

        answer = elem['answer']
        img_id = elem['id']
        cls = elem['class'] if 'class' in elem.keys() else 'null'
        noise = str(elem['noise']) if 'noise' in elem.keys() else 'false'

        if 'classification' in self.task:
            qs = '[identify] ' + question
        elif self.task in {'rec', 'phrase', 'detection'}:
            qs = '[refer] ' + question
        else:
            qs = question

        qs = DEFAULT_IMAGE_TOKEN + '\n' + qs

        conv = conv_templates['llava_v1'].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt')
        pixel_values = self.image_processor.preprocess([image],
                                                       crop_size={'height': 504, 'width': 504},
                                                       size={'shortest_edge': 504},
                                                       return_tensors='pt')['pixel_values']

        return input_ids, pixel_values, answer, cls, noise, img_id


def run(args, config):
    dtype = torch.float16

    disable_torch_init()
    model_name = get_model_name_from_path(args.model)
    tokenizer, model, image_processor, context_len = load_pretrained_model(args.model,
                                                                           None,
                                                                           model_name,
                                                                           load_4bit=True,
                                                                           device_map={"": accelerator.process_index})
    pad_id = tokenizer.pad_token_id
    model = torch.compile(model.eval())
    accelerator.wait_for_everyone()

    done = collect_done(args.output, config.task)
    dataset = EvalDataset(args.dataset, args.images, tokenizer, image_processor, config.task, done)
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
    parser.add_argument('-m', '--model', type=str, default='MBZUAI/geochat-7B')
    parser.add_argument('-b', '--batch', type=int, default=1)
    parser.add_argument('-o', '--output', type=str, default='../output/eval/vrs_geochat')

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
