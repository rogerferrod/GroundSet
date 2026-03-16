import os
import argparse
from tqdm import tqdm
import re
from PIL import Image

from torch.utils.data import Dataset, DataLoader

from minigpt4.common.eval_utils import prepare_texts
from minigpt4.conversation.conversation import CONV_VISION_minigptv2
from minigpt4.common.config import Config as ConfigNet

from minigpt4.common.registry import registry

from utils import Config, load_data, collect_done, write_output


def normalize(txt):
    replacements = []
    for num in re.findall(r'(-?\d+(?:\.\d+)?)', txt):
        rep = int(float(num) * 100)
        replacements.append((num, str(rep)))

    for match, rep in replacements:
        txt = txt.replace(match, rep)

    return txt


class EvalDataset(Dataset):
    def __init__(self, annotations, img_dir, image_processor, task, done):
        self.data = load_data(annotations, done)
        self.img_dir = img_dir
        self.image_processor = image_processor
        self.task = task

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        elem = self.data[idx]
        img_path = os.path.join(self.img_dir, elem['image'])
        image = Image.open(img_path).convert('RGB')
        question = elem['question']
        if self.task == 'phrase':
            match = re.search(r"\[\s*(?:'[^']*'(?:\s*,\s*'[^']*')*)?\s*\]", question)
            if match:
                question = match.group(0).replace('[', '').replace(']', '')
        else:
            question = normalize(question)

        answer = elem['answer']
        img_id = elem['id']
        cls = elem['class'] if 'class' in elem.keys() else 'null'
        noise = str(elem['noise']) if 'noise' in elem.keys() else 'false'

        if 'classification' in self.task:
            qs = '[identify] ' + question
        elif self.task in {'rec', 'detection'}:
            qs = '[refer] ' + question
        elif self.task == 'phrase':
            qs = '[detection] ' + question
        elif self.task == 'caption':
            qs = '[caption] ' + question
        elif self.task == 'vqa':
            qs = '[vqa] ' + question
        else:
            qs = question

        image = self.image_processor(image)
        return qs, image, answer, cls, noise, img_id


def run(args, config):
    cfg = ConfigNet(args)

    model_config = cfg.model_cfg
    model_cls = registry.get_model_class(model_config.arch)
    model = model_cls.from_config(model_config).to('cuda:0')

    key = list(cfg.datasets_cfg.keys())[0]
    vis_processor_cfg = cfg.datasets_cfg.get(key).vis_processor.train
    vis_processor = registry.get_processor_class(vis_processor_cfg.name).from_config(vis_processor_cfg)
    model.eval()

    conv = CONV_VISION_minigptv2.copy()

    done = collect_done(args.output, config.task)
    dataset = EvalDataset(args.dataset, args.images, vis_processor, config.task, done)

    out_path = os.path.join(args.output, f'{config.task}_0.csv')
    is_new = not os.path.exists(out_path)
    with open(out_path, 'a') as out:
        if is_new:
            out.write('ID\tPREDICTION\tGROUND_TRUTH\tCLASS\tNOISE\n')
            out.flush()

        dataloader = DataLoader(dataset, shuffle=False, batch_size=args.batch)

        for texts, images, answers, classes, noise, ids in tqdm(dataloader):
            texts = prepare_texts(texts, conv)
            preds = model.generate(images, texts,
                                   max_new_tokens=config.max_tok,
                                   do_sample=False)

            write_output(preds, answers, classes, noise, ids, out)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--images', type=str, default='../input/VRSBench/Images_val')
    parser.add_argument('-d', '--dataset', type=str,
                        default='../output/annotations/VRSBench/VRSBench_EVAL_caption.jsonl')
    parser.add_argument('-b', '--batch', type=int, default=1)
    parser.add_argument('--cfg-path', type=str, default='./minigpt4/eval_configs/minigptv2_eval.yaml')
    parser.add_argument('--options', nargs="+", help="override some settings in the used config, the key-value pair "
                                                     "in xxx=yyy format will be merged into config file (deprecate), "
                                                     "change to --cfg-options instead.")
    parser.add_argument('-o', '--output', type=str, default='../output/eval/vrs_minigpt')

    args = parser.parse_args()
    task = args.dataset.split('_EVAL_')[-1].split('.')[0]
    config = Config(task)
    print(task)
    if not os.path.exists(args.output):
        os.mkdir(args.output)

    run(args, config)


if __name__ == '__main__':
    main()
