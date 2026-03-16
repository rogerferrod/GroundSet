import os
import torch
import json
import glob
import re
import pandas as pd
from torch.nn.utils.rnn import pad_sequence
from transformers.feature_extraction_utils import BatchFeature


class Config:
    def __init__(self, task):
        self.task = task
        self.sample = False
        self.temperature = 1
        self.top_k = 50
        self.top_p = 1
        self.max_tok = 20

        if task == 'caption':
            self.sample = True
            self.temperature = 0.2
            self.top_k = 64
            self.top_p = 0.95
            self.max_tok = 400
        if task in ['phrase', 'detection', 'rec']:
            self.max_tok = 1024
        if task == 'segment':
            self.max_tok = 2048


def load_data(file, done):
    data = []
    with open(file, 'r') as json_file:
        for line in json_file:
            elem = json.loads(line)
            if elem['id'] not in done:
                data.append(elem)

    return data


def collect_done(path, task):
    done = set()
    files = [x for x in glob.glob(os.path.join(path, '*.csv')) if re.search(task + r'_[\d]+\.csv$', x)]
    for file in files:
        df = pd.read_csv(file, sep='\t', header=0)
        done.update(df['ID'].tolist())

    return done


def write_output(preds, answers, classes, noise, ids, out):
    for pred, answer, cls, nse, img_id in list(zip(preds, answers, classes, noise, ids)):
        pred = pred.replace('\n', ' ').replace('\t', ' ')
        answer = answer.replace('\n', ' ').replace('\t', ' ')
        out.write(f'{img_id}\t{pred}\t{answer}\t{cls}\t{nse}\n')

    out.flush()


def collate_llava(data, pad_id):
    input_ids, pixel_values, answer, cls, noise, ids = zip(*data)
    input_ids = pad_sequence(input_ids, batch_first=True, padding_value=pad_id, padding_side='left')
    attention_mask = input_ids.ne(pad_id)
    pixel_values = torch.stack(pixel_values, dim=0)
    answers = list(answer)
    classes = list(cls)
    noise = list(noise)
    ids = list(ids)
    len_prompt = input_ids.shape[1]

    inputs = BatchFeature({'input_ids': input_ids, 'attention_mask': attention_mask, 'pixel_values': pixel_values})
    return inputs, len_prompt, answers, classes, noise, ids
