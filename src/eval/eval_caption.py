import os
import glob
import argparse
import pandas as pd
from tqdm import tqdm
import re

from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.meteor_score import meteor_score
from nltk.tokenize import word_tokenize
from cider.cider import Cider

special_tokens = {'<p>', '</p>', '<delim>'}
list_pattern = r'(\[\s*(?:-?\d+(?:\.\d+)?\s*,\s*)*-?\d+(?:\.\d+)?\s*\])'
angle_pattern = r'(\{\s*(?:<\s*-?\d+(?:\.\d+)?\s*>)+\s*})'
rbox_pattern = r'(\{\s*(?:<\s*-?\d+(?:\.\d+)?\s*>)+\s*\|\s*<\s*-?\d+(?:\.\d+)?\s*>\s*\})'
patterns = [list_pattern, angle_pattern, rbox_pattern]


def run(args):
    bleu_scores = []
    meteor_scores = []
    cider_scores = []
    cider = Cider()

    dfs = []
    files = glob.glob(os.path.join(args.input, 'caption_*.csv'))
    for file in files:
        dfs.append(pd.read_csv(file, sep='\t', header=0))

    df = pd.concat(dfs).fillna(' ')
    for _, row in tqdm(df.iterrows(), total=len(df)):
        caption = row['PREDICTION'].lower()
        answer = row['GROUND_TRUTH'].lower()

        for tok in special_tokens:
            caption = caption.replace(tok, '')

        for regex in patterns:
            caption = re.sub(regex, '', caption)

        caption = re.sub(' +', ' ', caption).strip()

        reference = word_tokenize(answer)
        hypothesis = word_tokenize(caption)
        meteor_scores.append(meteor_score([reference], hypothesis))
        bleu_scores.append(sentence_bleu([reference], hypothesis, weights=(0.25, 0.25, 0.25, 0.25)))
        cider_score, _ = cider.compute_score([[answer]], [[caption]])
        cider_scores.append(cider_score)

    bleu = (sum(bleu_scores) / len(bleu_scores)) * 100
    meteor = (sum(meteor_scores) / len(meteor_scores)) * 100
    cider = (sum(cider_scores) / len(cider_scores)) * 100

    with open(os.path.join(args.output, 'captioning_scores.txt'), 'w') as out:
        out.write(f'BLEU@4 {bleu:.2f}\nMETEOR {meteor:.2f}\nCIDEr {cider:.2f}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', type=str, default='../output/inference/finetune')
    parser.add_argument('-o', '--output', type=str, default='../output/inference/finetune')

    args = parser.parse_args()

    if not os.path.exists(args.output):
        os.mkdir(args.output)

    run(args)


if __name__ == '__main__':
    main()
