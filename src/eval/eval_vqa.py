import os
import re
import glob
import json
import argparse
import pandas as pd
from tqdm import tqdm

from dataset.tree import Tree, load_tree


def run(args):
    with open(args.tree, 'r') as tree_file:
        tree = json.load(tree_file)

    tree = Tree(load_tree(tree)[0])

    dfs = []
    files = [x for x in glob.glob(os.path.join(args.input, '*.csv')) if re.search(r'vqa_[\d]+\.csv$', x)]
    for file in files:
        dfs.append(pd.read_csv(file, sep='\t', header=0))

    df = pd.concat(dfs).fillna(' ')

    tp = {}
    tn = {}
    fp = {}
    fn = {}
    metrics = [tp, tn, fp, fn]
    classes = set()
    for _, row in tqdm(df.iterrows(), total=len(df)):
        pred = row['PREDICTION'].lower()
        answer = row['GROUND_TRUTH'].lower()
        if args.exclude_noise:
            if row['NOISE']:
                continue

        node = tree.find(row['CLASS'])
        cls = 'other'
        if node is not None:
            cls = node.get_parent(level=2).name

        for punct in {'"', "'", '`', '.', ','}:
            pred = pred.strip(punct)

        pred = pred.strip()

        classes.add(cls)
        for metric in metrics:
            if cls not in metric.keys():
                metric[cls] = 0

        if answer == 'yes':
            if pred[:3] == 'yes':
                tp[cls] += 1
            else:
                fn[cls] += 1
        else:
            if pred[:2] == 'no':
                tn[cls] += 1
            else:
                fp[cls] += 1

    classes.add('ALL')
    for metric in metrics:
        metric['ALL'] = sum(metric.values())

    with open(os.path.join(args.output, 'vqa_scores.txt'), 'w') as out:
        for cls in sorted(classes):
            p = tp[cls] / (tp[cls] + fp[cls]) if (tp[cls] + fp[cls]) > 0 else 0
            r = tp[cls] / (tp[cls] + fn[cls]) if (tp[cls] + fn[cls]) > 0 else 0
            f1 = (2 * p * r) / (p + r) if (p + r) > 0 else 0
            acc = (tp[cls] + tn[cls]) / (tp[cls] + fn[cls] + tn[cls] + fp[cls])
            out.write(f'CLASS: {cls}\n')
            out.write(f'P {p * 100:.2f}\nR {r * 100:.2f}\nF1 {f1 * 100:.2f}\nacc {acc * 100:.2f}\n\n')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', type=str, default='../output/inference/finetune')
    parser.add_argument('-t', '--tree', type=str, default='../resources/tree.json')
    parser.add_argument('-o', '--output', type=str, default='../output/inference/finetune')
    parser.add_argument('--exclude_noise', action='store_true', default=False)

    args = parser.parse_args()

    if not os.path.exists(args.output):
        os.mkdir(args.output)

    run(args)


if __name__ == '__main__':
    main()
