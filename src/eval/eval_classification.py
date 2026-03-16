import os
import json
import argparse
import pandas as pd
from tqdm import tqdm

import torch
from sentence_transformers import SentenceTransformer, SimilarityFunction

from dataset.tree import Tree, load_tree


def run(args):
    with open(args.tree, 'r') as tree_file:
        tree = json.load(tree_file)

    tree = Tree(load_tree(tree)[0])

    df = pd.read_csv(os.path.join(args.input, 'classification.csv'), sep='\t', header=0).fillna(' ')

    device = torch.device('cpu')
    if torch.cuda.is_available():
        device = torch.device('cuda')

    model = SentenceTransformer('all-mpnet-base-v2', similarity_fn_name=SimilarityFunction.COSINE)
    model = model.to(device)

    similarities = {}
    matches = {}
    for _, row in tqdm(df.iterrows(), total=len(df)):
        pred = row['PREDICTION'].lower()
        answer = row['GROUND_TRUTH'].lower()
        if args.exclude_noise:
            if row['NOISE']:
                continue

        for punct in {'"', "'", '`', '.', ','}:
            pred = pred.strip(punct)

        pred = pred.strip()

        node = tree.find(row['CLASS'])
        cls = 'other'
        if node is not None:
            cls = node.get_parent(level=2).name

        if cls not in similarities.keys():
            similarities[cls] = []
        if cls not in matches.keys():
            matches[cls] = []

        matches[cls].append(1 if pred == answer else 0)

        emb = model.encode([pred, answer])
        sim = model.similarity(emb[0], emb[1]).item()
        similarities[cls].append(sim)

    similarities['ALL'] = [y for x in similarities.values() for y in x]
    matches['ALL'] = [y for x in matches.values() for y in x]

    with open(os.path.join(args.output, 'classification_scores.txt'), 'w') as out:
        for cls in sorted(similarities.keys()):
            sims = similarities[cls]
            list08 = [x for x in sims if x >= 0.8]
            list06 = [x for x in sims if x >= 0.6]
            list04 = [x for x in sims if x >= 0.4]

            acc08 = len(list08) / len(sims)
            acc06 = len(list06) / len(sims)
            acc04 = len(list04) / len(sims)

            accuracy = sum(matches[cls]) / len(matches[cls])

            out.write(f'CLASS: {cls}\n')
            out.write(f'Accuracy {accuracy * 100:.2f}\n')
            out.write(
                f'acc@0.8 {acc08 * 100:.2f}\nacc@0.6 {acc06 * 100:.2f}\nacc@0.4 {acc04 * 100:.2f}\n\n')


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
