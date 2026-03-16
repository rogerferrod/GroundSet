import os
import re
import glob
import json
import argparse
import pandas as pd
from tqdm import tqdm
from ollama import chat

from dataset.tree import Tree, load_tree

to_strip = {'"', "'", '`', '.', ','}
articles = {'the', 'a', 'an'}

SYS_PROMPT = """
Role:
You are an expert text normalizer and information extractor. Your task is to extract a single, clean classification label from noisy or free-form model outputs.

Goal:
Given a raw text output from another language model, identify and return only the classification label (the named entity or category) mentioned in the text.
The label may appear:
	•	Inside quotation marks
	•	As part of a sentence (e.g., “the label is ‘bridge’”)
	•	Surrounded by punctuation or descriptive phrases

Critical Rule:
	•	Do not invent, rewrite, paraphrase, infer, or transform any label.
	•	You must extract the label exactly as it appears in the input, aside from removing surrounding quotes or punctuation.
	•	If a word or phrase does not appear in the input, you must not output it.

Instructions:
	1.	Carefully read the input text.
	2.	Determine which explicit label or category is being referred to.
	3.	If the model explicitly negates a label (e.g., “no buildings”), do not output the negated label.
	4.	Output only the clean label string, without any quotes, punctuation, or additional words.
	5.	If no clear label can be found, output the empty string "".

Label Guidelines:
    •	Labels do not include moving objects (e.g., cars, airplanes, trains), people, or adjectives/adverbs (e.g., red, grassy).
    •	Valid labels are static geographic or structural categories taken from Open Street Maps.
    •	Examples of valid labels include: church, large sports field, parking lot, vegetation, building.

Output format:
	•	Return only the extracted label (e.g., church)
	•	Do not include any explanations, quotes, or additional text
	
Examples:
    "label that applies is 'bridge'"  -> bridge
    "the correct label is 'road'"  -> road
    "There are no sport fields in the image"  -> '' (empty string)
    "Answer: a castle"  ->    castle
    "This image most likely contains a parking lot."  -> parking lot
    "It’s clearly showing a roundabout."  -> roundabout
    
Now read the following text and output only the extracted label:
"""


def ask_llm(text):
    response = chat(model='gemma3:4b',
                    options={"num_predict": 5, "temperature": 0, "top_k": 1}, messages=[
            {
                'role': 'system',
                'content': SYS_PROMPT,
            },
            {
                'role': 'user',
                'content': text
            },
        ])

    pred = response.message.content
    return pred


def run(args):
    with open(args.tree, 'r') as tree_file:
        tree = json.load(tree_file)

    tree = Tree(load_tree(tree)[0])
    labels = set(tree.root.get_descendants())

    dfs = []
    files = [x for x in glob.glob(os.path.join(args.input, '*.csv')) if re.search(r'classification_[\d]+\.csv$', x)]
    for file in files:
        dfs.append(pd.read_csv(file, sep='\t', header=0))

    df = pd.concat(dfs).fillna(' ')
    with open(os.path.join(args.output, 'classification.csv'), 'w') as out:
        out.write('ID\tPREDICTION\tGROUND_TRUTH\tCLASS\tNOISE\n')

        for _, row in tqdm(df.iterrows(), total=len(df)):
            pred = row['PREDICTION'].lower()
            answer = row['GROUND_TRUTH'].lower()

            label = pred
            for remove in to_strip:
                label = label.strip(remove)

            tokens = label.split(' ')
            filtered = [word for word in tokens if word not in articles]
            label = ' '.join(filtered)
            label = label.strip()

            if label != answer and label not in labels:
                label = ask_llm(pred).strip()

            out.write(f'{row['ID']}\t{label}\t{answer}\t{row['CLASS']}\t{row['NOISE']}\n')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', type=str, default='../output/inference/finetune')
    parser.add_argument('-t', '--tree', type=str, default='../resources/tree.json')
    parser.add_argument('-o', '--output', type=str, default='../output/inference/finetune')

    args = parser.parse_args()

    if not os.path.exists(args.output):
        os.mkdir(args.output)

    run(args)


if __name__ == '__main__':
    main()
