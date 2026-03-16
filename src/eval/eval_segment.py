import os
import re
import glob
import json
import argparse
import pandas as pd
import numpy as np
from tqdm import tqdm

import shapely
from shapely import box, geometry
import big_vision.evaluators.proj.paligemma.transfers.segmentation as segeval
import cv2

from dataset.tree import Tree, load_tree

reconstruct_masks = segeval.get_reconstruct_masks('oi')
list_pattern = r'(\[\s*(?:-?\d+(?:\.\d+)?\s*,\s*)*-?\d+(?:\.\d+)?\s*\])'
loc_pattern = r'((?:\s*<loc\d+>\s*){4})'
seg_pattern = r'((?:\s*<seg\d+>\s*){2,})'

img_box = box(0, 0, 672, 672)


def compute_iou(p1, p2):
    intersection = p1.intersection(p2).area
    union = p1.union(p2).area

    iou = intersection / union if union != 0 else 0
    return iou


def extract_polygon(txt, model=None):
    numbers = re.findall(r'(-?\d+(?:\.\d+)?)', txt)
    numbers = [float(x.strip()) for x in numbers]

    coords = numbers
    if model is not None and model in ['ferret', 'finetune']:
        coords = [x / 1000 for x in coords]
    elif model == 'paligemma':
        tmp = [x / 1024 for x in coords]
        coords = [tmp[1], tmp[0], tmp[3], tmp[2]]
    elif model in {'gemini', 'llava', 'llava-next'}:
        if any([int(x) == x for x in coords]):
            coords = [x / 1000 for x in coords]

    coords = [x * 672 for x in coords]
    xs = [coords[i] for i in range(0, len(coords), 2)]
    ys = [coords[i] for i in range(1, len(coords), 2)]
    pixels = list(zip(xs, ys))

    try:
        if len(coords) == 4:
            shape = geometry.box(*coords)
        else:
            shape = geometry.Polygon(pixels)

        if not shapely.is_valid(shape):
            return shape.buffer(0)
    except Exception:
        shape = None

    if shape is not None:
        shape = shape.intersection(img_box)
        if shape.is_empty:
            shape = None

    return shape


def parse_segments(detokenized_output: str) -> np.ndarray | None:
    matches = re.finditer(
        '<loc(?P<y0>\d\d\d\d)><loc(?P<x0>\d\d\d\d)><loc(?P<y1>\d\d\d\d)><loc(?P<x1>\d\d\d\d)>'
        + ''.join(f'<seg(?P<s{i}>\d\d\d)>' for i in range(16)),
        detokenized_output,
    )
    boxes, segs = [], []
    fmt_box = lambda x: float(x) / 1024.0
    for m in matches:
        d = m.groupdict()
        boxes.append([fmt_box(d['y0']), fmt_box(d['x0']), fmt_box(d['y1']), fmt_box(d['x1'])])
        segs.append([int(d[f's{i}']) for i in range(16)])

    if len(segs) == 0:
        return None

    return np.array(reconstruct_masks(np.array(segs)))


def proc_quantized(txt):
    seg_output = parse_segments(txt)

    if seg_output is None:
        return []

    h, w = 672, 672
    x_scale = w / 64
    y_scale = h / 64

    segment_mask = seg_output[0]
    x_coords = np.arange(w)
    y_coords = np.arange(h)
    x_coords = (x_coords / x_scale).astype(int)
    y_coords = (y_coords / y_scale).astype(int)
    resized_array = segment_mask[y_coords[:, np.newaxis], x_coords]

    mask = np.squeeze(resized_array)
    if mask.dtype != np.uint8:
        mask = (mask > 0).astype(np.uint8)

    shapes = []
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cont in contours:
        try:
            coords = [(float(pt[0][0]), float(pt[0][1])) for pt in cont]
            shape = geometry.Polygon(coords)

            if not shapely.is_valid(shape):
                shape = shape.buffer(0)
        except Exception:
            shape = None

        if shape is not None:
            shape = shape.intersection(img_box)
            if shape.is_empty:
                shape = None

        if shape is not None:
            shapes.append(shape)

    return shapes


def proc_simple(output, model):
    preds = []
    for pattern in [list_pattern, loc_pattern]:
        search = re.findall(pattern, output)
        if len(search) > 0:
            break

    for found in search:
        prd = extract_polygon(found, model)
        if prd is not None:
            preds.append(prd)

    return preds


def run(args):
    with open(args.tree, 'r') as tree_file:
        tree = json.load(tree_file)

    tree = Tree(load_tree(tree)[0])

    exclude = []
    exclude.extend(tree.find('Vegetation').get_descendants())
    exclude.extend({'Industrial zone', 'Industrial, agricultural or commercial zone', 'Public space'})
    exclude = set([x.lower() for x in exclude])

    dfs = []
    files = [x for x in glob.glob(os.path.join(args.input, '*.csv')) if re.search(r'segment_[\d]+\.csv$', x)]
    for file in files:
        dfs.append(pd.read_csv(file, sep='\t', header=0))

    df = pd.concat(dfs).fillna(' ')

    tp = {}
    tn = {}
    fp = {}
    fn = {}
    iou_scores = {}
    metrics = [tp, tn, fp, fn]
    for _, row in tqdm(df.iterrows(), total=len(df)):
        output = row['PREDICTION']
        answer = row['GROUND_TRUTH']

        if row['CLASS'].lower() in exclude:
            continue

        node = tree.find(row['CLASS'])
        cls = 'other'
        if node is not None:
            cls = node.get_parent(level=2).name

        shapes = []
        search = re.findall(list_pattern, answer)
        for found in search:
            ground_truth = extract_polygon(found)
            shapes.append(ground_truth)

        output = output.replace('(', '[').replace(')', ']')
        preds = []

        if re.search(seg_pattern, output):
            preds.extend(proc_quantized(output))
        else:
            preds.extend(proc_simple(output, args.model))

        if cls not in iou_scores.keys():
            iou_scores[cls] = []

        for metric in metrics:
            if cls not in metric.keys():
                metric[cls] = 0

        if len(shapes) == 0 and len(preds) == 0:
            tn[cls] += 1
        else:
            matched_indices = set()
            for pred in preds:
                best_iou = 0
                best_idx = -1

                for idx, pol in enumerate(shapes):
                    iou = compute_iou(pred, pol)
                    if iou > best_iou:
                        best_iou = iou
                        best_idx = idx

                iou_scores[cls].append(best_iou)

                if best_iou >= 0.5 and best_idx not in matched_indices:
                    tp[cls] += 1
                    matched_indices.add(best_idx)
                else:
                    fp[cls] += 1

            fn[cls] += len(shapes) - len(matched_indices)

    iou_scores['ALL'] = [y for x in iou_scores.values() for y in x]
    for metric in metrics:
        metric['ALL'] = sum(metric.values())

    with open(os.path.join(args.output, 'segment_scores.txt'), 'w') as out:
        for cls in sorted(iou_scores.keys()):
            miou = sum(iou_scores[cls]) / len(iou_scores[cls]) if len(iou_scores[cls]) > 0 else 0
            p = tp[cls] / (tp[cls] + fp[cls]) if (tp[cls] + fp[cls]) > 0 else 0
            r = tp[cls] / (tp[cls] + fn[cls]) if (tp[cls] + fn[cls]) > 0 else 0
            f1 = (2 * p * r) / (p + r) if (p + r) > 0 else 0
            acc = (tp[cls] + tn[cls]) / (tp[cls] + fn[cls] + tn[cls] + fp[cls])
            out.write(f'CLASS: {cls}\n')
            out.write(
                f'mIoU {miou * 100:.2f}\nP@0.5 {p * 100:.2f}\nR0.5 {r * 100:.2f}\nF1@0.5 {f1 * 100:.2f}\nacc0.5 {acc * 100:.2f}\n\n')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', type=str, default='../output/inference/finetune')
    parser.add_argument('-t', '--tree', type=str, default='../resources/tree.json')
    parser.add_argument('-o', '--output', type=str, default='../output/inference/finetune')

    args = parser.parse_args()
    model = os.path.basename(args.input)
    if model not in {'llava', 'llava-next', 'gemma', 'ferret', 'paligemma', 'gemini', 'finetune'}:
        raise Exception('model not recognized')

    if not os.path.exists(args.output):
        os.mkdir(args.output)

    args.model = model
    run(args)


if __name__ == '__main__':
    main()
