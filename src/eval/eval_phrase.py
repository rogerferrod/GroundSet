import os
import re
import glob
import argparse
import pandas as pd
import numpy as np
from tqdm import tqdm

from shapely import box, geometry

list_pattern = r'(\[\s*(?:-?\d+(?:\.\d+)?\s*,\s*)*-?\d+(?:\.\d+)?\s*\])'
angle_pattern = r'(\{\s*(?:<\s*-?\d+(?:\.\d+)?\s*>)+\s*})'
rbox_pattern = r'(\{\s*(?:<\s*-?\d+(?:\.\d+)?\s*>)+\s*\|\s*<\s*-?\d+(?:\.\d+)?\s*>\s*\})'
loc_pattern = r'((?:\s*<loc\d+>\s*){4})'

patterns = [list_pattern, angle_pattern, rbox_pattern, loc_pattern]

img_box = box(0, 0, 672, 672)


def compute_iou(p1, p2):
    intersection = p1.intersection(p2).area
    union = p1.union(p2).area

    iou = intersection / union if union != 0 else 0
    return iou


def parse_obb(numbers, model):
    coords = numbers[:-1]
    a = numbers[-1]

    if len(coords) != 4:
        return None

    coords = [x / 100 for x in coords]
    coords = [x * 672 for x in coords]

    if model == 'geochat':
        x_left, y_top, x_right, y_bottom = coords
        x = (x_left + x_right) / 2
        y = (y_top + y_bottom) / 2
        w = x_right - x_left
        h = y_bottom - y_top
    else:
        x, y, w, h = coords

    theta = np.deg2rad(a)
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)

    dx = w / 2
    dy = h / 2

    corners = []
    for cx, cy in [(-dx, -dy), (dx, -dy), (dx, dy), (-dx, dy)]:
        rx = x + cx * cos_t - cy * sin_t
        ry = y + cx * sin_t + cy * cos_t
        corners.append((rx, ry))

    try:
        shape = geometry.Polygon(corners)
    except Exception:
        shape = None

    return shape


def parse_hbb(numbers, model):
    coords = numbers
    if model in ['ferret', 'finetune']:
        coords = [x / 1000 for x in coords]
    elif model in {'minigpt', 'geochat', 'skysense', 'vrsbench'}:
        coords = [x / 100 for x in coords]
    elif model == 'paligemma':
        tmp = [x / 1024 for x in coords]
        coords = [tmp[1], tmp[0], tmp[3], tmp[2]]
    elif model in {'gemini', 'llava', 'llava-next'}:
        if any([int(x) == x for x in coords]):
            coords = [x / 1000 for x in coords]

    coords = [x * 672 for x in coords]

    try:
        shape = geometry.box(*coords)
    except Exception:
        shape = None

    return shape


def extract_polygon(txt, model=None):
    numbers = re.findall(r'(-?\d+(?:\.\d+)?)', txt)
    numbers = [float(x.strip()) for x in numbers]

    if '|' in txt and model in {'geochat', 'skysense'}:
        shape = parse_obb(numbers, model)
    else:
        shape = parse_hbb(numbers, model)

    if shape is not None:
        shape = shape.intersection(img_box)
        if shape.is_empty:
            shape = None

    return shape


def run(args):
    dfs = []
    files = [x for x in glob.glob(os.path.join(args.input, '*.csv')) if re.search(r'phrase_[\d]+\.csv$', x)]
    for file in files:
        dfs.append(pd.read_csv(file, sep='\t', header=0))

    df = pd.concat(dfs).fillna(' ')

    tp = tn = fp = fn = 0
    iou_scores = []
    for _, row in tqdm(df.iterrows(), total=len(df)):
        output = row['PREDICTION']
        answer = row['GROUND_TRUTH']

        boxes = []
        search = re.findall(list_pattern, answer)
        for found in search:
            ground_truth = extract_polygon(found)
            boxes.append(ground_truth)

        output = output.replace('(', '[').replace(')', ']')
        preds = []
        for pattern in patterns:
            search = re.findall(pattern, output)
            if len(search) > 0:
                break

        for found in search:
            prd = extract_polygon(found, args.model)
            if prd is not None:
                preds.append(prd)

        if len(boxes) == 0 and len(preds) == 0:
            tn += 1
        else:
            matched_indices = set()
            for pred in preds:
                best_iou = 0
                best_idx = -1

                for idx, box in enumerate(boxes):
                    iou = compute_iou(pred, box)
                    if iou > best_iou:
                        best_iou = iou
                        best_idx = idx

                iou_scores.append(best_iou)

                if best_iou >= 0.5 and best_idx not in matched_indices:
                    tp += 1
                    matched_indices.add(best_idx)
                else:
                    fp += 1  # IoU < 0.5 or already claimed by a previous prediction

            fn += len(boxes) - len(matched_indices)

    miou = sum(iou_scores) / len(iou_scores)
    p = tp / (tp + fp)
    r = tp / (tp + fn)
    f1 = (2 * p * r) / (p + r)
    acc = (tp + tn) / (tp + fn + tn + fp)
    with open(os.path.join(args.output, 'phrase_scores.txt'), 'w') as out:
        out.write(
            f'mIoU {miou * 100:.2f}\nP@0.5 {p * 100:.2f}\nR0.5 {r * 100:.2f}\nF1@0.5 {f1 * 100:.2f}\nacc0.5 {acc * 100:.2f}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', type=str, default='../output/inference/finetune')
    parser.add_argument('-o', '--output', type=str, default='../output/inference/finetune')

    args = parser.parse_args()
    model = os.path.basename(args.input)
    if model not in {'llava', 'llava-next', 'gemma', 'gemini', 'geochat', 'minigpt', 'skysense', 'ferret', 'paligemma',
                     'vrsbench', 'finetune'}:
        raise Exception('model not recognized')

    if not os.path.exists(args.output):
        os.mkdir(args.output)

    args.model = model
    run(args)


if __name__ == '__main__':
    main()
