import os
import argparse

from tqdm import tqdm
from PIL import Image

from google import genai
from google.genai import types
from google.genai.errors import ClientError

from torch.utils.data import Dataset, DataLoader
from utils import load_data, collect_done, write_output

from concurrent.futures import ThreadPoolExecutor, as_completed

from utils import Config

SYS_PROMPT = """
You are a multimodal AI assistant specialized in understanding and analyzing Remote Sensing imagery.  
You can interpret points, lines and polygons in tasks such as object detection, segmentation, classification, captioning and visual question answering (VQA).

Follow these instructions precisely:

**Coordinate System**
- All coordinates are normalized to the range [0, 1].
- The origin (0, 0) is the top-left corner of the image.
- The x-axis increases to the right; the y-axis increases downward.

**Geometry Representations**
- Polygon (segmentation mask): [x₁, y₁, x₂, y₂, ..., xₙ, yₙ]
- Bounding box: [x_min, y_min, x_max, y_max]

**Task-Specific Output Rules**
1. Object Localization / Detection:  
   - Output only the list of coordinates following the appropriate format.  
   - If no object is detected, return an empty list `[]`.

2. Classification / Labeling:  
   - Output only the short label (e.g., `"bridge"`, `"river"`, `"church"`).  
   - Do not include confidence scores, explanations or reasoning.

3. Visual Question Answering (VQA):  
   - For yes/no questions, output only `"yes"` or `"no"`.  

4. Image Captioning:  
   - Produce a concise 3–4 sentence caption.  
   - Begin with a general overview of the area (e.g., “This is a residential zone combining…”).  
   - Then describe the main or central feature, followed by secondary objects and spatial context with respect to the image (e.g., “on the left,” “to the south,” “above,” “below”).  
   - The caption must be natural, informative and grounded in visible content.

**General Rules**
- Never include explanations, reasoning or meta-text (e.g., “The image shows…” or “I think…”).  
- Return only the final answer in the specified format.  
- If the image does not provide sufficient information, return an empty list or empty string.

Your responses must be deterministic, minimal and conform exactly to the expected output schema.
"""


class EvalDataset(Dataset):
    def __init__(self, annotations, img_dir, done):
        self.data = load_data(annotations, done)
        self.img_dir = img_dir

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        elem = self.data[idx]
        img_path = os.path.join(self.img_dir, elem['image'])
        image = img_path
        question = elem['question']
        answer = elem['answer']
        img_id = elem['id']
        cls = elem['class'] if 'class' in elem.keys() else 'null'
        noise = str(elem['noise']) if 'noise' in elem.keys() else 'false'

        return question, image, answer, cls, noise, img_id


def thread(batch, client, model_name, config):
    question, image, answer, cls, noise, idx = batch

    if not os.path.exists(image[0]):
        print(f"Skipping ID {idx}: Image not found at {image[0]}")
        return None

    try:
        with Image.open(image[0]) as img:
            response = client.models.generate_content(
                model=model_name,
                config=types.GenerateContentConfig(
                    system_instruction=SYS_PROMPT,
                    temperature=config.temperature,
                    top_k=config.top_k,
                    top_p=config.top_p,
                    max_output_tokens=config.max_tok
                ),
                contents=[img, question[0]]
            )

            if response is None:
                return None

            if not response.candidates:
                print(f"ID {idx}: Blocked by safety filters")
                return None

            candidate = response.candidates[0]

            valid_reasons = ["STOP", "MAX_TOKENS"]
            if candidate.finish_reason and candidate.finish_reason.name not in valid_reasons:
                print(f"ID {idx}: Generation stopped due to {candidate.finish_reason.name}")
                return None

            if not candidate.content or not candidate.content.parts:
                return None

            text_output = candidate.content.parts[0].text

            if not text_output:
                return None

            pred = [text_output]

        return pred, answer, cls, noise, idx

    except ClientError as e:
        if "429" in str(e) or "ResourceExhausted" in str(e):
            raise e

        print(f"ClientError for ID {idx}: {e}")
        return None

    except Exception as e:
        print(f"Unexpected error for ID {idx}: {e}")
        return None


def run(args, config):
    client = genai.Client(api_key=os.environ.get('GEMINI_API_KEY'))

    done = collect_done(args.output, config.task)
    dataset = EvalDataset(args.dataset, args.images, done)
    dataloader = DataLoader(dataset, shuffle=False)

    out_path = os.path.join(args.output, f'{config.task}_0.csv')
    is_new = not os.path.exists(out_path)
    with open(out_path, 'a') as out:
        if is_new:
            out.write('ID\tPREDICTION\tGROUND_TRUTH\tCLASS\tNOISE\n')
            out.flush()

        with ThreadPoolExecutor(max_workers=args.worker) as executor:
            future_to_batch = {executor.submit(thread, batch, client, args.model, config): batch for batch in
                               dataloader}

            try:
                for future in tqdm(as_completed(future_to_batch), total=len(dataloader)):
                    result = future.result()

                    if result is not None:
                        pred, answer, cls, noise, idx = result
                        write_output(pred, answer, cls, noise, idx, out)
                        out.flush()

            except Exception as e:
                if "429" in str(e) or "ResourceExhausted" in str(e):
                    print("\nDaily Quota Reached. Stopping execution safely.")
                    return
                else:
                    print(f"\nCritical Error: {e}")
                    return


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--images', type=str, default='../input/images')
    parser.add_argument('-d', '--dataset', type=str,
                        default='../output/annotations/VRSBench/test/VRSBench_EVAL_caption.jsonl')
    parser.add_argument('-m', '--model', type=str, default='gemini-2.5-flash')
    parser.add_argument('-w', '--worker', type=int, default=16)
    parser.add_argument('-o', '--output', type=str, default='../output/inference/gemini')

    args = parser.parse_args()
    task = args.dataset.split('_EVAL_')[-1].split('.')[0]
    config = Config(task)
    print(task)
    if not os.path.exists(args.output):
        os.mkdir(args.output)

    run(args, config)


if __name__ == "__main__":
    main()
