# GroundSet: Codebase for Cadastral-Grounded Spatial Understanding

[![Paper](https://img.shields.io/badge/arXiv-Paper-b31b1b.svg)]()
[![Dataset](https://img.shields.io/badge/🤗_Hub-Dataset-ffd21e.svg)](https://huggingface.co/datasets/RogerFerrod/GroundSet)
[![Model](https://img.shields.io/badge/🤗_Hub-Model-ffd21e.svg)](https://huggingface.co/RogerFerrod/GroundSet-LLaVA-1.6-7B)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)

This repository contains the official code for data processing, model training, inference and evaluation for the paper **GroundSet: A Cadastral-Grounded Dataset for Spatial Understanding with Vector Data**. 

This work introduces a large-scale Earth Observation dataset (3.8M objects across 510k high-resolution images) and proposes a simple baseline by fine-tuning a standard LLaVA-1.6 architecture.

---

## 🗂️ Repository Structure

The codebase is organized into dedicated modules to support the full pipeline, from training the baseline to evaluating diverse Vision-Language Models (VLMs):

```text
src/
├── eval/                  # Evaluation Scripts
│   ├── eval_caption.py    # Generative metrics (BLEU, CIDEr, etc.) 
│   ├── eval_detection.py  # Grounding metrics (mIoU, F1@0.5, etc.) for object detection
│   └── ...
├── inference/             # VLM Inference Code
│   ├── llava.py           # Inference script for LLaVA
│   ├── gemini.py          # Inference script for Gemini (API)
│   ├── ferret.py          # Inference script for Ferret
│   └── ...
├── resources/             
│   └── tree.json          # Dataset taxonomy
├── dataset/             
│   └── tree.py            # Tree implementation
└── train/                 # Finetuning Code
    ├── finetune.py        # Main training loop
    ├── inference.py       # Model inference
    ├── merge.py           # Script to merge LoRA weights with the base model
    └── zero2.json         # DeepSpeed ZeRO-2 configuration
```

---

## 🚀 Getting Started

### 1. Environment Setup

Clone the repository and install the required dependencies listed in `geo-gpu.txt`. Core libraries include `torch`, `transformers`, `accelerate` and `bitsandbytes`.

You can set up the environment using `pip` or by building the provided `Dockerfile`:
```bash
git clone https://github.com/rogerferrod/GroundSet.git
cd GroundSet
pip install -r src/geo-gpu.txt
```

### 2. Data Preparation
To run the training or evaluation scripts, you must first download the GroundSet dataset from our [Hugging Face Repository](https://huggingface.co/datasets/RogerFerrod/GroundSet). 

> **💡 Note on Taxonomy:** The semantic categories in GroundSet are hierarchical. When evaluating predictions, a query for a parent class (e.g., `Building`) ought to accept all valid subtypes (e.g., `Church`) as positive instances. Please ensure `resources/tree.json` is accessible to your evaluation scripts to handle this taxonomy correctly.

---

## 💻 Usage

### Training the Baseline
We provide the scripts used to fine-tune the LLaVA-1.6-7B baseline. The model was trained using Parameter-Efficient Fine-Tuning (LoRA) via DeepSpeed ZeRO-2 and FlashAttention-2.

To launch training across multiple GPUs (e.g., 8x A100s as reported in our paper ):
```bash
accelerate launch --multi_gpu --num_processes=8 src/train/finetune.py \
    --deepspeed src/train/zero2.json \
    --model_id llava-hf/llava-v1.6-vicuna-7b-hf \
    --dataset_path path/to/groundset/instructions/GroundSet_TRAIN_it.json \
    --image_folder path/to/groundset/finetuning/images \
    --output_dir ./checkpoints/groundset-llava \
    --unfreeze_vision_tower False \
    --gradient_checkpointing True \
    --use_lora True \
    --bf16 True \
    --tf32 True \
    --per_device_train_batch_size 16 \
    --gradient_accumulation_steps 1 \
    --learning_rate 2e-4
```

### Inference
To generate predictions on the GroundSet test set using various open-source or commercial models, use the scripts provided in `src/inference/`. These scripts automatically adapt the input prompts to match the specific template expected by each architecture to ensure optimal zero-shot performance.

For example, for the base llava model:
```bash
python src/inference/llava.py \
    --model llava-hf/llava-v1.6-vicuna-7b-hf \
    --images path/to/groundset/finetuning/images \
    --dataset path/to/groundset/instructions/test.jsonl \
    --output ./inference_dir
```

### Evaluation
Once you have generated a predictions file, use the scripts in `src/eval/` to compute the benchmark metrics.

For example, for object detection:
```bash
python src/eval/eval_detection.py --input ./inference_dir --tree src/resources/tree.json
```

---

## 📝 Citation

If you use this codebase, our dataset or the pre-trained weights in your research, please cite our work:

```bibtex
@article{groundset,
  title={GroundSet: A Cadastral-Grounded Dataset for Spatial Understanding with Vector Data},
  author={Ferrod, Roger and Lecene, Ma{\"e}l and Sapkota, Krishna and Leifman, George and Silverman, Vered and Beryozkin, Genady and Lobry, Sylvain},
  journal={arXiv preprint},
  year={2026}
}
```

## 🙌 Acknowledgements

This work was supported by Google under a research collaboration agreement with Université Paris Cité. The underlying dataset leverages official open-data from IGN (French National Institute of Geographic and Forest Information), specifically BD ORTHO® and BD TOPO®, released under Open Licence 2.0.