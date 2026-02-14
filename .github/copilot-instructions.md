# Copilot Instructions for gradproj

## Project overview
- Research-oriented ML codebase mixing scripts and notebooks for medical imaging, VQA, and text-to-image finetuning.
- Primary work lives under [experiments/](experiments/) (training scripts, preprocessing, evaluation) and several notebooks (see [experiments/readme.md](experiments/readme.md)).
- Data is external and referenced via **absolute paths** inside scripts; expect to update paths for your machine.

## Key components & data flow
- Image classification finetuning uses Hugging Face Accelerate + Transformers in [experiments/run_image_classification_no_trainer.py](experiments/run_image_classification_no_trainer.py). The entrypoint is CLI-driven and writes checkpoints under `experiments/runs/`.
- Batch experiment launcher is [experiments/run_image_classification.sh](experiments/run_image_classification.sh), iterating over multiple model backbones and calling `accelerate launch`.
- Text-to-image finetuning is in [experiments/fine-tuning-text-to-image.py](experiments/fine-tuning-text-to-image.py); it builds `Dataset` classes from CSV/JSON and trains Stable Diffusion (Diffusers + CLIP).
- XML annotation preprocessing for the real-colon dataset is in [experiments/xml_parser.py](experiments/xml_parser.py); it converts XML to JSON, CSV, and YOLO labels and writes annotated images.
- Kvasir VQA dataset download/inspection is in [kvasir_vqa.py](kvasir_vqa.py) (HF dataset loader + optional export).

## Developer workflows & commands
- Image classification training is typically run via `accelerate launch` from [experiments/run_image_classification.sh](experiments/run_image_classification.sh).
- Text-to-image finetuning is run directly from [experiments/fine-tuning-text-to-image.py](experiments/fine-tuning-text-to-image.py) (expects GPU, `bitsandbytes`, `diffusers`, `accelerate`).
- Environment dependencies are tracked separately for RAG and TTI work: [rag_requirements.txt](rag_requirements.txt) and [tti_requirements.txt](tti_requirements.txt).

## Project-specific conventions
- Hardcoded absolute dataset paths are common in scripts (e.g., `/home/qzhao9/datasets/...`, `/media/...`). Update these before running.
- Output artifacts are stored in repo folders: `fine_tuned_models/`, `fine_tuning_logs/`, `experiments/runs/`, and `lora_sd_turbo_150/` (LoRA weights).
- Many experiments are done in notebooks; prefer updating the corresponding script if there is both a .py and .ipynb for the same workflow (see [experiments/readme.md](experiments/readme.md)).

## Integration points / dependencies
- Hugging Face ecosystems: `datasets`, `transformers`, `accelerate`, `diffusers`, and `huggingface_hub` are central across scripts.
- CV utilities: `opencv` is used for annotation visualization and label generation in [experiments/xml_parser.py](experiments/xml_parser.py).

## Examples to follow
- For CLI argument patterns and training loop structure, mirror [experiments/run_image_classification_no_trainer.py](experiments/run_image_classification_no_trainer.py).
- For dataset construction from caption files, mirror `KvasirDataset`/`DiagramDataset` in [experiments/fine-tuning-text-to-image.py](experiments/fine-tuning-text-to-image.py).
