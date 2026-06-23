# IPTNet

Curated source export for IPTNet, an RGB-thermal object detection model built on DINO with ImageBind features, FFM/STN alignment, SuperFusion matching, and selected LPMFlow components.

## Included

- `main.py` and `engine.py` for training and evaluation.
- Current configs in `config/DINO/`, including `DINO_4scaleSD.py`, `DINO_4scalesee.py`, and `DINO_4scale.py`.
- The active model path under `models/dino/dinobind.py` and its local dependencies.
- Dataset, COCO evaluation, and utility modules required by the active entry point.
- Multi-scale deformable attention C++/CUDA source under `models/dino/ops/`.
- ImageBind tokenizer asset under `models/dino/imagebind/bpe/`.

## Excluded

- Python caches, build outputs, compiled extensions, and egg metadata.
- Checkpoints, local `resume/` files, logs, outputs, and datasets.
- The standalone `SD/` Stable Diffusion tree and image/demo assets.
- Historical backup modules and scripts that are not required by the current `main.py` model path.

## Setup

Install Python dependencies:

```bash
pip install -r requirements.txt
```

Build the deformable attention extension:

```bash
cd models/dino/ops
python setup.py build install
cd ../../..
```

## Temporary Torchvision Patch

This repository currently includes a temporary `vision.py` compatibility patch for dataset loading. Before running the current code, replace the matching file in your installed torchvision package with the repository copy.

You can locate the installed file with:

```bash
python -c "import torchvision.datasets.vision as v; print(v.__file__)"
```

Back up that file first, then copy this repository's `vision.py` over the printed path, which is usually similar to `.../site-packages/torchvision/datasets/vision.py`.

This is a temporary workaround for the current data-loading code. The project will be cleaned up later so it no longer requires modifying files inside torchvision. Sorry for the extra setup complexity while the code is still being organized.

## Run

The default entry point uses `config/DINO/DINO_4scaleSD.py`:

```bash
python main.py \
  -c config/DINO/DINO_4scaleSD.py \
  --coco_path /path/to/rgb_dataset \
  --coco_path2 /path/to/thermal_dataset \
  --pretrain_model_path /path/to/dino_checkpoint.pth \
  --bind_path /path/to/imagebind_huge.pth
```

Model weights and datasets are intentionally not committed.
