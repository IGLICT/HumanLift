<!-- # <span><img src="./assets/humanlift_logo.png" height="50" style="vertical-align: top;"> - Official Implementation</span> -->

<div align="center">
<img src="./assets/humanlift_logo.png" height="100" style="vertical-align: top;">
<h1>HumanLift: Single-Image 3D Human Reconstruction with 3D-Aware Diffusion Priors and Facial Enhancement</h1>

<a href="http://geometrylearning.com/HumanLift/index.html"><img src="https://img.shields.io/badge/Project_Page-green" alt="Project Page"></a>


**1. Institute of Computing Technology, Chinese Academy of Sciences**  **2. University of Chinese Academy of Sciences**  
**3. Hong Kong University of Science and Technology**  **4. Cardiff University**

Jie Yang<sup>1</sup>, Bo-Tao Zhang<sup>1,2</sup>, Feng-Lin Liu<sup>1,2</sup>, Hongbo Fu<sup>3</sup>, Yu-Kun Lai<sup>4</sup>, Lin Gao<sup>1,2</sup>

<!-- [Jie Yang](http://people.geometrylearning.com/~jieyang/)<sup>1</sup>, [Bo-Tao Zhang]()<sup>1,2</sup>,[Feng-Lin Liu](http://people.geometrylearning.com/lfl/)<sup>1,2</sup>,[Hongbo Fu](https://hongbofu.people.ust.hk/index.htm)<sup>3</sup>,[Yu-Kun Lai](https://profiles.cardiff.ac.uk/staff/laiy4)<sup>4</sup>,[Lin Gao](http://www.geometrylearning.com/lin/)<sup>1,2</sup> -->

**SIGGRAPH ASIA 2025**
</div>

---

## Overview

![teaser](assets/teaser1.jpg)

**HumanLift** elevates a single reference image to a 3D animatable human, enabling view-consistent and photorealistic full-body image synthesis with high-quality facial details.

---

## Quick Start

### Multi-view Generation

Install the basic dependencies for multi-view generation (based on [DiffSynth](https://github.com/modelscope/DiffSynth-Studio/blob/main/requirements.txt)):

```bash
pip install torch torchvision numpy==1.23 Pillow huggingface_hub
```

Then install the following to obtain SMPL condition images:

```bash
# Install PyTorch3D
pip install "git+https://github.com/facebookresearch/pytorch3d.git"

# Install mmcv-full
pip install "mmcv-full>=1.3.17,<1.6.0" -f https://download.openmmlab.com/mmcv/dist/cu117/torch2.0.1/index.html

# Install mmhuman3d
pip install "git+https://github.com/open-mmlab/mmhuman3d.git"
```

### Reconstruction

Install the 3D Gaussian Splatting package:

```bash
pip install gsplat
```

### Animation (Optional)

To obtain an animatable 3D human, set up the [LHM](https://github.com/aigc3d/LHM) environment and download the pretrained models.

---

## Inference for Reconstruction

### 1. SMPL-X Parameter Estimation

Estimate SMPL-X parameters and render multi-view images from an input image:

```bash
python pose_estimation/video2motion.py \
    --input_path ./images/2.jpg \
    --output_path ./motion \
    --visualize
```

---

### 2. Multi-view RGB Image Generation

Generate multi-view RGB images using the input image and semantic maps.

#### Download Checkpoints:
Download the required model checkpoints from [Google Drive](https://drive.google.com/drive/folders/13M_CQzCaIfSTsyzZD9TFrab221Uf7x2O?usp=sharing) and place them in the `ckpt` (refer to `inference_wan_rgb.py` for the expected path structure).

#### Setup:
- Copy `./images/` to `data/data/` and rename it to `test`
- Copy `./motion/` to `data/output/` and rename it to `test`
- Update model paths in `inference_wan_rgb.py` (the downloaded Wan2.1-14B and fine-tuned weights from our google drive)

#### Run:
```bash
python inference_wan_rgb.py
```
---

### 3. Human Gaussian Reconstruction

#### Pre-processing:
- Remove backgrounds from generated RGB images and save as transparent RGBA
- Pad images to 832×832 resolution
- Copy processed images to `3-gs_recon/data/test/images/` and rename sequentially as `lgt0_r_0000.png` to `lgt0_r_0080.png`

#### Run reconstruction:
```bash
python train.py -s data/test -m output/test
```

---

## Training for Reconstruction

### 1. Configure Environment

Edit `train.sh` to set:
- Dataset path (`dataset_path`)
- Wan2.1-14B model path

### 2. Start Training
```bash
bash train.sh
```

---

## Animation (Alternative Workflow)

> ⚠️ This section provides an alternative animation method that may yield lower quality compared to the main reconstruction pipeline.

### 0. Pose Change (optional)
- Use [WeShopAI Fashion Model Pose Change](https://huggingface.co/spaces/WeShopAI/WeShopAI-Fashion-Model-Pose-Change) to generate a T‑pose image (image A) with the prompt:  
  **"a full-body portrait of a person standing with arms and legs spread apart"**.

### 1. Prepare SMPL Renderings
- Set `IMAGE_INPUT` in `predict.sh` and run:
  ```bash
  bash predict.sh
  ```
- SMPL-rendered images will be saved in `tmp/test/smplimagesrgb`.

### 2. Align Reference Image
- Use Photoshop to align the T‑pose image (image A) with the first SMPL rendering (`000000.png`), producing an aligned reference image (image B).  
- *Why Photoshop?* Current SMPL estimation models are not designed for orthographic camera alignment.

### 3. Generate Multi-view Images
- Place SMPL renderings and image B into HumanWan-Dit and modify hyperparameters in `inference_wan_rgb.py`.
- Run:
  ```bash
  python inference_wan_rgb.py
  ```

### 4. Background Removal
- Remove backgrounds from all 81 multi-view images.
- Save as RGBA format with transparency.

### 5. Run Inference
Set the following paths in `inference.sh`:
- `IMAGE_INPUT`: path to image A (T‑pose)
- `MOTION_SEQS_DIR`: SMPL motion folder
- `DATASET_DIR`: RGBA multi-view images folder

Then run:
```bash
bash inference.sh
```

---

## Acknowledgements

We thank the following open-source projects:  
[DiffSynth](https://github.com/modelscope/DiffSynth-Studio), [LHM](https://github.com/aigc3d/LHM),  
[WeShopAI Fashion Model Pose Change](https://huggingface.co/spaces/WeShopAI/WeShopAI-Fashion-Model-Pose-Change),  
[gsplat](https://github.com/nerfstudio-project/gsplat), and many other inspiring works.

---

## License

[![MIT License](https://img.shields.io/badge/📃-Apache--2.0-929292)](https://www.apache.org/licenses/LICENSE-2.0)

---

If you use this work, please cite our SIGGRAPH ASIA 2025 paper. For questions or issues, open an issue on the repository.


```bibtex
@inproceedings{humanlift2025
  author    = {Yang, Jie and Zhang, Bo-Tao and Liu, Feng-Lin abd Fu, Hongbo and Lai, Yu-Kun and Gao, Lin},
  title     = {HumanLift: Single-Image 3D Human Reconstruction with 3D-Aware Diffusion Priors and Facial Enhancement},
  year      = {2025},
  url       = {https://doi.org/10.1145/3757377.3763839},
  doi       = {10.1145/3757377.3763839},
  booktitle = {SIGGRAPH Asia 2025 Conference Papers (SA Conference Papers '25)},
  articleno = {31},
  numpages  = {12},
  series    = {SIGGRAPH ASIA Conference Papers '25}
}
```
