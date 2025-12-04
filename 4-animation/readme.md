## Environment

* ** Follow [LHM](https://github.com/aigc3d/LHM)** environment + pretrained model for prior

---

## Pipeline Overview

### 0. Pose Change  
- Use the Hugging Face Space: [WeShopAI Fashion Model Pose Change](https://huggingface.co/spaces/WeShopAI/WeShopAI-Fashion-Model-Pose-Change) by WeShopAI.  
- Control pose changes using text prompts:**a full-body portrait of a person standing with arms and legs spread apart**. 
- We get T-pose image(image A).

### 1. Preparation for HumanWan-Dit
* Edit IMAGE_INPUT in **predict.sh** and run. You can get smpl rendering images in "tmp/test/smplimagesrgb".

```bash
bash predict.sh
```

* Get ref image input in HumanWan-Dit: Use **Photoshop** to align the **reference image(image A)** with the first SMPL-rendered image("tmp/test/smplimagesrgb/000000.png"), producing a new aligned ref image(image B).
Q:why we use PS?
A:we can't find smpl estimation model suitable for orthographic camera

### 2. Run Wandit
* Put smpl rendering images and aligned ref image(image B) into HumanWan-Dit. We get multi-view images. Refer to '../ReadMe.md' to generate the multiview-images.

```bash
cd ../2-mv-gen/
python inference_wan_rgb.py
```

### 3. Multi-view Background Removal (RGBA)
* Remove the background of the 81 generated multi-view images with our HumanWan-DiT model.
* Save them as **RGBA** format for transparent backgrounds.


### 4. Inference
modify IMAGE_INPUT(**T-pose image(image A)** path), MOTION_SEQS_DIR(motion smpl path), DATASET_DIR(RGBA multi-view images) in inference.sh. Run the following script:
```bash
bash inference.sh
```


## Note
> * follow **[LHM](https://github.com/aigc3d/LHM)** to download the weights in "./pretrained_models" and "./third_parties"
> * RUN and GET motion smpl list: **python ./engine/pose_estimation/video2motion.py --video_path {VIDEO_PATH} --output_path {OUTPUT_PATH}**
> * The inference outputs follow the same directory structure as LHM.



