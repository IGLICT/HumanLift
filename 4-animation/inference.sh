#!/bin/bash
# given pose sequence, generating animation video .

MODEL_NAME=LHM-1B

IMAGE_INPUT="./data/ref_image/31" 
MOTION_SEQS_DIR="./data/motion_video/jntm/smplx_params/"
DATASET_DIR="./data/multiview_images/frames_480P_31processnew"

MODEL_NAME=${1:-$MODEL_NAME}
IMAGE_INPUT=${2:-$IMAGE_INPUT}
MOTION_SEQS_DIR=${3:-$MOTION_SEQS_DIR}

echo "IMAGE_INPUT: $IMAGE_INPUT"
echo "MODEL_NAME: $MODEL_NAME"
echo "MOTION_SEQS_DIR: $MOTION_SEQS_DIR"

echo "INFERENCE VIDEO"

MOTION_IMG_DIR=None
VIS_MOTION=true
MOTION_IMG_NEED_MASK=true
RENDER_FPS=30
MOTION_VIDEO_READ_FPS=30
EXPORT_VIDEO=True
ONLY_PREDICT_SMPL=False

CUDA_VISIBLE_DEVICES=0 python -m LHM.launch infer.human_lrm model_name=$MODEL_NAME \
        image_input=$IMAGE_INPUT \
        export_video=$EXPORT_VIDEO \
        motion_seqs_dir=$MOTION_SEQS_DIR motion_img_dir=$MOTION_IMG_DIR  \
        vis_motion=$VIS_MOTION motion_img_need_mask=$MOTION_IMG_NEED_MASK \
        render_fps=$RENDER_FPS motion_video_read_fps=$MOTION_VIDEO_READ_FPS \
        dataset_dir=$DATASET_DIR \
        only_predict_smpl=$ONLY_PREDICT_SMPL