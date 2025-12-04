# -*- coding: utf-8 -*-
# @Organization  : Alibaba XR-Lab
# @Author        : Peihao Li
# @Email         : liphao99@gmail.com
# @Time          : 2025-03-19 12:47:58
# @Function      : video motion process pipeline
import copy
import json
import os
import sys
from math import sqrt

current_dir_path = os.path.dirname(__file__)
sys.path.append(current_dir_path + "/../pose_estimation")
import argparse
import copy
import gc
import json
import os
import random
import sys
import time

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from blocks import SMPL_Layer
from blocks.detector import DetectionModel
from model import forward_model, load_model
from pose_utils.constants import KEYPOINT_THR
from pose_utils.image import img_center_padding, normalize_rgb_tensor
from pose_utils.inference_utils import get_camera_parameters
from pose_utils.postprocess import OneEuroFilter, smplx_gs_smooth
from pose_utils.render import render_video,render_video2
from pose_utils.tracker import track_by_area
from smplify import TemporalSMPLify

from rembg import remove
from rembg.session_factory import new_session
# import av
from einops import rearrange
from PIL import Image
import imageio
from torchvision import transforms

def bbox_xyxy_to_cxcywh(bboxes: np.ndarray, scale=1.0, device=None):
    bboxes = torch.tensor(bboxes, device=device)
    w = bboxes[..., 2] - bboxes[..., 0]
    h = bboxes[..., 3] - bboxes[..., 1]
    cx = (bboxes[..., 0] + bboxes[..., 2]) / 2.0
    cy = (bboxes[..., 1] + bboxes[..., 3]) / 2.0
    new_bboxes = torch.stack([cx, cy, w * scale, h * scale], dim=-1)
    if device is not None:
        new_bboxes = torch.tensor(new_bboxes, device=device)
    return new_bboxes

def preprocess_image(input_path, output_path,ratio=1.85/2.0, resolution=480):
    
    img_pil = Image.open(input_path).convert("RGB")
    img = np.array(img_pil) # H,W,C=3 
    # remove background
    img_rembg = remove(img, post_process_mask=True, session=new_session("u2net")) 
    # resize & center human
    ret, mask = cv2.threshold(img_rembg[..., -1], 0, 255, cv2.THRESH_BINARY)
    x, y, w, h = cv2.boundingRect(mask)
    max_size = max(w, h)
    side_len = int(max_size / ratio) 
    padded_image = np.zeros((side_len, side_len, 4), dtype=np.uint8)
    center = side_len // 2
    padded_image[
        center - h // 2 : center - h // 2 + h,
        center - w // 2 : center - w // 2 + w,
    ] = img_rembg[y : y + h, x : x + w]
    # resize image
    rgba = Image.fromarray(padded_image).resize((resolution, resolution), Image.LANCZOS)
    # rgba = Image.fromarray(img_rembg)
    # rgba = Image.fromarray(img_rembg)
    # white bg
    rgba_arr = np.array(rgba) / 255.0
    rgb = rgba_arr[..., :3] * rgba_arr[..., -1:] + (1 - rgba_arr[..., -1:])
    rgb_pil = Image.fromarray((rgb * 255).astype(np.uint8))
    # mask
    image = (rgba_arr * 255).astype(np.uint8)
    color_mask = image[..., -1]
    image = (rgb * 255).astype(np.uint8)
    invalid_color_mask = color_mask < 255*0.5
    threshold =  np.ones_like(image[:,:,0]) * 250
    invalid_white_mask = (image[:, :, 0] > threshold) & (image[:, :, 1] > threshold) & (image[:, :, 2] > threshold)
    invalid_color_mask_final = invalid_color_mask & invalid_white_mask
    color_mask = (1 - invalid_color_mask_final) > 0
    mask_pil = Image.fromarray((color_mask * 255).astype(np.uint8))
    rgb_pil.save(output_path)
    return rgb_pil, mask_pil

def preprocess_image2(img_pil,ratio=2/2.0, resolution=480):
    
    img = np.array(img_pil) # H,W,C=3 
    print(img.shape)
    # remove background
    img_rembg = remove(img, post_process_mask=True, session=new_session("u2net")) 
    # resize & center human
    ret, mask = cv2.threshold(img_rembg[..., -1], 0, 255, cv2.THRESH_BINARY)
    h = img_rembg.shape[0]
    w = img_rembg.shape[1]
 
    padded_image = np.zeros((h, int(832/480*h), 4), dtype=np.uint8)
    center_y = h // 2
    center_x = 832//2
    side_len = int(max_size / ratio) 
    padded_image = np.zeros((side_len, h, 4), dtype=np.uint8)
    center = side_len // 2
    padded_image[
        center_y - h // 2 : center_y - h // 2 + h,
        center_x - w // 2 : center_x - w // 2 + w,
    ] = img_rembg
    # resize image
    rgba = Image.fromarray(padded_image).resize((480, 832), Image.LANCZOS)
    # rgba = Image.fromarray(img_rembg)
    # rgba = Image.fromarray(img_rembg)
    # white bg
    rgba_arr = np.array(rgba) / 255.0
    rgb = rgba_arr[..., :3] * rgba_arr[..., -1:] + (1 - rgba_arr[..., -1:])
    rgb_pil = (rgb * 255).astype(np.uint8)

    return rgb_pil

def load_models(model_path, device):
    ckpt_path = os.path.join(model_path, "pose_estimate", "multiHMR_896_L.pt")
    pose_model = load_model(ckpt_path, model_path, device=device)
    print("load hmr")
    pose_model_ckpt = os.path.join(
        model_path, "pose_estimate", "vitpose-h-wholebody.pth"
    )
    keypoint_detector = DetectionModel(pose_model_ckpt, device)
    print("load detection")
    smplx_model = SMPL_Layer(
        model_path,
        type="smplx",
        gender="neutral",
        num_betas=10,
        kid=False,
        person_center="head",
    ).to(device)
    print("load smplx")
    return pose_model, keypoint_detector, smplx_model


class Video2MotionPipeline:
    def __init__(
        self,
        model_path,
        fitting_steps,
        device,
        kp_mode="vitpose",
        visualize=True,
        pad_ratio=0.2,
        fov=60,
    ):
        self.MAX_RESOLUTION = 1280 * 720
        self.device = device
        self.visualize = visualize
        self.kp_mode = kp_mode
        self.pad_ratio = pad_ratio
        self.fov = fov
        self.fps = 30
        self.pose_model, self.keypoint_detector, self.smplx_model = load_models(
            model_path, self.device
        )
        self.smplx_model.to(self.device)
        self.smplify = TemporalSMPLify(
            smpl=self.smplx_model, device=self.device, num_steps=fitting_steps
        )
    def track(self, all_frames):
        self.keypoint_detector.initialize_tracking()
        for frame in all_frames:
            self.keypoint_detector.track(frame,len(all_frames), len(all_frames))
        tracking_results = self.keypoint_detector.process(len(all_frames))
        # note: only surpport pose estimation for one character
        main_character = None
        max_frame_length = -1
        for _id in tracking_results.keys():
            if len(tracking_results[_id]["frame_id"]) > max_frame_length:
                max_frame_length = len(tracking_results[_id]["frame_id"])
                main_character = _id

        bboxes = tracking_results[main_character]["bbox"]
        frame_ids = tracking_results[main_character]["frame_id"]
        frames = [all_frames[i] for i in frame_ids]
        assert not (bboxes[0][0] == 0 and bboxes[0][2] == 0)
        return bboxes, frame_ids, frames
    
def load_video(video_path):
    frames = []
    for i in range(1):
        cap = cv2.VideoCapture(video_path)
        assert cap.isOpened(), f"fail to load video file {video_path}"
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        downsample_factor = -1
        count = 0
        offset_w, offset_h = 0, 0
        while cap.isOpened():
            flag, frame = cap.read()
            if not flag:
                break
            frames.append(frame)
        height, width, _ = frames[0].shape
    return frames, height, width

def images_crop(images, bboxes, target_size, device=torch.device("cuda")):
    # bboxes: cx, cy, w, h
    crop_img_list = []
    crop_annotations = []
    i = 0
    raw_img_size = max(images[0].shape[:2])
    bbox  = bboxes[0]
    left = max(0, int(bbox[0] - bbox[2] // 2))
    right = min(images[0].shape[1] - 1, int(bbox[0] + bbox[2] // 2))
    top = max(0, int(bbox[1] - bbox[3] // 2))
    bottom = min(images[0].shape[0] - 1, int(bbox[1] + bbox[3] // 2))
    for img, _ in zip(images, bboxes):

        crop_img = img[top:bottom, left:right]
        # crop_img = torch.Tensor(crop_img).to(device).unsqueeze(0).permute(0, 3, 1, 2)
        # _, _, h, w = crop_img.shape
        # scale_factor = min(target_size / w, target_size / h)
        # crop_img = F.interpolate(crop_img, scale_factor=scale_factor, mode="bilinear")

        # _, _, h, w = crop_img.shape
        # pad_left = (target_size - w) // 2
        # pad_top = (target_size - h) // 2
        # pad_right = target_size - w - pad_left
        # pad_bottom = target_size - h - pad_top
        # crop_img = F.pad(
        #     crop_img,
        #     (pad_left, pad_right, pad_top, pad_bottom),
        #     mode="constant",
        #     value=0,
        # )
        # resize_img = normalize_rgb_tensor(crop_img)
        crop_img_list.append(crop_img)


    return crop_img_list, crop_annotations

def copy_pic2video(input_path,out_path,num=30):
    img_pil = Image.open(input_path).convert("RGB")
    img = np.array(img_pil) # H,W,C=3 
    writer = imageio.get_writer(
             out_path,
             fps=30, mode='I', format='FFMPEG', macro_block_size=1
        )
    for _ in range(num):
        writer.append_data(img)
    writer.close()

if __name__ == "__main__":

    #这里打包成一个函数
    device = torch.device("cuda:0")
    model_path = "./pretrained_models/human_model_files"
    fitting_steps=[30,50]
    pipeline = Video2MotionPipeline(
        model_path,
        fitting_steps,
        device,
    )
    input = "/data1/zhangbotao/LHM/train_data/example_imgs/4.JPG"
    video_path = "/data1/zhangbotao/LHM/train_data/4.mp4"
    out_path = "/data1/zhangbotao/LHM/train_data/4_crop.mp4"
    copy_pic2video(input,video_path)
    frames,h,w = load_video(video_path)

    bboxes, frame_ids, frames = pipeline.track(frames)
    print(bboxes.size)
    bboxes = bbox_xyxy_to_cxcywh(bboxes, scale=1.0)

    crop_images, crop_annotations = images_crop(
            frames, bboxes, target_size=480, device="cuda"
        )

    writer = imageio.get_writer(
             out_path,
             fps=30, mode='I', format='FFMPEG', macro_block_size=1
        )
    ref = crop_images[0][:,:,::-1]
    img_rembg = remove(ref, post_process_mask=True, session=new_session("u2net")) 
    h = img_rembg.shape[0]
    w = img_rembg.shape[1]
    padded_image = np.zeros((h, int(832/480*h), 4), dtype=np.uint8)
    center_y = h // 2
    center_x = int(832/480*h) // 2
    padded_image[
        center_y - h // 2 : center_y - h // 2 + h,
        center_x - w // 2 : center_x - w // 2 + w,
    ] = img_rembg
    rgba = Image.fromarray(padded_image).resize((832, 480), Image.LANCZOS)
    rgba_arr = np.array(rgba) / 255.0
    rgb = rgba_arr[..., :3] * rgba_arr[..., -1:] + (1 - rgba_arr[..., -1:])
    padded_image = np.array(rgb)*255

    for image in crop_images[0:30]:
        writer.append_data(padded_image)
    writer.close()





    # ref_rgb_pil, ref_mask_pil = preprocess_image(input,output) # remove background & resize & center
    