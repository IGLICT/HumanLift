# -*- coding: utf-8 -*-
# @Organization  : Alibaba XR-Lab
# @Author        : Peihao Li
# @Email         : liphao99@gmail.com
# @Time          : 2025-03-11 12:47:58
# @Function      : inference code for pose estimation

import os
import sys
import gc
sys.path.append("./")

import pdb
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import cv2
from engine.ouputs import BaseOutput
# from engine.pose_estimation.model import load_model
# from engine.pose_estimation.pose_utils.tracker import bbox_xyxy_to_cxcywh, track_by_area

current_dir_path = os.path.dirname(__file__)
sys.path.append(current_dir_path + "/../pose_estimation")
from blocks import SMPL_Layer
from blocks.detector import DetectionModel
from model import forward_model, load_model
from pose_utils.constants import KEYPOINT_THR
from pose_utils.image import img_center_padding, normalize_rgb_tensor
from pose_utils.inference_utils import get_camera_parameters
from pose_utils.postprocess import OneEuroFilter, smplx_gs_smooth
from pose_utils.render import render_video,render_video2,get_mesh
from pose_utils.tracker import bbox_xyxy_to_cxcywh, track_by_area
from smplify import TemporalSMPLify
import json
from rembg import remove
from rembg.session_factory import new_session
import smplx
sys.path.append("./thirdparties/econ")
from thirdparties.econ.lib.common.smpl_utils import (
    SMPLEstimator, SMPLRenderer,
    save_optimed_video, save_optimed_smpl_param, save_optimed_mesh,
)

IMG_NORM_MEAN = [0.485, 0.456, 0.406]
IMG_NORM_STD = [0.229, 0.224, 0.225]

def load_image(image_path,resolution=512):
    frames = []
    image = cv2.imread(image_path)
    assert image is not None, f"fail to load image file {image_path}"

    # 获取原始图片尺寸
    original_height, original_width = image.shape[:2]
    # 目标尺寸
    target_height = 832
    target_width = 480
    # 计算宽高比
    target_ratio = target_width / target_height
    original_ratio = original_width / original_height
    # 方法1：裁剪并缩放（保持宽高比，裁剪多余部分）
    def crop_and_resize(image, target_width, target_height):
        # 计算缩放比例
        scale_ratio = max(target_width / original_width, target_height / original_height)
        new_width = int(original_width * scale_ratio)
        new_height = int(original_height * scale_ratio)
        # 缩放图片
        resized_image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
        # 计算裁剪位置（居中裁剪）
        start_x = max(0, (new_width - target_width) // 2)
        start_y = max(0, (new_height - target_height) // 2)
        # 裁剪图片
        cropped_image = resized_image[start_y:start_y + target_height, start_x:start_x + target_width]
        return cropped_image
    # 方法2：填充并缩放（保持宽高比，填充不足部分）
    def pad_and_resize(image, target_width, target_height, pad_color=(0, 0, 0)):
        # 计算缩放比例
        scale_ratio = min(target_width / original_width, target_height / original_height)
        new_width = int(original_width * scale_ratio)
        new_height = int(original_height * scale_ratio)
        # 缩放图片
        resized_image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
        # 计算填充尺寸
        pad_x = max(0, (target_width - new_width) // 2)
        pad_y = max(0, (target_height - new_height) // 2)
        # 创建目标尺寸的画布并填充指定颜色
        padded_image = np.full((target_height, target_width, 3), pad_color, dtype=np.uint8)
        # 将缩放后的图像放置在中心
        padded_image[pad_y:pad_y + new_height, pad_x:pad_x + new_width] = resized_image
        return padded_image

    # 根据宽高比选择最佳处理方式
    if original_ratio >= target_ratio:
        # 原始图片更宽，适合裁剪
        image = crop_and_resize(image, target_width, target_height)
    else:
        # 原始图片更高，适合填充
        image = pad_and_resize(image, target_width, target_height)
    cv2.imwrite("./tmp/new.png", image)
    print("save_success")
    # Get image dimensions
    height, width, _ = image.shape
    offset_w, offset_h = 0, 0
    frames.append(image)
    return frames, height, width, 30, offset_w, offset_h

def images_crop(images, bboxes, target_size, device=torch.device("cuda")):
    # bboxes: cx, cy, w, h
    crop_img_list = []
    crop_annotations = []
    i = 0
    raw_img_size = max(images[0].shape[:2])
    for img, bbox in zip(images, bboxes):

        left = max(0, int(bbox[0] - bbox[2] // 2))
        right = min(img.shape[1] - 1, int(bbox[0] + bbox[2] // 2))
        top = max(0, int(bbox[1] - bbox[3] // 2))
        bottom = min(img.shape[0] - 1, int(bbox[1] + bbox[3] // 2))
        print(bbox)
        print(left,right,top,bottom)
        crop_img = img[top:bottom, left:right]
        crop_img = torch.Tensor(crop_img).to(device).unsqueeze(0).permute(0, 3, 1, 2)

        _, _, h, w = crop_img.shape
        scale_factor = min(target_size / w, target_size / h)
        crop_img = F.interpolate(crop_img, scale_factor=scale_factor, mode="bilinear")

        _, _, h, w = crop_img.shape
        pad_left = (target_size - w) // 2
        pad_top = (target_size - h) // 2
        pad_right = target_size - w - pad_left
        pad_bottom = target_size - h - pad_top
        crop_img = F.pad(
            crop_img,
            (pad_left, pad_right, pad_top, pad_bottom),
            mode="constant",
            value=0,
        )

        resize_img = normalize_rgb_tensor(crop_img)

        crop_img_list.append(resize_img)
        crop_annotations.append(
            (
                left,
                top,
                pad_left,
                pad_top,
                scale_factor,
                target_size / scale_factor,
                raw_img_size,
            )
        )

    return crop_img_list, crop_annotations


def preprocess_image(img_pil, ratio=1.85/2.0, resolution=512):
    img = np.array(img_pil) # H,W,C=3 
    print(img.shape)
    # remove background
    if(img.shape[-1]==4):
        print("here")
        img_rembg = img
    else:
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
    return rgb_pil, mask_pil



def extract_smpl_params(out_dict):
    # transl
    transl = out_dict["trans"].view(1, 3)  # (1,3)
    
    # betas
    betas = out_dict["betas"][:, :10]  # (1,10)，只取前10维
    
    # poses: 把旋转矩阵转成axis-angle（3维）
    def rotmat_to_axis_angle(rotmat):
        # rotmat: (..., 3, 3)
        rot = torch.linalg.norm(
            torch.stack([
                rotmat[..., 2, 1] - rotmat[..., 1, 2],
                rotmat[..., 0, 2] - rotmat[..., 2, 0],
                rotmat[..., 1, 0] - rotmat[..., 0, 1]
            ], dim=-1),
            dim=-1, keepdim=True
        )
        angle = torch.acos(((rotmat[...,0,0] + rotmat[...,1,1] + rotmat[...,2,2]) - 1)/2)
        axis = torch.stack([
            rotmat[..., 2,1] - rotmat[..., 1,2],
            rotmat[..., 0,2] - rotmat[..., 2,0],
            rotmat[..., 1,0] - rotmat[..., 0,1]
        ], dim=-1) / (2*torch.sin(angle).unsqueeze(-1) + 1e-8)
        return axis * angle.unsqueeze(-1)

    # 把各部分拼起来
    pose_list = []
    for key in ["global_pose", "partbody_pose", "neck_pose", 
                "head_pose", "jaw_pose", 
                "left_hand_pose", "right_hand_pose"]:
        R = out_dict[key]  # (1, N, 3, 3)
        aa = rotmat_to_axis_angle(R)  # (1, N, 3)
        pose_list.append(aa)
    
    poses = torch.cat(pose_list, dim=1)  # (1,55,3)
    filler = torch.zeros(poses.shape[0], 4, 3, device=poses.device)
    # 插入 eyes 在 body_pose 后面 (index=22,23)，toes 在最后 (index=53,54)
    poses = torch.cat([poses[:, :21], filler, poses[:, 21:], ], dim=1)
    transl = transl + torch.tensor([[0,0,3.0]]).cuda()

    all_verts = [None] * 1
    all_verts[0] = []
    out_dict["smpl_verts"][0][:,2] = out_dict["smpl_verts"][0][:,2] + 3.0
    all_verts[0].append(out_dict["smpl_verts"][0])
    return poses.cuda(), betas.cuda(), transl.cuda(),all_verts

def load_smplx_json(json_path, smplx_model_path="./pretrained_models/human_model_files", device="cuda"):
    # ---- 读取 json ----
    with open(json_path, "r") as f:
        data = json.load(f)

    # ---- 转 tensor ----
    betas = torch.tensor(data["betas"], dtype=torch.float32).unsqueeze(0).to(device)    # (1,10)
    transl = torch.tensor(data["trans"], dtype=torch.float32).unsqueeze(0).to(device)   # (1,3)

    root_pose = torch.tensor(data["root_pose"], dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)   # (1,3)
    body_pose = torch.tensor(data["body_pose"], dtype=torch.float32).unsqueeze(0).to(device)   # (1,21,3)
    jaw_pose  = torch.tensor(data["jaw_pose"],  dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)   # (1,3)
    leye_pose = torch.tensor(data["leye_pose"], dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)   # (1,3)
    reye_pose = torch.tensor(data["reye_pose"], dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)   # (1,3)
    lhand_pose = torch.tensor(data["lhand_pose"], dtype=torch.float32).unsqueeze(0).to(device) # (1,15,3)
    rhand_pose = torch.tensor(data["rhand_pose"], dtype=torch.float32).unsqueeze(0).to(device) # (1,15,3)
    print(root_pose.shape)
    print(body_pose.shape)
    print(jaw_pose.shape)
    print(leye_pose.shape)
    print(lhand_pose.shape)
    # ---- 拼接成 poses (1,55,3) ----
    poses = torch.cat([
        root_pose,         # 1
        body_pose,         # 21
        jaw_pose,          # 1
        leye_pose,         # 1
        reye_pose,         # 1
        lhand_pose,        # 15
        rhand_pose         # 15
    ], dim=1)  # (1,55,3)

    # ---- reshape 为 (1,55*3) if 需要 ----
    poses_flat = poses.reshape(1, -1)  # (1,165)

    # ---- SMPL-X 模型 ----
    model = smplx.create(smplx_model_path, model_type="smplx",
                         gender="neutral", use_pca=False, num_betas=10).to(device)

    output = model(
        betas=betas,
        transl=transl,
        global_orient=root_pose,
        body_pose=body_pose,
        jaw_pose=jaw_pose,
        leye_pose=leye_pose,
        reye_pose=reye_pose,
        left_hand_pose=lhand_pose,
        right_hand_pose=rhand_pose,
    )

    vertices = output.vertices  # (1, 10475, 3)
    all_verts = [None] * 1
    all_verts[0] = []
    all_verts[0].append(vertices[0])

    return poses, betas, transl, all_verts

def generate_pseudo_idx(keypoints, patch_size, n_patch, crop_annotation):

    device = keypoints.device
    anchors = torch.stack([keypoints[3], keypoints[4], keypoints[5], keypoints[6]])

    mask = anchors[..., -1] >= KEYPOINT_THR
    if mask.sum() < 2:

        return None, None
    anchors = anchors[mask, :2]  # N, 2

    radius = torch.norm(anchors.max(dim=0)[0] - anchors.min(dim=0)[0]) / 2

    head_pseudo_loc = anchors.mean(0)
    if crop_annotation is not None:
        left, top, pad_left, pad_top, scale_factor, crop_size, raw_size = (
            crop_annotation
        )
        head_pseudo_loc = (
            head_pseudo_loc - torch.tensor([left, top], device=device)
        ) * scale_factor + torch.tensor([pad_left, pad_top], device=device)
        radius = radius * scale_factor
    coarse_loc = (head_pseudo_loc // patch_size).int()  # (nhv,2)
    pseudo_idx = torch.clamp(coarse_loc, 0, n_patch - 1)  # (nhv,2)
    pseudo_idx = (
        torch.zeros((1,), dtype=torch.int32, device=device),
        pseudo_idx[1:2],
        pseudo_idx[0:1],
        torch.zeros((1,), dtype=torch.int32, device=device),
    )
    max_dist = (radius // patch_size).int()
    if max_dist < 2:
        max_dist = None
    return pseudo_idx, max_dist


def project2origin_img(target_human, crop_annotation):
    if target_human is None:
        return target_human
    left, top, pad_left, pad_top, scale_factor, crop_size, raw_size = crop_annotation
    device = target_human["loc"].device

    target_human["loc"] = (
        target_human["loc"] - torch.tensor([pad_left, pad_top], device=device)
    ) / scale_factor + torch.tensor([left, top], device=device)

    target_human["dist"] = target_human["dist"] / (crop_size / raw_size)
    return target_human


def empty_frame_pad(pose_results):
    if len(pose_results) == 1:
        return pose_results
    all_is_None = True
    for i in range(1, len(pose_results)):
        if pose_results[i] is None and pose_results[i - 1] is not None:
            pose_results[i] = copy.deepcopy(pose_results[i - 1])
        if pose_results[i] is not None:
            all_is_None = False
    if all_is_None:
        return []
    for i in range(len(pose_results) - 2, -1, -1):
        if pose_results[i] is None and pose_results[i + 1] is not None:
            pose_results[i] = copy.deepcopy(pose_results[i + 1])
    return pose_results


def parse_chunks(
    frame_ids,
    pose_results,
    k2d,
    bboxes,
    min_len=10,
):
    """If a track disappear in the middle,
    we separate it to different segments
    """
    data_chunks = []
    
    if isinstance(frame_ids, list):
        frame_ids = np.array(frame_ids)
    step = frame_ids[1:] - frame_ids[:-1]
    step = np.concatenate([[0], step])
    breaks = np.where(step != 1)[0]
    start = 0
    for bk in breaks[1:]:
        f_chunk = frame_ids[start:bk]

        if len(f_chunk) >= min_len:
            data_chunk = {
                "frame_id": f_chunk,
                "keypoints_2d": k2d[start:bk],
                "bbox": bboxes[start:bk],
                "rotvec": [],
                "beta": [],
                "loc": [],
                "dist": [],
            }
            padded_pose_results = empty_frame_pad(pose_results[start:bk])

            for pose_result in padded_pose_results:
                data_chunk["rotvec"].append(pose_result["rotvec"])
                data_chunk["beta"].append(pose_result["shape"])
                data_chunk["loc"].append(pose_result["loc"])
                data_chunk["dist"].append(pose_result["dist"])
            if len(padded_pose_results) > 0:
                data_chunks.append(data_chunk)
        start = bk

    start = breaks[-1]  # last chunk
    bk = len(frame_ids)
    f_chunk = frame_ids[start:bk]

    if len(f_chunk) >= min_len:
        data_chunk = {
            "frame_id": f_chunk,
            "keypoints_2d": k2d[start:bk].clone().detach(),
            "bbox": bboxes[start:bk].clone().detach(),
            "rotvec": [],
            "beta": [],
            "loc": [],
            "dist": [],
        }
        padded_pose_results = empty_frame_pad(pose_results[start:bk])
        for pose_result in padded_pose_results:
            data_chunk["rotvec"].append(pose_result["rotvec"])
            data_chunk["beta"].append(pose_result["shape"])
            data_chunk["loc"].append(pose_result["loc"])
            data_chunk["dist"].append(pose_result["dist"])

        if len(padded_pose_results) > 0:

            data_chunks.append(data_chunk)

    for data_chunk in data_chunks:
        for key in ["rotvec", "beta", "loc", "dist"]:
            try:
                data_chunk[key] = torch.stack(data_chunk[key])
            except:
                print(key)

    return data_chunks
@dataclass
class SMPLXOutput(BaseOutput):
    beta: np.ndarray
    is_full_body: bool
    ratio: float
    msg: str


def normalize_rgb_tensor(img, imgenet_normalization=True):
    img = img / 255.0
    if imgenet_normalization:
        img = (
            img - torch.tensor(IMG_NORM_MEAN, device=img.device).view(1, 3, 1, 1)
        ) / torch.tensor(IMG_NORM_STD, device=img.device).view(1, 3, 1, 1)
    return img


class PoseEstimator:
    def __init__(self, model_path, device="cuda",is_predict=True):
        self.device = torch.device(device)
        self.mhmr_model = load_model(
            os.path.join(model_path, "pose_estimate", "multiHMR_896_L.pt"),
            model_path=model_path,
            device=self.device,
        )
        self.pad_ratio = 0.2
        self.img_size = 896
        self.fov = 60
        if is_predict:
            pose_model_ckpt = os.path.join(model_path, "pose_estimate", "vitpose-h-wholebody.pth")
            self.keypoint_detector = DetectionModel(pose_model_ckpt, device)
            self.smplx_model = SMPL_Layer(
                model_path,
                type="smplx",
                gender="neutral",
                num_betas=10,
                kid=False,
                person_center="head",
            ).to(device)
            self.smplify = TemporalSMPLify(
                smpl=self.smplx_model, device=device, num_steps=[30,50]
            )

    def track(self, all_frames):
        print(all_frames[0].shape)
        w = all_frames[0].shape[1]
        h = all_frames[0].shape[0] 
        bboxes = np.array([[0, 0, w, h]])
        frame_ids = [0]
        frames = all_frames
        return bboxes, frame_ids, frames
    def detect_keypoint2d(self, bboxes, frames):
        keypoints, bboxes = self.keypoint_detector.batch_detection(bboxes, frames)

        return bboxes, keypoints

    def to(self, device):
        self.device = device
        self.mhmr_model.to(device)
        return self

    def get_camera_parameters(self):
        K = torch.eye(3)
        # Get focal length.
        focal = self.img_size / (2 * np.tan(np.radians(self.fov) / 2))
        K[0, 0], K[1, 1] = focal, focal

        K[0, -1], K[1, -1] = self.img_size // 2, self.img_size // 2

        # Add batch dimension
        K = K.unsqueeze(0).to(self.device)
        return K

    def img_center_padding(self, img_np):

        ori_h, ori_w = img_np.shape[:2]

        w = round((1 + self.pad_ratio) * ori_w)
        h = round((1 + self.pad_ratio) * ori_h)

        img_pad_np = np.zeros((h, w, 3), dtype=np.uint8)
        offset_h, offset_w = (h - img_np.shape[0]) // 2, (w - img_np.shape[1]) // 2
        img_pad_np[
            offset_h : offset_h + img_np.shape[0] :,
            offset_w : offset_w + img_np.shape[1],
        ] = img_np

        return img_pad_np, offset_w, offset_h

    def _preprocess(self, img_np):

        raw_img_size = max(img_np.shape[:2])

        img_tensor = (
            torch.Tensor(img_np).to(self.device).unsqueeze(0).permute(0, 3, 1, 2)
        )

        _, _, h, w = img_tensor.shape
        scale_factor = min(self.img_size / w, self.img_size / h)
        img_tensor = F.interpolate(
            img_tensor, scale_factor=scale_factor, mode="bilinear"
        )

        _, _, h, w = img_tensor.shape
        pad_left = (self.img_size - w) // 2
        pad_top = (self.img_size - h) // 2
        pad_right = self.img_size - w - pad_left
        pad_bottom = self.img_size - h - pad_top
        img_tensor = F.pad(
            img_tensor,
            (pad_left, pad_right, pad_top, pad_bottom),
            mode="constant",
            value=0,
        )

        resize_img = normalize_rgb_tensor(img_tensor)


        # same as engine/pose_estimation/pose_estimator
        left = 0
        top = 0
        annotation = (
            left,
            top,
            pad_left,
            pad_top,
            scale_factor,
            self.img_size / scale_factor,
            raw_img_size,
        )

        return resize_img, annotation

    @torch.no_grad()
    def __call__(self, img_path):
        # image_tensor H W C

        img_np = np.asarray(Image.open(img_path).convert("RGB"))

        raw_h, raw_w, _ = img_np.shape

        # pad image for more accurate pose estimation
        img_np, offset_w, offset_h = self.img_center_padding(img_np)
        img_tensor, annotation = self._preprocess(img_np)
        K = self.get_camera_parameters()

        with torch.cuda.amp.autocast(enabled=True):
            target_human = self.mhmr_model(
                img_tensor,
                is_training=False,
                nms_kernel_size=int(3),
                det_thresh=0.3,
                K=K,
                idx=None,
                max_dist=None,
            )
        if not len(target_human) == 1:
            return SMPLXOutput(
                beta=None,
                is_full_body=False,
                msg=(
                    "more than one human detected"
                    if len(target_human) > 1
                    else "no human detected"
                ),
            )

        # check is full body
        left, top, pad_left, pad_top, scale_factor, _, _ = annotation
        for key, value in target_human[0].items():
            print(f"Key: {key}, Value: {value.shape}")
        j2d = target_human[0]["j2d"]
        # tranform to raw image space
        j2d = (
            j2d - torch.tensor([pad_left, pad_top], device=self.device).unsqueeze(0)
        ) / scale_factor
        j2d = j2d - torch.tensor([offset_w, offset_h], device=self.device).unsqueeze(0)

        # scale ratio
        top = j2d[..., 1].min()
        bottom = j2d[..., 1].max()
        full_body_length = bottom - top
        visible_body_length = min(raw_h, bottom) - max(0, top)
        visible_ratio = visible_body_length / full_body_length
        is_full_body = visible_ratio.cpu().item() >= 0.4  # suppose (upper / the lenght of body = 0.4,  4: 6)

        return SMPLXOutput(
            beta=target_human[0]["shape"].cpu().numpy(),
            is_full_body=is_full_body,
            ratio=visible_ratio.cpu().item(),
            msg="success" if is_full_body else "no full-body human detected",
        )

    def estimate_pose(self, frame_ids, frames, keypoints, bboxes, raw_K, video_length):
        target_img_size = self.mhmr_model.img_size
        patch_size = self.mhmr_model.patch_size
        # print(target_img_size)
        K = get_camera_parameters(
            target_img_size, fov=self.fov, p_x=None, p_y=None, device=self.device
        )

        keypoints = torch.tensor(keypoints, device=self.device)
        bboxes = torch.tensor(bboxes, device=self.device)
        bboxes = bbox_xyxy_to_cxcywh(bboxes, scale=1.0)

        crop_images, crop_annotations = images_crop(
            frames, bboxes, target_size=target_img_size, device=self.device
        )

        all_frame_results = []
        # model inference
        for i, image in enumerate(crop_images):
            print("image shape", image.shape)
            from torchvision import transforms
            # nimage = transforms.ToPILImage()(image.squeeze(0))

            # nimage.save("output_image.jpg")            
            # Calculate the possible search area for the primary joint (head) based on 2D keypoints
            # pseudo_idx: The index of the search area center after patching
            # max_dist: The maximum radius of the search area
            pseudo_idx, max_dist = generate_pseudo_idx(
                keypoints[i],
                patch_size,
                int(target_img_size / patch_size),
                crop_annotations[i],
            )
            humans = forward_model(
                self.mhmr_model, image, K, pseudo_idx=pseudo_idx, max_dist=max_dist
            )
  
            target_human = track_by_area(humans, target_img_size)
            target_human = project2origin_img(target_human, crop_annotations[i])
            all_frame_results.append(target_human)

        # parse chunk & missed frame padding
        # data_chunks = parse_chunks(
        #     frame_ids,
        #     all_frame_results,
        #     keypoints,
        #     bboxes,
        #     min_len=int(self.fps / 10),
        # )
        data_chunks = parse_chunks(
            frame_ids,
            all_frame_results,
            keypoints,
            bboxes,
            min_len=int(self.fps / 10),
        )

        trans_cam_fill = np.zeros((video_length, 3))
        smpl_poses_cam_fill = np.zeros((video_length, 55, 3))
        smpl_shapes_fill = np.zeros((video_length, 10))
        all_verts = [None] * video_length
        for data_chunk in data_chunks:
            # one_euro filter on 2d keypoints
            # print(data_chunk)
            one_euro = OneEuroFilter(
                min_cutoff=1.2, beta=0.3, sampling_rate=self.fps, device=self.device
            )
            for i in range(len(data_chunk["keypoints_2d"])):
                data_chunk["keypoints_2d"][i, :, :2] = one_euro.filter(
                    data_chunk["keypoints_2d"][i, :, :2]
                )

            poses, betas, transl = self.smplify.fit(
                data_chunk["rotvec"],
                data_chunk["beta"],
                data_chunk["dist"],
                data_chunk["loc"],
                raw_K,
                data_chunk["keypoints_2d"],
                data_chunk["bbox"],
            )

            # gaussian filter
            with torch.no_grad():

                poses, betas, transl = smplx_gs_smooth(
                    poses, betas, transl, fps=self.fps
                )

                out = self.smplx_model(
                    poses,
                    betas,
                    None,
                    None,
                    transl=transl,
                    K=raw_K,
                    expression=None,
                    rot6d=False,
                )

                transl = out["transl_pelvis"].squeeze(1)
                poses_ = self.smplx_model.convert_standard_pose(poses)
                smpl_poses_cam_fill[data_chunk["frame_id"]] = poses_.cpu().numpy()
                smpl_shapes_fill[data_chunk["frame_id"]] = betas.cpu().numpy()
                trans_cam_fill[data_chunk["frame_id"]] = transl.cpu().numpy()

            for i, frame_id in enumerate(data_chunk["frame_id"]):
                try:
                    if all_verts[frame_id] is None:
                        all_verts[frame_id] = []
                    all_verts[frame_id].append(out["v3d"][i])
                except:
                    break

        return smpl_poses_cam_fill, smpl_shapes_fill, trans_cam_fill, all_verts


    def save_results(self, out_path, frame_ids, poses, betas, transl, K, img_wh):
        K = K[0].cpu().numpy()
        for i in frame_ids:
            smplx_param = {}
            smplx_param["betas"] = betas[i].tolist()
            smplx_param["root_pose"] = poses[i, 0].tolist()
            smplx_param["body_pose"] = poses[i, 1:22].tolist()
            smplx_param["jaw_pose"] = poses[i, 22].tolist()
            smplx_param["leye_pose"] = [0.0, 0.0, 0.0]
            smplx_param["reye_pose"] = [0.0, 0.0, 0.0]
            smplx_param["lhand_pose"] = poses[i, 25:40].tolist()
            smplx_param["rhand_pose"] = poses[i, 40:55].tolist()
            smplx_param["trans"] = transl[i].tolist()
            smplx_param["focal"] = [float(K[0, 0]), float(K[1, 1])]
            smplx_param["princpt"] = [float(K[0, 2]), float(K[1, 2])]
            smplx_param["img_size_wh"] = [img_wh[0], img_wh[1]]
            smplx_param["pad_ratio"] = self.pad_ratio
            with open(os.path.join(out_path, f"{(i+1):05}.json"), "w") as fp:
                json.dump(smplx_param, fp)

        for i in frame_ids:
            smplx_param = {}
            smplx_param["betas"] = betas[i]  # 假设 betas[i] 已经是 NumPy 数组
            smplx_param["root_pose"] = poses[i, 0]  # 假设 poses 是 NumPy 数组
            smplx_param["body_pose"] = poses[i, 1:22]
            smplx_param["jaw_pose"] = poses[i, 22]
            smplx_param["leye_pose"] = np.zeros(3)  # 用 np.zeros 替代手动构造
            smplx_param["reye_pose"] = np.zeros(3)
            smplx_param["lhand_pose"] = poses[i, 25:40]
            smplx_param["rhand_pose"] = poses[i, 40:55]
            smplx_param["trans"] = transl[i]  # 假设 transl[i] 是 NumPy 数组
            smplx_param["focal"] = np.array([K[0, 0], K[1, 1]])  # 如果 K 是 NumPy 数组，可以直接切片
            smplx_param["princpt"] = np.array([K[0, 2], K[1, 2]])
            smplx_param["img_size_wh"] = img_wh  # 假设 img_wh 是 NumPy 数组
            smplx_param["pad_ratio"] = self.pad_ratio  # 假设 self.pad_ratio 是 NumPy 数组
        return smplx_param


    def save_pic(
        self, all_frames, frame_ids, bboxes, keypoints, verts, K, out_folder_image,out_folder_smpl,gt_mesh_path=""
    ):
        all_frames = [cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) for frame in all_frames]
        # get_mesh(
        #     verts,
        #     self.mhmr_model.smpl_layer["neutral_10"].bm_x.faces,
        #     K,
        #     all_frames,
        #     2,
        #     out_folder_image,
        #     out_folder_smpl,
        #     self.device,
        #     True,
        #     gt_mesh_path,
        # )
        print(out_folder_image)
        print(out_folder_smpl)
        self.data_dir,self.R,self.T,self.K,self.visible_verts,self.invisible_verts= render_video2(            
            verts,
            self.mhmr_model.smpl_layer["neutral_10"].bm_x.faces,
            K,
            all_frames,
            2,
            out_folder_image,
            out_folder_smpl,
            self.device,
            True,
            gt_mesh_path,
        )

    def predict_smpl(self, img_path):
        # image_tensor H W C

        all_frames, raw_H, raw_W, fps, offset_w, offset_h = load_image(
            img_path
        )

        self.fps = 1
        self.fov = 60
        video_length = len(all_frames)

        raw_K = get_camera_parameters(
            max(raw_H, raw_W), fov=self.fov, p_x=None, p_y=None, device=self.device
        )
        raw_K[..., 0, -1] = raw_W / 2
        raw_K[..., 1, -1] = raw_H / 2
        
        bboxes, frame_ids, frames = self.track(all_frames)
        print("frames",frames)
        print(bboxes)
        bboxes, keypoints = self.detect_keypoint2d(bboxes, frames)
        gc.collect()
        torch.cuda.empty_cache()

        poses, betas, transl, verts = self.estimate_pose(
            frame_ids, frames, keypoints, bboxes, raw_K, video_length
        )
        print("poses",poses.shape) #(1,55,3)
        print("betas",betas.shape) #(1,10)
        print("transl",transl.shape) #(1,3)
        print("verts",len(verts)) 
        print(verts)
        smplx_output_folder = "./tmp"
        smplx_params = self.save_results(
            smplx_output_folder, frame_ids, poses, betas, transl, raw_K, (raw_W, raw_H)
        )

        output_folder = "./tmp/test"
        smplimages_output_folder = os.path.join(output_folder, "smplimages.mp4")
        images_output_folder = output_folder
        os.makedirs(smplx_output_folder, exist_ok=True)

        self.save_pic(
                all_frames, frame_ids, bboxes, keypoints, verts, raw_K, images_output_folder, smplimages_output_folder
            )
        
        return smplx_params


    def predict_smpl_orth(self, img_path):
        # image_tensor H W C

        all_frames, raw_H, raw_W, fps, offset_w, offset_h = load_image(
            img_path
        )

        self.fps = 1
        self.fov = 60
        video_length = len(all_frames)

        raw_K = get_camera_parameters(
            max(raw_H, raw_W), fov=self.fov, p_x=None, p_y=None, device=self.device
        )
        raw_K[..., 0, -1] = raw_W / 2
        raw_K[..., 1, -1] = raw_H / 2

        bboxes, frame_ids, frames = self.track(all_frames)
        print(bboxes)
        bboxes, keypoints = self.detect_keypoint2d(bboxes, frames)
        gc.collect()
        torch.cuda.empty_cache()


        poses, betas, transl, verts = self.estimate_pose_orth(
            frame_ids, frames, keypoints, bboxes, raw_K, video_length
        )


        smplx_output_folder = "./tmp/test"
        smplx_params = self.save_results(
            smplx_output_folder, frame_ids, poses, betas, transl, raw_K, (raw_W, raw_H)
        )

        output_folder = "./tmp/test"
        smplimages_output_folder = os.path.join(output_folder, "smplimages.mp4")
        images_output_folder = output_folder
        os.makedirs(smplx_output_folder, exist_ok=True)

        self.save_pic(
                all_frames, frame_ids, bboxes, keypoints, verts, raw_K, images_output_folder, smplimages_output_folder
            )
        
        return smplx_params


    def predict_smpl_orth2(self, img_path):
        # image_tensor H W C

        all_frames, raw_H, raw_W, fps, offset_w, offset_h = load_image(
            img_path
        )

        self.fps = 1
        self.fov = 60
        video_length = len(all_frames)

        raw_K = get_camera_parameters(
            max(raw_H, raw_W), fov=self.fov, p_x=None, p_y=None, device=self.device
        )
        raw_K[..., 0, -1] = raw_W / 2
        raw_K[..., 1, -1] = raw_H / 2

        bboxes, frame_ids, frames = self.track(all_frames)
        print(bboxes)
        bboxes, keypoints = self.detect_keypoint2d(bboxes, frames)
        gc.collect()
        torch.cuda.empty_cache()

        with torch.no_grad():
            poses, betas, transl, verts = self.estimate_pose_orth(
                frame_ids, frames, keypoints, bboxes, raw_K, video_length
            )


        smplx_output_folder = "./tmp/test"
        smplx_params = self.save_results(
            smplx_output_folder, frame_ids, poses, betas, transl, raw_K, (raw_W, raw_H)
        )

        output_folder = "./tmp/test"
        smplimages_output_folder = os.path.join(output_folder, "smplimages.mp4")
        images_output_folder = output_folder
        os.makedirs(smplx_output_folder, exist_ok=True)

        self.save_pic(
                all_frames, frame_ids, bboxes, keypoints, verts, raw_K, images_output_folder, smplimages_output_folder
            )
        
        return smplx_params





