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
from pose_utils.render import render_video
from pose_utils.tracker import bbox_xyxy_to_cxcywh, track_by_area
from smplify import TemporalSMPLify

import trimesh
import numpy as np

torch.cuda.empty_cache()

def load_json(json_path, device=torch.device("cuda"), ref = None):
    with open(json_path, "r") as fp:
        smplx_param = json.load(fp)

    # === 1. 还原 pose（55个关节） ===
    root_pose = torch.tensor(smplx_param["root_pose"], dtype=torch.float32, device=device).reshape(1, 3)
    body_pose = torch.tensor(smplx_param["body_pose"], dtype=torch.float32, device=device).reshape(21, 3)
    jaw_pose = torch.tensor(smplx_param["jaw_pose"], dtype=torch.float32, device=device).reshape(1, 3)
    leye_pose = torch.tensor(smplx_param["leye_pose"], dtype=torch.float32, device=device).reshape(1, 3)
    reye_pose = torch.tensor(smplx_param["reye_pose"], dtype=torch.float32, device=device).reshape(1, 3)
    lhand_pose = torch.tensor(smplx_param["lhand_pose"], dtype=torch.float32, device=device).reshape(15, 3)
    rhand_pose = torch.tensor(smplx_param["rhand_pose"], dtype=torch.float32, device=device).reshape(15, 3)

    pose = torch.cat([
        root_pose,
        body_pose,
        jaw_pose,
        leye_pose,
        reye_pose,
        lhand_pose,
        rhand_pose
    ], dim=0)  # 最终维度应为 (55, 3)

    # === 2. 还原其他参数 ===
    betas = torch.tensor(smplx_param["betas"], dtype=torch.float32, device=device)
    trans = torch.tensor(smplx_param["trans"], dtype=torch.float32, device=device)

    focal = smplx_param["focal"]
    princpt = smplx_param["princpt"]
    K = torch.tensor([
        [focal[0], 0.0, princpt[0]],
        [0.0, focal[1], princpt[1]],
        [0.0, 0.0, 1.0]
    ], dtype=torch.float32, device=device)

    img_wh = smplx_param["img_size_wh"]
    pad_ratio = smplx_param["pad_ratio"]

    return {
        "pose": pose,             # (55, 3)
        "betas": betas,           # (10,)
        "trans": trans,           # (3,)
        "K": K,                   # (3, 3)
        "img_wh": img_wh,         # [W, H]
        "pad_ratio": pad_ratio    # float
    }

def aabb(ref_v):
    return np.min(ref_v, axis=0), np.max(ref_v, axis=0)

def auto_size(v, ref_v): 
    vmin, vmax = aabb(ref_v)
    scale = 1.8*0.54 / np.max(vmax - vmin) 
    v = v - (vmax + vmin) / 2 
    v_c = (vmax + vmin) / 2
    v = v * scale
    ref_v = (ref_v - (vmax + vmin) / 2) * scale
    return v



def load_mesh(smplx_param,smplx_model,output_path, ref_param = None, ref_v = None):
    poses = smplx_param["pose"].unsqueeze(0)

    poses = smplx_model.convert_standard_pose_inverse(poses)

    if(ref_param is not None):
        betas = ref_param["betas"].unsqueeze(0)
        raw_K = ref_param["K"].unsqueeze(0)
        transl = ref_param["trans"].unsqueeze(0)
    else:
        betas = smplx_param["betas"].unsqueeze(0)
        raw_K = smplx_param["K"].unsqueeze(0)
        transl = smplx_param["trans"].unsqueeze(0)
    out = smplx_model.forward_zbt(
                        poses,
                        betas,
                        None,
                        None,
                        transl=transl,
                        K=raw_K,
                        expression=None,
                        rot6d=False,
                    )

    new_vertices = out["v3d"].squeeze(0).detach().cpu().numpy()
    mesh = trimesh.load('./smplx.obj')  # 也可以是 .ply, .off 等
    faces = mesh.faces  # (F, 3)
    old_vertices = mesh.vertices  # (V, 3)



    if(ref_param is not None):
        new_vertices -= np.mean(new_vertices, axis=0)
        new_vertices = np.dot(new_vertices, np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]]))
        new_vertices2 = auto_size(new_vertices, ref_v)
    else:
        new_vertices -= np.mean(new_vertices, axis=0)
        new_vertices = np.dot(new_vertices, np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]]))
        new_vertices2 = auto_size(new_vertices, new_vertices)

    new_mesh = trimesh.Trimesh(vertices=new_vertices2, faces=faces, process=False)
    new_mesh.export(output_path)  # 保存为新的 mesh 文件
    return new_vertices




 



   