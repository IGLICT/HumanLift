import os.path as osp
import argparse

import numpy as np
import torch
import cv2
import os, shutil
import argparse
import torch.nn as nn
from matplotlib import pyplot as plt
from tqdm import tqdm
import trimesh
# from obj import Mesh
from imageio_ffmpeg import write_frames
from PIL import Image
from .renderer import render_smpl_vcano_map, render_smpl_vcano_map_orth, render_smpl_vcano_map_semantic_orth

def render_smpl(smpl_path, camera_path, smpl_name, output_dir):
    if not os.path.exists(camera_path):
        return
    if not os.path.exists(smpl_path):
        return
    if os.path.exists(os.path.join(output_dir, smpl_name + ".mp4")):
        return
    smpl = trimesh.load(smpl_path)

    video_writer = write_frames(os.path.join(output_dir, smpl_name + ".mp4"), (1024, 1024), fps=30.0, macro_block_size=2)
    video_writer.send(None)
    
    cam_file = open(camera_path, "r")
    for cam_idx in range(90):
        _ = cam_file.readline()
        intrinsic = torch.eye(3)
        for i in range(3):
            a, b, c = cam_file.readline().split(' ')
            a, b, c = float(a), float(b), float(c)
            intrinsic[i, 0] = a
            intrinsic[i, 1] = b
            intrinsic[i, 2] = c
        extr = torch.eye(4)
        for i in range(3):
            a, b, c, d = cam_file.readline().split(' ')
            a, b, c, d = float(a), float(b), float(c), float(d)
            extr[i, 0] = a
            extr[i, 1] = b
            extr[i, 2] = c
            extr[i, 3] = d
        render_map = render_smpl_vcano_map(smpl.vertices, smpl.faces, cam_R=extr[:3, :3], cam_t=extr[:3, 3:], cam_K=intrinsic, img_w=1024, img_h=1024, device=torch.device("cuda:0"))
        render_map = ((render_map.detach().cpu().numpy()[0, :, :, :3])*255).astype(np.uint8)
        render_map = np.ascontiguousarray(render_map)
        video_writer.send(render_map)
    video_writer.close()

def render_smpl_orth(smpl_path, camera_path, smpl_name, output_dir):
    if not os.path.exists(smpl_path):
        return
    if os.path.exists(os.path.join(output_dir, smpl_name + ".mp4")):
        return
    smpl = trimesh.load(smpl_path)
    smpl.vertices *= 1.5

    video_writer = write_frames(os.path.join(output_dir, smpl_name + ".mp4"), (1024, 1024), fps=30.0, macro_block_size=2)
    video_writer.send(None)
    
    stride_ci = 22.5
    stride_ti = 10
    for ti in tqdm(range(-10, 40+stride_ti, stride_ti)):
        theta = ti
        for ci in np.arange(0.0, 360.0, stride_ci).tolist():
            rad = ci
            render_map = render_smpl_vcano_map_orth(smpl.vertices, smpl.faces, azim_list = [rad], elev_list = [theta], img_w=1024, img_h=1024, device=torch.device("cuda:0"))
            render_map = ((render_map.detach().cpu().numpy()[0, :, :, :3])*255).astype(np.uint8)
            render_map = np.ascontiguousarray(render_map)
            video_writer.send(render_map)
    video_writer.close()


def aabb(ref_v):
    return np.min(ref_v, axis=0), np.max(ref_v, axis=0)

# Unit size
def auto_size(v, ref_v):  # to [-0.5, 0.5]
    vmin, vmax = aabb(ref_v)
    scale = 1.8 / np.max(vmax - vmin)  # Compute scale
    v = v - (vmax + vmin) / 2  # Center mesh on origin
    v_c = (vmax + vmin) / 2
    v = v * scale

    ref_v = (ref_v - (vmax + vmin) / 2) * scale

    return v

def center_image_on_black_background(input_path, output_path):
    """
    将480x480的图片居中放置在480x832的黑色背景中央，并保存结果
    
    参数:
        input_path (str): 输入图片路径（必须是480x480）
        output_path (str): 输出图片保存路径
    """
    # 读取输入图片
    img = cv2.imread(input_path)
    if img is None:
        raise ValueError("无法加载图片，请检查路径是否正确")
    
    # 检查输入图片尺寸是否为480x480
    if img.shape[0] != 480 or img.shape[1] != 480:
        raise ValueError("输入图片尺寸必须是480x480")
    
    # 创建480x832的黑色背景
    background = np.zeros((480, 832, 3), dtype=np.uint8)
    
    # 计算居中位置 (y1:y2, x1:x2)
    y_offset = 0  # 高度相同，不需要垂直偏移
    x_offset = (832 - 480) // 2  # 水平居中偏移量
    
    # 将原图放入背景中央
    background[y_offset:y_offset+480, x_offset:x_offset+480] = img
    
    # 保存结果
    cv2.imwrite(output_path, background)
    # print(f"图片已保存至: {output_path}")

def extract_center_832x480(input_path, output_path):
    """
    从832x832的图片中提取中间的832x480区域并保存
    
    参数:
        input_path (str): 输入图片路径（必须是832x832）
        output_path (str): 输出图片保存路径
    """
    # 读取输入图片
    img = cv2.imread(input_path)
    if img is None:
        raise ValueError("无法加载图片，请检查路径是否正确")
    
    # 检查输入图片尺寸是否为832x832
    if img.shape[0] != 832 or img.shape[1] != 832:
        raise ValueError("输入图片尺寸必须是832x832")
    
    # 计算要提取的区域 (y1:y2, x1:x2)
    height = 480
    y_start = (832 - height) // 2  # 计算垂直方向的起始位置
    
    # 提取中间832x480的区域
    center_region = img[0:832, y_start:y_start+height]
    
    # 保存结果
    cv2.imwrite(output_path, center_region)
    # print(f"中心区域已提取并保存至: {output_path}")

import cv2
import numpy as np

# def process_first_frame(rgb_path, mask_path):
#     rgb_img = cv2.imread(rgb_path)
#     mask_img = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    # 计算 mask 的 bbox
#     ret, binary_mask = cv2.threshold(mask_img, 127, 255, cv2.THRESH_BINARY)
#     x, y, w, h = cv2.boundingRect(binary_mask)

#     # 计算缩放比例
#     min_pad_h = 0.05 * 480  # 24px
#     min_pad_w = 0.05 * 832  # 42px
#     scale_h = (480 - 2 * min_pad_h) / h
#     scale_w = (832 - 2 * min_pad_w) / w
#     scale = min(scale_h, scale_w)

#     # 计算缩放后的尺寸
#     new_w, new_h = int(w * scale), int(h * scale)

#     # 计算偏移量（居中）
#     offset_x = (832 - new_w) // 2
#     offset_y = (480 - new_h) // 2

#     # 裁剪 + resize
#     cropped = rgb_img[y:y+h, x:x+w]
#     resized = cv2.resize(cropped, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)

#     # 放入画布
#     canvas = np.zeros((480, 832, 3), dtype=np.uint8)
#     canvas[offset_y:offset_y+new_h, offset_x:offset_x+new_w] = resized

#     # 返回结果和参数
#     scale_params = {
#         "scale": scale,
#         "offset_x": offset_x,
#         "offset_y": offset_y,
#         "crop_x": x,
#         "crop_y": y,
#         "crop_w": w,
#         "crop_h": h,
#     }
#     return canvas, scale_params

# def process_other_frame(rgb_path, scale_params):
#     rgb_img = cv2.imread(rgb_path)
#     # 直接使用第一帧的参数
#     x, y, w, h = scale_params["crop_x"], scale_params["crop_y"], scale_params["crop_w"], scale_params["crop_h"]
#     new_w, new_h = int(w * scale_params["scale"]), int(h * scale_params["scale"])
#     offset_x, offset_y = scale_params["offset_x"], scale_params["offset_y"]

#     # 裁剪 + resize
#     cropped = rgb_img[y:y+h, x:x+w]
#     resized = cv2.resize(cropped, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)

#     # 放入画布
#     canvas = np.zeros((480, 832, 3), dtype=np.uint8)
#     canvas[offset_y:offset_y+new_h, offset_x:offset_x+new_w] = resized
#     return canvas

import cv2
import numpy as np

def get_mask_bboxes(mask_paths):
    """
    计算所有 mask 的边界框 (x, y, w, h)
    
    参数:
        mask_paths: list - mask 图像路径列表
        
    返回:
        list - 每个 mask 的 (x, y, w, h) 列表
        tuple - 所有 mask 的最大公共边界框 (x_min, y_min, w_max, h_max)
    """
    bboxes = []
    x_min, y_min = float('inf'), float('inf')
    x_max, y_max = 0, 0
    
    for mask_path in mask_paths:
        # 读取 mask（灰度图）
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        
        # 二值化处理
        _, binary_mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
        
        # 计算当前 mask 的边界框
        x, y, w, h = cv2.boundingRect(binary_mask)
        bboxes.append((x, y, w, h))
        
        # 更新最大公共边界框
        x_min = min(x_min, x)
        y_min = min(y_min, y)
        x_max = max(x_max, x + w)
        y_max = max(y_max, y + h)
    
    # 计算最大公共边界框的宽高
    w_max = x_max - x_min
    h_max = y_max - y_min
    
    return bboxes, x_min, y_min, w_max, h_max


def process_first_frame(rgb_path, mask_path):
    # 读取图片
    rgb_img = cv2.imread(rgb_path)
    bboxes, x, y, w, h =get_mask_bboxes(mask_path)
    # print(bboxes, x, y, w, h)


    # 目标画布尺寸 (height=832, width=480)
    target_height = 832
    target_width = 480

    # 计算缩放比例 (保证边距至少 5%)
    min_pad_h = 0.05 * target_height  # 832 * 0.05 = 41.6px
    min_pad_w = 0.05 * target_width   # 480 * 0.05 = 24px
    scale_h = (target_height - 2 * min_pad_h) / h  # 可用高度 / 原高度
    scale_w = (target_width - 2 * min_pad_w) / w   # 可用宽度 / 原宽度
    scale = min(scale_h, scale_w)  # 取较小值，确保两个方向都满足

    # 计算缩放后的尺寸
    new_w = int(w * scale)
    new_h = int(h * scale)

    # 计算偏移量（居中）
    offset_x = (target_width - new_w) // 2  # 水平居中 (width方向)
    offset_y = (target_height - new_h) // 2 # 垂直居中 (height方向)

    # 裁剪原图 + resize
    cropped = rgb_img[y:y+h, x:x+w]
    resized = cv2.resize(cropped, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)

    # 创建目标画布 (height=832, width=480)
    canvas = np.zeros((target_height, target_width, 3), dtype=np.uint8)

    # 计算实际可写入的区域（防止超出画布边界）
    canvas_h, canvas_w = canvas.shape[:2]
    y_start = max(0, offset_y)
    y_end = min(canvas_h, offset_y + new_h)
    x_start = max(0, offset_x)
    x_end = min(canvas_w, offset_x + new_w)
    
    # 计算 resized 对应的区域
    resized_y_start = max(0, -offset_y)
    resized_y_end = resized_y_start + (y_end - y_start)
    resized_x_start = max(0, -offset_x)
    resized_x_end = resized_x_start + (x_end - x_start)
    
    # 安全写入（自动裁剪超出部分）
    canvas[y_start:y_end, x_start:x_end] = resized[resized_y_start:resized_y_end, resized_x_start:resized_x_end]
    # canvas[offset_y:offset_y+new_h, offset_x:offset_x+new_w] = resized

    # 返回结果和参数
    scale_params = {
        "scale": scale,
        "offset_x": offset_x,
        "offset_y": offset_y,
        "crop_x": x,
        "crop_y": y,
        "crop_w": w,
        "crop_h": h,
    }
    return canvas, scale_params

def process_other_frame(rgb_path, mask_path, scale_params):
    # 读取图片
    rgb_img = cv2.imread(rgb_path)

    # mask_img = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    
    # # 计算 mask 的 bbox
    # ret, binary_mask = cv2.threshold(mask_img, 127, 255, cv2.THRESH_BINARY)
    # x, y, w, h = cv2.boundingRect(binary_mask)

    # 使用第一帧的参数
    x, y, w, h = scale_params["crop_x"], scale_params["crop_y"], scale_params["crop_w"], scale_params["crop_h"]
    new_w = int(w * scale_params["scale"])
    new_h = int(h * scale_params["scale"])
    offset_x = scale_params["offset_x"]
    offset_y = scale_params["offset_y"]

    # 裁剪 + resize
    cropped = rgb_img[y:y+h, x:x+w]
    resized = cv2.resize(cropped, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)

    # 创建目标画布 (height=832, width=480)
    canvas = np.zeros((832, 480, 3), dtype=np.uint8)
    
    # 边界检查
    canvas_h, canvas_w = canvas.shape[:2]
    y_start = max(0, offset_y)
    y_end = min(canvas_h, offset_y + new_h)
    x_start = max(0, offset_x)
    x_end = min(canvas_w, offset_x + new_w)
    
    resized_y_start = max(0, -offset_y)
    resized_y_end = resized_y_start + (y_end - y_start)
    resized_x_start = max(0, -offset_x)
    resized_x_end = resized_x_start + (x_end - x_start)
    
    # 安全写入
    canvas[y_start:y_end, x_start:x_end] = resized[resized_y_start:resized_y_end, resized_x_start:resized_x_end]

    # canvas[offset_y:offset_y+new_h, offset_x:offset_x+new_w] = resized
    return canvas

def process_sementic_imgs(rgb_paths, mask_paths):
    """
    处理 RGB 和 Mask 图像序列，覆盖保存到原路径
    参数:
        rgb_paths: list - RGB 图像路径列表
        mask_paths: list - Mask 图像路径列表
    返回:
        None (直接覆盖保存)
    """
    # 处理第一帧
    first_frame_result, scale_params = process_first_frame(rgb_paths[0], mask_paths)
    cv2.imwrite(rgb_paths[0], first_frame_result)

    # 处理后续帧
    for i in range(1, len(rgb_paths)):
        # if i == len(rgb_paths) - 1:
        #     mask_paths=mask_paths[0]
        # else:
        #     mask_paths=mask_paths[i]
        result_img = process_other_frame(rgb_paths[i], mask_paths, scale_params)
        cv2.imwrite(rgb_paths[i], result_img)


def generate_circular_view_sequence(num_points=81):
    """
    生成仰角(elev)和方位角(azim)的圆形轨迹
    
    参数:
        num_points: 总点数 (默认81)
        
    返回:
        tuple: (elev_array, azim_array) 角度数组
    """
    # 生成圆形参数 (0到2π)
    theta = np.linspace(0, 2*np.pi, num_points, endpoint=False)
    
    # 圆形参数 (在azim-elev参数空间中)
    radius = 30  # 圆形半径(角度)
    center_azim = 0  # 圆心方位角
    center_elev = 0  # 圆心仰角
    
    # 生成圆形轨迹
    azim = center_azim + radius * np.cos(theta)
    elev = center_elev + radius * np.sin(theta)
    
    return elev, azim

def render_smpl_semantic_orth(smpl_path, camera_path, smpl_name, output_dir,resolution = 480):
    output_dir1 = os.path.join(output_dir, smpl_name, "images")
    output_dir2 = os.path.join(output_dir, smpl_name, "normals")
    output_dir3 = os.path.join(output_dir, smpl_name, "masks")
    if not os.path.exists(output_dir1):
        os.makedirs(output_dir1)
    if not os.path.exists(output_dir2):
        os.makedirs(output_dir2)
    if not os.path.exists(output_dir3):
        os.makedirs(output_dir3)
    
    # output_path1 = os.path.join(output_dir1, smpl_name + ".mp4")
    # output_path2 = os.path.join(output_dir2, smpl_name + ".mp4")
    # output_path3 = os.path.join(output_dir3, smpl_name + ".mp4")
    if len(os.listdir(output_dir1)) > 29 and len(os.listdir(output_dir2)) > 29 and len(os.listdir(output_dir3)) > 29:
        return
    # if not os.path.exists(smpl_path):
    #     print(smpl_path)
    #     return
    # if os.path.exists(os.path.join(output_dir, smpl_name + ".mp4")):
    #     return
    # smpl = trimesh.load(smpl_path)
    # smpl.vertices *= 2.3
    # smpl = Mesh.load_obj(path = smpl_path, ref_path = smpl_path.replace('norm_mtl.obj', 'norm_smpl.obj'))
    smpl_trimesh = trimesh.load(smpl_path)
    # fuze_trimesh.vertices *= 2.3
    # mesh1 = Mesh.load_obj(path = obj_path, ref_path = obj_path.replace('norm_mtl.obj', 'norm_smpl.obj'))
    # norm_trimesh = trimesh.load(smpl_path.replace('norm_smpl.obj', 'norm_smpl.obj'))
    smpl_trimesh.vertices -= np.mean(smpl_trimesh.vertices, axis=0)
    smpl_trimesh.vertices = np.dot(smpl_trimesh.vertices, np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]]))
    smpl_trimesh.vertices = auto_size(smpl_trimesh.vertices, smpl_trimesh.vertices)
    # for _ in range(2):  # Subdivide twice
    #     smpl_trimesh = smpl_trimesh.subdivide()
    smplv = smpl_trimesh.vertices
    smplf = smpl_trimesh.faces

    # print(smplv.shape)
    # print(smplf.shape)



    # video_writer1 = write_frames(os.path.join(output_dir1,  smpl_name + ".mp4"), (1024, 1024), fps=30.0, macro_block_size=2)
    # video_writer1.send(None)

    # video_writer2 = write_frames(os.path.join(output_dir2, smpl_name + ".mp4"), (1024, 1024), fps=30.0, macro_block_size=2)
    # video_writer2.send(None)
    
    # video_writer3 = write_frames(os.path.join(output_dir3, smpl_name + ".mp4"), (1024, 1024), fps=30.0, macro_block_size=2)
    # video_writer3.send(None)

    stride_ci = 360.0/81.0
    stride_ti = 10
    # resolution = 480
    semantic = True
    random_color = None#torch.rand(smplv.shape).to(torch.device("cuda:0"))
    # for ti in tqdm(range(-10, 40+stride_ti, stride_ti)):
    elevs, azims = generate_circular_view_sequence()
    for i in range(1):
        theta = 0
        count = 0
        for ci in tqdm(np.arange(360.0, 0.0, -stride_ci).tolist()):
            rad = ci
            # theta = elevs[count]
            # rad = azims[count]
            render_map1, render_map2 = render_smpl_vcano_map_semantic_orth(smplv, smplf, azim_list = [rad], elev_list = [theta], \
                img_w=resolution, img_h=resolution, device=torch.device("cuda:0"), semantic=semantic, random_color=random_color)
            
            render_semantic = ((render_map2.detach().cpu().numpy()[0, :, :, :3])*255).astype(np.uint8)
            render_normal = ((render_map1.detach().cpu().numpy()[0, :, :, :3])*255).astype(np.uint8)
            render_mask = ((render_map1.detach().cpu().numpy()[0, :, :, 3:])*255).astype(np.uint8)

            if semantic:
                rgb = Image.fromarray(np.array(render_semantic, dtype=np.byte), "RGB")
                rgb.save(os.path.join(output_dir1, "frame_{:02d}.jpg".format(count)))
            rgb = Image.fromarray(np.array(render_normal, dtype=np.byte), "RGB")
            rgb.save(os.path.join(output_dir2, "frame_{:02d}.jpg".format(count)))
            rgb = Image.fromarray(np.array(np.repeat(render_mask, 3, axis=2), dtype=np.byte), "RGB")
            rgb.save(os.path.join(output_dir3, "frame_{:02d}.jpg".format(count)))
            if count==0:
                # cp os.path.join(output_dir1, "frame_{:02d}.jpg".format(count)) to pose.jpg
                shutil.copy(os.path.join(output_dir1, "frame_{:02d}.jpg".format(count)), os.path.join(output_dir1, "pose.jpg"))
                shutil.copy(os.path.join(output_dir2, "frame_{:02d}.jpg".format(count)), os.path.join(output_dir2, "pose.jpg"))
            count += 1
            # video_writer1.send(np.ascontiguousarray(render_semantic))
            # video_writer2.send(np.ascontiguousarray(render_normal))
            # video_writer3.send(np.ascontiguousarray(np.repeat(render_mask, 3, axis=2)))
    if resolution == 480:
        for i in range(count):
            center_image_on_black_background(os.path.join(output_dir1, "frame_{:02d}.jpg".format(i)), os.path.join(output_dir1, "frame_{:02d}.jpg".format(i)))
            center_image_on_black_background(os.path.join(output_dir2, "frame_{:02d}.jpg".format(i)), os.path.join(output_dir2, "frame_{:02d}.jpg".format(i)))
            center_image_on_black_background(os.path.join(output_dir3, "frame_{:02d}.jpg".format(i)), os.path.join(output_dir3, "frame_{:02d}.jpg".format(i)))
        center_image_on_black_background(os.path.join(output_dir1, "pose.jpg"), os.path.join(output_dir1, "pose.jpg"))
        center_image_on_black_background(os.path.join(output_dir2, "pose.jpg"), os.path.join(output_dir2, "pose.jpg"))
    else:
        # cc()
        rgb_paths = sorted([os.path.join(output_dir1, f) for f in os.listdir(output_dir1)])
        mask_paths = sorted([os.path.join(output_dir3, f) for f in os.listdir(output_dir3)])
        process_sementic_imgs(rgb_paths, mask_paths)
        # for i in range(count):
        #     extract_center_832x480(os.path.join(output_dir1, "frame_{:02d}.jpg".format(i)), os.path.join(output_dir1, "frame_{:02d}.jpg".format(i)))
        #     extract_center_832x480(os.path.join(output_dir2, "frame_{:02d}.jpg".format(i)), os.path.join(output_dir2, "frame_{:02d}.jpg".format(i)))
        #     extract_center_832x480(os.path.join(output_dir3, "frame_{:02d}.jpg".format(i)), os.path.join(output_dir3, "frame_{:02d}.jpg".format(i)))
        # extract_center_832x480(os.path.join(output_dir1, "pose.jpg"), os.path.join(output_dir1, "pose.jpg"))
        # extract_center_832x480(os.path.join(output_dir2, "pose.jpg"), os.path.join(output_dir2, "pose.jpg"))
    # video_writer1.close()
    # video_writer2.close()
    # video_writer3.close()

# render_smpl_semantic_orth("~/data/001_smplx.obj", "", "001", "/home/yangjie/data/")
# cc()

# output_dir1 = 'output/test111/4260d4733feb0e9a5817b0b5314b9bb6/temp/smplx1/images'
# output_dir3 = 'output/test111/4260d4733feb0e9a5817b0b5314b9bb6/temp/smplx1/masks'
# rgb_paths = sorted([os.path.join(output_dir1, f) for f in os.listdir(output_dir1)])
# mask_paths = sorted([os.path.join(output_dir3, f) for f in os.listdir(output_dir3)])
# process_sementic_imgs(rgb_paths, mask_paths)
# cc()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--obj-dir", type=str, default="", required=True)
    parser.add_argument("--camera-dir", type=str, default="", required=True)
    parser.add_argument("--output-dir", type=str, default="", required=True)
    args = parser.parse_args()

    obj_dir = args.obj_dir
    output_dir = args.output_dir
    camera_dir = args.camera_dir
    os.makedirs(output_dir, exist_ok=True)
    lines = [line.rstrip() for line in open('used_files.txt')]
    lista = sorted(os.listdir(obj_dir), reverse=False)

    for obj_name in tqdm(lista):
        if obj_name not in lines:
            continue
        print(obj_name)
        render_smpl_semantic_orth(os.path.join(obj_dir, obj_name, "norm_smpl.obj"), os.path.join(camera_dir, obj_name + ".txt"), obj_name, output_dir)
        cc()

