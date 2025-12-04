# -*- coding: utf-8 -*-
# @Organization  : Alibaba XR-Lab
# @Author        : Lingteng Qiu  & Xiaodong Gu
# @Email         : 220019047@link.cuhk.edu.cn
# @Time          : 2025-03-1 17:30:37
# @Function      : Inference code for human_lrm model

import argparse
import os
import pdb
import time

import cv2
import numpy as np
import torch
from accelerate.logging import get_logger
from omegaconf import OmegaConf
from PIL import Image, ImageDraw
from tqdm.auto import tqdm
# from engine.pose_estimation.smplify import TemporalSMPLify
from engine.pose_estimation.pose_estimator import PoseEstimator
from engine.SegmentAPI.base import Bbox
from pytorch3d.ops import knn_points
from thirdparties.econ.lib.common.imutils import process_video_fit
# from LHM.utils.model_download_utils import AutoModelQuery
from LHM.utils.model_download_utils import AutoModelQuery

try:
    from engine.SegmentAPI.SAM import SAM2Seg
except:
    print("\033[31mNo SAM2 found! Try using rembg to remove the background. This may slightly degrade the quality of the results!\033[0m")
    from rembg import remove

from LHM.datasets.cam_utils import (
    build_camera_principle,
    build_camera_standard,
    create_intrinsics,
    surrounding_views_linspace,
)
from LHM.models.modeling_human_lrm import ModelHumanLRM
from LHM.runners import REGISTRY_RUNNERS
from LHM.runners.infer.utils import (
    calc_new_tgt_size_by_aspect,
    center_crop_according_to_mask,
    prepare_motion_seqs,
    prepare_motion_seqs_single_seq,
    resize_image_keepaspect_np,
    
)
from LHM.utils.download_utils import download_extract_tar_from_url, download_from_url
from LHM.utils.face_detector import FaceDetector

# from LHM.utils.video import images_to_video
from LHM.utils.ffmpeg_utils import images_to_video
from LHM.utils.hf_hub import wrap_model_hub
from LHM.utils.logging import configure_logger
from LHM.utils.model_card import MODEL_CARD, MODEL_CONFIG

import smplx
import sys
sys.path.append("./thirdparties/econ")
from thirdparties.econ.lib.common.smpl_utils import (
    SMPLEstimator, SMPLRenderer,
    save_optimed_video, save_optimed_smpl_param, save_optimed_mesh,
)
from thirdparties.econ.lib.common.train_util import init_loss




def save_landmarks_pil(gt_lmks, smpl_lmks, out_dir="lmk_pil", frame_idx=0, img_size=(480, 832)):
    """
    用 PIL 分别保存 GT 和 SMPL 的关键点
    gt_lmks: (B, 25, 2)
    smpl_lmks: (B, 25, 2)
    """
    os.makedirs(out_dir, exist_ok=True)

    W, H = img_size
    # 取一帧
    gt = gt_lmks[frame_idx].detach().cpu().numpy()
    smpl = smpl_lmks[frame_idx].detach().cpu().numpy()


    gt_px = gt.copy()
    gt_px[:, 0] = gt_px[:, 0] * W
    gt_px[:, 1] = gt_px[:, 1] * H

    smpl_px = smpl.copy()
    smpl_px[:, 0] = smpl_px[:, 0] * W
    smpl_px[:, 1] = smpl_px[:, 1] * H

    # ---- GT ----
    img_gt = Image.new("RGB", img_size, "white")  # 白底
    draw_gt = ImageDraw.Draw(img_gt)
    for x, y in gt_px:
        r = 3
        draw_gt.ellipse((x-r, y-r, x+r, y+r), fill="blue")
    img_gt.save(os.path.join(out_dir, f"gt_{frame_idx:04d}.png"))

    # ---- SMPL ----
    img_smpl = Image.new("RGB", img_size, "white")
    draw_smpl = ImageDraw.Draw(img_smpl)
    for x, y in smpl_px:
        r = 3
        draw_smpl.ellipse((x-r, y-r, x+r, y+r), fill="red")
    img_smpl.save(os.path.join(out_dir, f"smpl_{frame_idx:04d}.png"))

    print(f"✅ Saved gt_{frame_idx:04d}.png and smpl_{frame_idx:04d}.png")

def download_geo_files():
    if not os.path.exists('./pretrained_models/dense_sample_points/1_20000.ply'):
        download_from_url('https://virutalbuy-public.oss-cn-hangzhou.aliyuncs.com/share/aigc3d/data/LHM/1_20000.ply','./pretrained_models/dense_sample_points/')

def prior_check():
    if not os.path.exists('./pretrained_models'):
        prior_data = MODEL_CARD['prior_model']
        download_extract_tar_from_url(prior_data)


from .base_inferrer import Inferrer

logger = get_logger(__name__)

def forward_model(
    model,
    input_image,
    camera_parameters,
    det_thresh=0.3,
    nms_kernel_size=1,
    pseudo_idx=None,
    max_dist=None,
):
    """Make a forward pass on an input image and camera parameters."""

    # Forward the model.
    with torch.no_grad():
        with torch.cuda.amp.autocast(enabled=True):
            humans = model(
                input_image,
                is_training=False,
                nms_kernel_size=int(nms_kernel_size),
                det_thresh=det_thresh,
                K=camera_parameters,
                idx=pseudo_idx,
                max_dist=max_dist,
            )
    return humans
    
def avaliable_device():
    if torch.cuda.is_available():
        current_device_id = torch.cuda.current_device()
        device = f"cuda:{current_device_id}"
    else:
        device = "cpu"

    return device

def resize_with_padding(img, target_size, padding_color=(255, 255, 255)):
    target_w, target_h = target_size
    h, w = img.shape[:2]

    ratio = min(target_w / w, target_h / h)
    new_w = int(w * ratio)
    new_h = int(h * ratio)

    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

    dw = target_w - new_w
    dh = target_h - new_h
    top = dh // 2
    bottom = dh - top
    left = dw // 2
    right = dw - left

    padded = cv2.copyMakeBorder(
        resized,
        top=top,
        bottom=bottom,
        left=left,
        right=right,
        borderType=cv2.BORDER_CONSTANT,
        value=padding_color,
    )

    return padded


def get_bbox(mask):
    height, width = mask.shape
    pha = mask / 255.0
    pha[pha < 0.5] = 0.0
    pha[pha >= 0.5] = 1.0

    # obtain bbox
    _h, _w = np.where(pha == 1)

    whwh = [
        _w.min().item(),
        _h.min().item(),
        _w.max().item(),
        _h.max().item(),
    ]

    box = Bbox(whwh)
    # scale box to 1.05
    scale_box = box.scale(1.1, width=width, height=height)
    return scale_box

def query_model_name(model_name):
    if model_name in MODEL_PATH:
        model_path = MODEL_PATH[model_name]
        if not os.path.exists(model_path):
            model_url = MODEL_CARD[model_name]
            download_extract_tar_from_url(model_url, './')
    else:
        model_path = model_name
    
    return model_path


def query_model_config(model_name):
    try:
        model_params = model_name.split('-')[1]
        
        return MODEL_CONFIG[model_params] 
    except:
        return None

def infer_preprocess_image(
    rgb_path,
    mask,
    intr,
    pad_ratio,
    bg_color,
    max_tgt_size,
    aspect_standard,
    enlarge_ratio,
    render_tgt_size,
    multiply,
    need_mask=True,
):
    """inferece
    image, _, _ = preprocess_image(image_path, mask_path=None, intr=None, pad_ratio=0, bg_color=1.0,
                                        max_tgt_size=896, aspect_standard=aspect_standard, enlarge_ratio=[1.0, 1.0],
                                        render_tgt_size=source_size, multiply=14, need_mask=True)

    """

    rgb = np.array(Image.open(rgb_path))
    rgb_raw = rgb.copy()

    bbox = get_bbox(mask)
    bbox_list = bbox.get_box()

    rgb = rgb[bbox_list[1] : bbox_list[3], bbox_list[0] : bbox_list[2]]
    mask = mask[bbox_list[1] : bbox_list[3], bbox_list[0] : bbox_list[2]]

    h, w, _ = rgb.shape
    print("h:",h)
    print("w:",w)
    # assert w <= h
    cur_ratio = h / w
    scale_ratio = cur_ratio / aspect_standard


    target_w = int(min(w * scale_ratio, h))
    if target_w - w >0:
        offset_w = (target_w - w) // 2

        rgb = np.pad(
            rgb,
            ((0, 0), (offset_w, offset_w), (0, 0)),
            mode="constant",
            constant_values=255,
        )

        mask = np.pad(
            mask,
            ((0, 0), (offset_w, offset_w)),
            mode="constant",
            constant_values=0,
        )
    else:
        target_h = w * aspect_standard
        offset_h = int(target_h - h)

        rgb = np.pad(
            rgb,
            ((offset_h, 0), (0, 0), (0, 0)),
            mode="constant",
            constant_values=255,
        )

        mask = np.pad(
            mask,
            ((offset_h, 0), (0, 0)),
            mode="constant",
            constant_values=0,
        )

    rgb = rgb / 255.0  # normalize to [0, 1]
    mask = mask / 255.0

    mask = (mask > 0.5).astype(np.float32)
    rgb = rgb[:, :, :3] * mask[:, :, None] + bg_color * (1 - mask[:, :, None])

    # resize to specific size require by preprocessor of smplx-estimator.
    rgb = resize_image_keepaspect_np(rgb, max_tgt_size)
    mask = resize_image_keepaspect_np(mask, max_tgt_size)

    # crop image to enlarge human area.
    rgb, mask, offset_x, offset_y = center_crop_according_to_mask(
        rgb, mask, aspect_standard, enlarge_ratio
    )
    if intr is not None:
        intr[0, 2] -= offset_x
        intr[1, 2] -= offset_y

    # resize to render_tgt_size for training

    tgt_hw_size, ratio_y, ratio_x = calc_new_tgt_size_by_aspect(
        cur_hw=rgb.shape[:2],
        aspect_standard=aspect_standard,
        tgt_size=render_tgt_size,
        multiply=multiply,
    )

    rgb = cv2.resize(
        rgb, dsize=(tgt_hw_size[1], tgt_hw_size[0]), interpolation=cv2.INTER_AREA
    )
    mask = cv2.resize(
        mask, dsize=(tgt_hw_size[1], tgt_hw_size[0]), interpolation=cv2.INTER_AREA
    )

    if intr is not None:

        # ******************** Merge *********************** #
        intr = scale_intrs(intr, ratio_x=ratio_x, ratio_y=ratio_y)
        assert (
            abs(intr[0, 2] * 2 - rgb.shape[1]) < 2.5
        ), f"{intr[0, 2] * 2}, {rgb.shape[1]}"
        assert (
            abs(intr[1, 2] * 2 - rgb.shape[0]) < 2.5
        ), f"{intr[1, 2] * 2}, {rgb.shape[0]}"

        # ******************** Merge *********************** #
        intr[0, 2] = rgb.shape[1] // 2
        intr[1, 2] = rgb.shape[0] // 2

    rgb = torch.from_numpy(rgb).float().permute(2, 0, 1).unsqueeze(0)  # [1, 3, H, W]
    mask = (
        torch.from_numpy(mask[:, :, None]).float().permute(2, 0, 1).unsqueeze(0)
    )  # [1, 1, H, W]
    return rgb, mask, intr


def parse_configs():


    download_geo_files()

    parser = argparse.ArgumentParser()
    parser.add_argument("--infer", type=str)
    args, unknown = parser.parse_known_args()

    cfg = OmegaConf.create()
    cli_cfg = OmegaConf.from_cli(unknown)

    if "export_mesh" not in cli_cfg: 
        cli_cfg.export_mesh = None
    if "export_video" not in cli_cfg: 
        cli_cfg.export_video= None

    query_model = AutoModelQuery()

    # parse from ENV
    if os.environ.get("APP_INFER") is not None:
        args.infer = os.environ.get("APP_INFER")
    if os.environ.get("APP_MODEL_NAME") is not None:
        model_name = query_model_name(os.environ.get("APP_MODEL_NAME"))
        cli_cfg.model_name = os.environ.get("APP_MODEL_NAME")
    else:
        model_name = cli_cfg.model_name
        model_path= query_model.query(model_name) 
        cli_cfg.model_name = model_path 
    
    model_config = query_model_config(model_name)

    if model_config is not None:
        cfg_train = OmegaConf.load(model_config)
        cfg.source_size = cfg_train.dataset.source_image_res
        try:
            cfg.src_head_size = cfg_train.dataset.src_head_size
        except:
            cfg.src_head_size = 112
        cfg.render_size = cfg_train.dataset.render_image.high
        _relative_path = os.path.join(
            cfg_train.experiment.parent,
            cfg_train.experiment.child,
            os.path.basename(cli_cfg.model_name).split("_")[-1],
        )

        cfg.save_tmp_dump = os.path.join("exps", "save_tmp", _relative_path)
        cfg.image_dump = os.path.join("exps", "images", _relative_path)
        cfg.video_dump = os.path.join("exps", "videos", _relative_path)  # output path
        cfg.mesh_dump = os.path.join("exps", "meshs", _relative_path)  # output path

    if args.infer is not None:
        cfg_infer = OmegaConf.load(args.infer)
        cfg.merge_with(cfg_infer)
        cfg.setdefault(
            "save_tmp_dump", os.path.join("exps", cli_cfg.model_name, "save_tmp")
        )
        cfg.setdefault("image_dump", os.path.join("exps", cli_cfg.model_name, "images"))
        cfg.setdefault(
            "video_dump", os.path.join("dumps", cli_cfg.model_name, "videos")
        )
        cfg.setdefault("mesh_dump", os.path.join("dumps", cli_cfg.model_name, "meshes"))

    cfg.motion_video_read_fps = 6
    cfg.merge_with(cli_cfg)

    cfg.setdefault("logger", "INFO")

    assert cfg.model_name is not None, "model_name is required"
    cfg.setdefault("dataset_dir", None)
    cfg.setdefault("only_predict_smpl", None)
    return cfg, cfg_train


@REGISTRY_RUNNERS.register("infer.human_lrm")
class HumanLRMInferrer(Inferrer):

    EXP_TYPE: str = "human_lrm_sapdino_bh_sd3_5"
    # EXP_TYPE: str = "human_lrm_sd3"

    def __init__(self):
        super().__init__()

        self.cfg, cfg_train = parse_configs()

        configure_logger(
            stream_level=self.cfg.logger,
            log_level=self.cfg.logger,
        )  # logger function

        # if do not download prior model, we automatically download them.
        prior_check()

        self.facedetect = FaceDetector(
            "./pretrained_models/gagatracker/vgghead/vgg_heads_l.trcd",
            device=avaliable_device(),
        )
        self.pose_estimator = PoseEstimator(
            "./pretrained_models/human_model_files/", device=avaliable_device()
        )


        try:
            self.parsingnet = SAM2Seg()
        except:
            self.parsingnet = None 

        self.model: ModelHumanLRM = self._build_model(self.cfg).to(self.device)

        self.motion_dict = dict()

    def _build_model(self, cfg):
        from LHM.models import model_dict

        hf_model_cls = wrap_model_hub(model_dict[self.EXP_TYPE])

        model = hf_model_cls.from_pretrained(cfg.model_name)
        return model

    def _default_source_camera(
        self,
        dist_to_center: float = 2.0,
        batch_size: int = 1,
        device: torch.device = torch.device("cpu"),
    ):
        # return: (N, D_cam_raw)
        canonical_camera_extrinsics = torch.tensor(
            [
                [
                    [1, 0, 0, 0],
                    [0, 0, -1, -dist_to_center],
                    [0, 1, 0, 0],
                ]
            ],
            dtype=torch.float32,
            device=device,
        )
        canonical_camera_intrinsics = create_intrinsics(
            f=0.75,
            c=0.5,
            device=device,
        ).unsqueeze(0)
        source_camera = build_camera_principle(
            canonical_camera_extrinsics, canonical_camera_intrinsics
        )
        return source_camera.repeat(batch_size, 1)

    def _default_render_cameras(
        self,
        n_views: int,
        batch_size: int = 1,
        device: torch.device = torch.device("cpu"),
    ):
        # return: (N, M, D_cam_render)
        render_camera_extrinsics = surrounding_views_linspace(
            n_views=n_views, device=device
        )
        render_camera_intrinsics = (
            create_intrinsics(
                f=0.75,
                c=0.5,
                device=device,
            )
            .unsqueeze(0)
            .repeat(render_camera_extrinsics.shape[0], 1, 1)
        )
        render_cameras = build_camera_standard(
            render_camera_extrinsics, render_camera_intrinsics
        )
        return render_cameras.unsqueeze(0).repeat(batch_size, 1, 1)

    def infer_video(
        self,
        planes: torch.Tensor,
        frame_size: int,
        render_size: int,
        render_views: int,
        render_fps: int,
        dump_video_path: str,
    ):
        N = planes.shape[0]
        render_cameras = self._default_render_cameras(
            n_views=render_views, batch_size=N, device=self.device
        )
        render_anchors = torch.zeros(N, render_cameras.shape[1], 2, device=self.device)
        render_resolutions = (
            torch.ones(N, render_cameras.shape[1], 1, device=self.device) * render_size
        )
        render_bg_colors = (
            torch.ones(
                N, render_cameras.shape[1], 1, device=self.device, dtype=torch.float32
            )
            * 1.0
        )

        frames = []
        for i in range(0, render_cameras.shape[1], frame_size):
            frames.append(
                self.model.synthesizer(
                    planes=planes,
                    cameras=render_cameras[:, i : i + frame_size],
                    anchors=render_anchors[:, i : i + frame_size],
                    resolutions=render_resolutions[:, i : i + frame_size],
                    bg_colors=render_bg_colors[:, i : i + frame_size],
                    region_size=render_size,
                )
            )
        # merge frames
        frames = {k: torch.cat([r[k] for r in frames], dim=1) for k in frames[0].keys()}
        # dump
        os.makedirs(os.path.dirname(dump_video_path), exist_ok=True)
        for k, v in frames.items():
            if k == "images_rgb":
                images_to_video(
                    images=v[0],
                    output_path=dump_video_path,
                    fps=render_fps,
                    gradio_codec=self.cfg.app_enabled,
                )

    def crop_face_image(self, image_path):
        rgb = np.array(Image.open(image_path))
        rgb = torch.from_numpy(rgb).permute(2, 0, 1)
        bbox = self.facedetect(rgb)
        head_rgb = rgb[:, int(bbox[1]) : int(bbox[3]), int(bbox[0]) : int(bbox[2])]
        head_rgb = head_rgb.permute(1, 2, 0)
        head_rgb = head_rgb.cpu().numpy()
        return head_rgb

    @torch.no_grad()
    def parsing(self, img_path):

        parsing_out = self.parsingnet(img_path=img_path, bbox=None)

        alpha = (parsing_out.masks * 255).astype(np.uint8)

        return alpha

    def infer_mesh(
        self,
        image_path: str,
        dump_tmp_dir: str,  
        dump_mesh_dir: str,
        shape_param=None,
    ):

        source_size = self.cfg.source_size
        aspect_standard = 5.0 / 3

        parsing_mask = self.parsing(image_path)

        # prepare reference image
        image, _, _ = infer_preprocess_image(
            image_path,
            mask=parsing_mask,
            intr=None,
            pad_ratio=0,
            bg_color=1.0,
            max_tgt_size=896,
            aspect_standard=aspect_standard,
            enlarge_ratio=[1.0, 1.0],
            render_tgt_size=source_size,
            multiply=14,
            need_mask=True,
        )
        try:
            src_head_rgb = self.crop_face_image(image_path)
        except:
            print("w/o head input!")
            src_head_rgb = np.zeros((112, 112, 3), dtype=np.uint8)


        try:
            src_head_rgb = cv2.resize(
                src_head_rgb,
                dsize=(self.cfg.src_head_size, self.cfg.src_head_size),
                interpolation=cv2.INTER_AREA,
            )  # resize to dino size
        except:
            src_head_rgb = np.zeros(
                (self.cfg.src_head_size, self.cfg.src_head_size, 3), dtype=np.uint8
            )
        

        src_head_rgb = (
            torch.from_numpy(src_head_rgb / 255.0).float().permute(2, 0, 1).unsqueeze(0)
        )  # [1, 3, H, W]

        # save masked image for vis
        save_ref_img_path = os.path.join(
            dump_tmp_dir, "refer_" + os.path.basename(image_path)
        )
        vis_ref_img = (image[0].permute(1, 2, 0).cpu().detach().numpy() * 255).astype(
            np.uint8
        )
        Image.fromarray(vis_ref_img).save(save_ref_img_path)

        device = "cuda"
        dtype = torch.float32
        shape_param = torch.tensor(shape_param, dtype=dtype).unsqueeze(0)

        smplx_params =  dict()
        # cano pose setting
        smplx_params['betas'] = shape_param.to(device)
        smplx_params['root_pose'] = torch.zeros(1,1,3).to(device)
        smplx_params['body_pose'] = torch.zeros(1,1,21, 3).to(device)
        smplx_params['jaw_pose'] = torch.zeros(1, 1, 3).to(device)
        smplx_params['leye_pose'] = torch.zeros(1, 1, 3).to(device)
        smplx_params['reye_pose'] = torch.zeros(1, 1, 3).to(device)
        smplx_params['lhand_pose'] = torch.zeros(1, 1, 15, 3).to(device)
        smplx_params['rhand_pose'] = torch.zeros(1, 1, 15, 3).to(device)
        smplx_params['expr'] = torch.zeros(1, 1, 100).to(device)
        smplx_params['trans'] = torch.zeros(1, 1, 3).to(device)

        self.model.to(dtype)

        gs_app_model_list, query_points, transform_mat_neutral_pose = self.model.infer_single_view(
            image.unsqueeze(0).to(device, dtype),
            src_head_rgb.unsqueeze(0).to(device, dtype),
            None,
            None,
            None,
            None,
            None,
            smplx_params={
                k: v.to(device) for k, v in smplx_params.items()
            },
        )
        smplx_params['transform_mat_neutral_pose'] = transform_mat_neutral_pose

        output_gs = self.model.animation_infer_gs(gs_app_model_list, query_points, smplx_params)

        output_gs_path = '_'.join(os.path.basename(image_path).split('.')[:-1])+'.ply'

        print(f"save mesh to {os.path.join(dump_mesh_dir, output_gs_path)}")
        output_gs.save_ply(os.path.join(dump_mesh_dir, output_gs_path))


    def infer_single(
        self,
        image_path: str,
        motion_seqs_dir,
        motion_img_dir,
        motion_video_read_fps,
        export_video: bool,
        export_mesh: bool,
        dump_tmp_dir: str,  # require by extracting motion seq from video, to save some results
        dump_image_dir: str,
        dump_video_path: str,
        shape_param=None,
    ):
        # for key, value in shape_param.items():
        #     print(f"Key: {key}, Value: {value}")
        beta = shape_param["betas"]
        source_size = self.cfg.source_size
        render_size = self.cfg.render_size
        # render_views = self.cfg.render_views
        render_fps = self.cfg.render_fps
        # mesh_size = self.cfg.mesh_size
        # mesh_thres = self.cfg.mesh_thres
        # frame_size = self.cfg.frame_size
        # source_cam_dist = self.cfg.source_cam_dist if source_cam_dist is None else source_cam_dist
        aspect_standard = 5.0 / 3
        motion_img_need_mask = self.cfg.get("motion_img_need_mask", False)  # False
        vis_motion = self.cfg.get("vis_motion", False)  # False


        if self.parsingnet is not None:
            parsing_mask = self.parsing(image_path)
        else:
            img_np = cv2.imread(image_path)
            remove_np = remove(img_np)
            parsing_mask = remove_np[...,3]
        

        # prepare reference image
        print("image preprocess")
        image, _, _ = infer_preprocess_image(
            image_path,
            mask=parsing_mask,
            intr=None,
            pad_ratio=0,
            bg_color=1.0,
            max_tgt_size=896,
            aspect_standard=aspect_standard,
            enlarge_ratio=[1.0, 1.0],
            render_tgt_size=source_size,
            multiply=14,
            need_mask=True,
        )
        try:
            src_head_rgb = self.crop_face_image(image_path)
        except:
            print("w/o head input!")
            src_head_rgb = np.zeros((112, 112, 3), dtype=np.uint8)


        try:
            src_head_rgb = cv2.resize(
                src_head_rgb,
                dsize=(self.cfg.src_head_size, self.cfg.src_head_size),
                interpolation=cv2.INTER_AREA,
            )  # resize to dino size
        except:
            src_head_rgb = np.zeros(
                (self.cfg.src_head_size, self.cfg.src_head_size, 3), dtype=np.uint8
            )

        src_head_rgb = (
            torch.from_numpy(src_head_rgb / 255.0).float().permute(2, 0, 1).unsqueeze(0)
        )  # [1, 3, H, W]

        # save masked image for vis
        save_ref_img_path = os.path.join(
            dump_tmp_dir, "refer_" + os.path.basename(image_path)
        )
        vis_ref_img = (image[0].permute(1, 2, 0).cpu().detach().numpy() * 255).astype(
            np.uint8
        )
        Image.fromarray(vis_ref_img).save(save_ref_img_path)

        # read motion seq
        print("read motion seq")
        motion_name = os.path.dirname(
            motion_seqs_dir[:-1] if motion_seqs_dir[-1] == "/" else motion_seqs_dir
        )
        motion_name = os.path.basename(motion_name)

        if motion_name in self.motion_dict:
            motion_seq = self.motion_dict[motion_name]
        else:
            motion_seqs_dir = "./tmp/00001.json"
            motion_seq = prepare_motion_seqs(
                motion_seqs_dir,
                motion_img_dir,
                save_root=dump_tmp_dir,
                fps=motion_video_read_fps,
                bg_color=1.0,
                aspect_standard=aspect_standard,
                enlarge_ratio=[1.0, 1, 0],
                render_image_res=render_size,
                multiply=16,
                need_mask=motion_img_need_mask,
                vis_motion=vis_motion,
                
            )
            self.motion_dict[motion_name] = motion_seq

        camera_size = len(motion_seq["motion_seqs"])

        device = "cuda"
        dtype = torch.float32
        beta = torch.tensor(beta, dtype=dtype).unsqueeze(0)

        self.model.to(dtype)
        smplx_params = motion_seq['smplx_params']
        smplx_params['betas'] = beta.to(device)
        # smplx_params =  dict()
        # # cano pose setting
        # smplx_params['betas'] = shape_param.to(device)
        # smplx_params['root_pose'] = torch.zeros(1,1,3).to(device)
        # smplx_params['body_pose'] = torch.zeros(1,1,21, 3).to(device)
        # smplx_params['jaw_pose'] = torch.zeros(1, 1, 3).to(device)
        # smplx_params['leye_pose'] = torch.zeros(1, 1, 3).to(device)
        # smplx_params['reye_pose'] = torch.zeros(1, 1, 3).to(device)
        # smplx_params['lhand_pose'] = torch.zeros(1, 1, 15, 3).to(device)
        # smplx_params['rhand_pose'] = torch.zeros(1, 1, 15, 3).to(device)
        # smplx_params['expr'] = torch.zeros(1, 1, 100).to(device)
        # smplx_params['trans'] = torch.zeros(1, 1, 3).to(device)

        # 输入单张图片、人脸图片、相机内外参、渲染背景和驱动SMPL序列
        gs_model_list, query_points, transform_mat_neutral_pose = self.model.infer_single_view(
            image.unsqueeze(0).to(device, dtype),
            src_head_rgb.unsqueeze(0).to(device, dtype),
            None,
            None,
            render_c2ws=motion_seq["render_c2ws"].to(device),
            render_intrs=motion_seq["render_intrs"].to(device),
            render_bg_colors=motion_seq["render_bg_colors"].to(device),
            smplx_params={
                k: v.to(device) for k, v in smplx_params.items()
            },
        )


        batch_list = [] 
        batch_size = 40  # avoid memeory out! 40

        for batch_i in range(0, camera_size, batch_size):
            with torch.no_grad():
                # TODO check device and dtype
                # dict_keys(['comp_rgb', 'comp_rgb_bg', 'comp_mask', 'comp_depth', '3dgs'])

                print(f"batch: {batch_i}, total: {camera_size //batch_size +1} ")

                keys = [
                    "root_pose",
                    "body_pose",
                    "jaw_pose",
                    "leye_pose",
                    "reye_pose",
                    "lhand_pose",
                    "rhand_pose",
                    "trans",
                    "focal",
                    "princpt",
                    "img_size_wh",
                    "expr",
                ]


                batch_smplx_params = dict()
                batch_smplx_params["betas"] = beta.to(device)
                batch_smplx_params['transform_mat_neutral_pose'] = transform_mat_neutral_pose
                for key in keys:
                    batch_smplx_params[key] = motion_seq["smplx_params"][key][
                        :, batch_i : batch_i + batch_size
                    ].to(device)

                # def animation_infer(self, gs_model_list, query_points, smplx_params, render_c2ws, render_intrs, render_bg_colors, render_h, render_w):
                res = self.model.animation_infer(gs_model_list, query_points, batch_smplx_params,
                    render_c2ws=motion_seq["render_c2ws"][
                        :, batch_i : batch_i + batch_size
                    ].to(device),
                    render_intrs=motion_seq["render_intrs"][
                        :, batch_i : batch_i + batch_size
                    ].to(device),
                    render_bg_colors=motion_seq["render_bg_colors"][
                        :, batch_i : batch_i + batch_size
                    ].to(device),
                    )

            comp_rgb = res["comp_rgb"] # [Nv, H, W, 3], 0-1
            comp_mask = res["comp_mask"] # [Nv, H, W, 3], 0-1

            comp_mask[comp_mask < 0.5] = 0.0

            batch_rgb = comp_rgb * comp_mask + (1 - comp_mask) * 1
            batch_rgb = (batch_rgb.clamp(0,1) * 255).to(torch.uint8).detach().cpu().numpy()
            batch_list.append(batch_rgb)

            del res
            torch.cuda.empty_cache()
        
        rgb = np.concatenate(batch_list, axis=0)

        os.makedirs(os.path.dirname(dump_video_path), exist_ok=True)

        print(f"save video to {dump_video_path}")


        images_to_video(
            rgb,
            output_path=dump_video_path,
            fps=render_fps,
            gradio_codec=False,
            verbose=True,
        )

    def train_single(
        self,
        image_path: str,
        motion_seqs_dir,
        motion_img_dir,
        motion_video_read_fps,
        export_video: bool,
        export_mesh: bool,
        dump_tmp_dir: str,  # require by extracting motion seq from video, to save some results
        dump_image_dir: str,
        dump_video_path: str,
        infer_motion_seqs_dir: str,
        dataset_dir,
        K,
        R,
        T,
        visible_verts,
        invisible_verts,
        shape_param=None,
        ):
        beta = shape_param["betas"]
        source_size = self.cfg.source_size
        render_size = self.cfg.render_size
        render_fps = self.cfg.render_fps
        # mesh_size = self.cfg.mesh_size
        # mesh_thres = self.cfg.mesh_thres
        # frame_size = self.cfg.frame_size
        # source_cam_dist = self.cfg.source_cam_dist if source_cam_dist is None else source_cam_dist
        aspect_standard = 5.0 / 3
        motion_img_need_mask = self.cfg.get("motion_img_need_mask", False)  # False
        vis_motion = self.cfg.get("vis_motion", False)  # False


        if self.parsingnet is not None:
            parsing_mask = self.parsing(image_path)
        else:
            img_np = cv2.imread(image_path)
            remove_np = remove(img_np)
            parsing_mask = remove_np[...,3]
        

        # prepare reference image
        print("prepare reference image")
        image, _, _ = infer_preprocess_image(
            image_path,
            mask=parsing_mask,
            intr=None,
            pad_ratio=0,
            bg_color=1.0,
            max_tgt_size=896,
            aspect_standard=aspect_standard,
            enlarge_ratio=[1.0, 1.0],
            render_tgt_size=source_size,
            multiply=14,
            need_mask=True,
        )
        try:
            src_head_rgb = self.crop_face_image(image_path)
        except:
            print("w/o head input!")
            src_head_rgb = np.zeros((112, 112, 3), dtype=np.uint8)


        try:
            src_head_rgb = cv2.resize(
                src_head_rgb,
                dsize=(self.cfg.src_head_size, self.cfg.src_head_size),
                interpolation=cv2.INTER_AREA,
            )  # resize to dino size
        except:
            src_head_rgb = np.zeros(
                (self.cfg.src_head_size, self.cfg.src_head_size, 3), dtype=np.uint8
            )

        src_head_rgb = (
            torch.from_numpy(src_head_rgb / 255.0).float().permute(2, 0, 1).unsqueeze(0)
        )  # [1, 3, H, W]

        # save masked image for vis
        save_ref_img_path = os.path.join(
            dump_tmp_dir, "refer_" + os.path.basename(image_path)
        )
        vis_ref_img = (image[0].permute(1, 2, 0).cpu().detach().numpy() * 255).astype(
            np.uint8
        )
        Image.fromarray(vis_ref_img).save(save_ref_img_path)

        # read motion seq
        print("read motion seq")
        motion_name = os.path.dirname(
            motion_seqs_dir[:-1] if motion_seqs_dir[-1] == "/" else motion_seqs_dir
        )
        motion_name = os.path.basename(motion_name)

        motion_seqs_dir = "./tmp/00001.json"
        new_motion_seqs_dir = "./tmp/00001opt.json"
        if motion_name in self.motion_dict:
            motion_seq = self.motion_dict[motion_name]
        else:
            print("begin optimize smpl")
            smpl_estimator = SMPLEstimator("pixie", "cuda")
            device = "cuda"
            #要在这里读取json并用多视角图片优化SMPL
            step_count = 0 # global steps
            torch.cuda.empty_cache()

            smpl_dict = smpl_estimator.read_lhm_json(motion_seqs_dir)
            optimed_pose = smpl_dict["optimed_pose"].to(device).requires_grad_(True)
            optimed_trans = smpl_dict["optimed_trans"].to(device).requires_grad_(True)
            optimed_betas = smpl_dict["optimed_betas"].to(device).requires_grad_(True)
            optimed_orient = smpl_dict["optimed_orient"].to(device).requires_grad_(True)
            optimizer_smpl = torch.optim.Adam([
                optimed_pose, optimed_trans, optimed_betas, optimed_orient
            ], lr=1e-2, amsgrad=True)
            scheduler_smpl = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer_smpl,
                mode="min",
                factor=0.5,
                verbose=0,
                min_lr=1e-5,
                patience=5,
            )
            
            def tensor2variable(tensor, device):
                return tensor.requires_grad_(True).to(device)
            expression = tensor2variable(smpl_dict["expression"], device)
            jaw_pose = tensor2variable(smpl_dict["jaw_pose"], device)
            left_hand_pose = tensor2variable(smpl_dict["left_hand_pose"], device)
            right_hand_pose = tensor2variable(smpl_dict["right_hand_pose"], device)
            scale = smpl_dict["scale"]
            smpl_faces = torch.tensor(self.pose_estimator.mhmr_model.smpl_layer["neutral_10"].bm_x.faces).unsqueeze(0).to(device)

            step_loop = tqdm(range(10)) #30

            print("SMPL render setting")
            # renderer initialization
            smpl_renderer = SMPLRenderer(size=(832,480), device=device)
            smpl_renderer.set_cameras(R,T,K)
            # loss initialization
            losses = init_loss()


            for step in step_loop:
                step_count += 1 # tb
                optimizer_smpl.zero_grad()
                
                # SMPL-X forward
                smpl_verts, smpl_joints_3d = smpl_estimator.smpl_forward_lhm(
                    optimed_betas=optimed_betas,
                    optimed_pose=optimed_pose,
                    optimed_trans=optimed_trans,
                    optimed_orient=optimed_orient,
                    expression=expression,
                    jaw_pose=jaw_pose,
                    left_hand_pose=left_hand_pose,
                    right_hand_pose=right_hand_pose,
                    scale=scale,
                )
                
                # save initialized SMPL-X mesh
                if step == 0:
                    mesh_path = f"./tmp/smplx_initialized.obj"
                    save_optimed_mesh(mesh_path, smpl_verts, smpl_faces)
                    nvs_data = process_video_fit(dataset_dir)




                # differentiable rendering
                smpl_renderer.load_mesh(smpl_verts, smpl_faces) # load updated mesh in each step
                # silhouette loss
                smpl_masks = smpl_renderer.render_mask(bg="black") # (B, 512, 512)
                gt_masks = nvs_data["img_mask"].to(device)
                diff_S = torch.abs(smpl_masks - gt_masks)
                losses["silhouette"]["value"] = diff_S.mean(dim=[1,2]) # (B,)
                # if step == 0 or step==29:
                #     for i in range(9):
                #         mask10 = smpl_masks[i].detach().cpu().numpy()  # (512,512)
                #         mask_img = (mask10 * 255).astype("uint8")
                #         Image.fromarray(mask_img).save(f"/data1/zhangbotao/LHM/tpami/mask_smpl_{step:03d}_{i:03d}.png")
                #         mask10 = gt_masks[i].detach().cpu().numpy()  # (512,512)
                #         mask_img = (mask10 * 255).astype("uint8")
                #         Image.fromarray(mask_img).save(f"/data1/zhangbotao/LHM/tpami/mask_gt_{step:03d}_{i:03d}.png")
                # Loss weights:
                # - normal weight = 1 (only in the intersected part of SMPL-X body and cloth)
                # - mask weight = 0.1 if self-occlusion else 1
                # - joint weight = 50 if loose cloth else 5
                # - front view larger weight
                _, smpl_masks_fake = smpl_renderer.render_normal_screen_space(
                    bg="black", return_mask=True
                    ) # (B, 3, H, W),  (B, H, W) to get SMPL-X body mask
                # self-occlusion detection
                body_overlap = (gt_masks * smpl_masks_fake).sum(dim=[1, 2]) / smpl_masks_fake.sum(dim=[1, 2]) # (B,)
                body_overlap_flag = body_overlap < 1.0
                losses["silhouette"]["weight"] = [0.1 if flag else 1.0 for flag in body_overlap_flag]
                # loose cloth detection
                cloth_overlap = diff_S.sum(dim=[1, 2]) / gt_masks.sum(dim=[1, 2])
                cloth_overlap_flag = cloth_overlap > 1.0 # (B,)
                losses["joint"]["weight"] = [50.0 if flag else 5.0 for flag in cloth_overlap_flag]
                
                                    
                # 2d joint loss
                smpl_joints_2d = smpl_renderer.project_joints(smpl_joints_3d)  # (B=1, 45, 3) [-1,1]-> (B=24, 45, 2) [0,1]
                smpl_lmks = smpl_joints_2d[:, smpl_estimator.SMPLX_object.ghum_smpl_pairs[:, 1], :] # select 25 joints (B, 25, 2)
                gt_lmks = nvs_data["landmark"][:, smpl_estimator.SMPLX_object.ghum_smpl_pairs[:, 0], :2].to(device)
                gt_conf = nvs_data["landmark"][:, smpl_estimator.SMPLX_object.ghum_smpl_pairs[:, 0], -1].to(device)
                occluded_idx = torch.where(body_overlap_flag)[0] # self-occluded frame
                gt_conf[occluded_idx] *= gt_conf[occluded_idx] > 0.50
                diff_J = torch.norm(gt_lmks - smpl_lmks, dim=2) * gt_conf # (B, 25)
                losses['joint']['value'] = diff_J.mean(dim=1) # (B,)

                if(step==0):
                    save_landmarks_pil(gt_lmks, smpl_lmks)

                # Calculate loss and optimize SMPL-X for this step
                smpl_loss = 0.0
                pbar_desc = "Body Fitting -- "
                loss_items = ["silhouette", "joint"]
                loss_items = ["silhouette"]
                for k in loss_items:
                    losses[k]["weight"][0] = losses[k]["weight"][0] * 10.0 # 10 weight for the front view
                    per_loop_loss = (
                        losses[k]["value"] * torch.tensor(losses[k]["weight"]).to(device)
                    ).mean()

                    smpl_loss += per_loop_loss
                
                smpl_loss.backward()

                with torch.no_grad():
                    # 假设第 12 和 15 是 neck/head
                    optimed_pose.grad[:, 12:16, :] = 0 
                 
                optimizer_smpl.step()
                scheduler_smpl.step(smpl_loss)

            mesh_path = f"./tmp/smplx_optimized.obj"
            save_optimed_mesh(mesh_path, smpl_verts, smpl_faces)

            new_smpl_dict = {}
            new_smpl_dict["optimed_pose"] = optimed_pose
            new_smpl_dict["optimed_trans"] = optimed_trans
            new_smpl_dict["optimed_betas"] = optimed_betas
            new_smpl_dict["optimed_orient"] = optimed_orient
            new_smpl_dict["expression"] = expression
            new_smpl_dict["jaw_pose"] = jaw_pose 
            new_smpl_dict["left_hand_pose"] = left_hand_pose
            new_smpl_dict["right_hand_pose"] = right_hand_pose
            new_smpl_dict["scale"] = scale
            smpl_estimator.smpldict_to_json(new_smpl_dict, motion_seqs_dir, new_motion_seqs_dir)


            motion_seq = prepare_motion_seqs_single_seq(
                new_motion_seqs_dir, #motion_seqs_dir
                motion_img_dir,
                save_root=dump_tmp_dir,
                fps=motion_video_read_fps,
                bg_color=1.0,
                aspect_standard=aspect_standard,
                enlarge_ratio=[1.0, 1, 0],
                render_image_res=render_size,
                multiply=16,
                need_mask=motion_img_need_mask,
                vis_motion=vis_motion,
            )
            self.motion_dict[motion_name] = motion_seq

        camera_size = len(motion_seq["motion_seqs"])

        device = "cuda"
        dtype = torch.float32
        beta = torch.tensor(beta, dtype=dtype).unsqueeze(0)

        self.model.to(dtype)
        smplx_params = motion_seq['smplx_params']
        smplx_params['betas'] = beta.to(device)

        infer_smplx_params_list = []

        
        infer_motion_seq = prepare_motion_seqs(
                infer_motion_seqs_dir,
                motion_img_dir,
                save_root=dump_tmp_dir,
                fps=motion_video_read_fps,
                bg_color=1.0,
                aspect_standard=aspect_standard,
                enlarge_ratio=[1.0, 1, 0],
                render_image_res=render_size,
                multiply=16,
                need_mask=motion_img_need_mask,
                vis_motion=vis_motion,
                ref_json_path=new_motion_seqs_dir,
            )
        infer_smplx_params = infer_motion_seq['smplx_params']
        infer_smplx_params['betas'] = beta.to(device)
        

        # 输入单张图片、人脸图片、相机内外参、渲染背景和驱动SMPL序列
        gs_model_list, query_points, query_points_wosample,transform_mat_neutral_pose = self.model.infer_single_view(
            image.unsqueeze(0).to(device, dtype),
            src_head_rgb.unsqueeze(0).to(device, dtype),
            None,
            None,
            render_c2ws=motion_seq["render_c2ws"].to(device),
            render_intrs=motion_seq["render_intrs"].to(device),
            render_bg_colors=motion_seq["render_bg_colors"].to(device),
            smplx_params={
                k: v.to(device) for k, v in smplx_params.items()
            },
        )

        points = query_points[0]
        mesh_v = query_points_wosample[0]
        # print(query_points.shape)
        # print(query_points_wosample)
        # print(invisible_verts.shape)

        dists, idx, _ = knn_points(
        points.unsqueeze(0),   # (1, P, 3)
        mesh_v.unsqueeze(0),   # (1, V, 3)
        K=1,
        return_nn=False
        )
        nearest_idx = idx[0, :, 0]  # (P,)
        #我这里要去掉人手、人脚、人头被遮挡的部分
        # 头部 & 脸
        import json
        with open("./smplx_vert_segmentation.json", "r") as f:
            part_dict = json.load(f)
        parts_to_exclude = [
        "head", "leftEye", "rightEye", "eyeballs",
        "leftFoot", "rightFoot", "leftToeBase", "rightToeBase",
        'neck',
        ]
        exclude_list = []
        for part in parts_to_exclude:
            if part in part_dict:
                exclude_list.extend(part_dict[part])
        exclude_indices = torch.tensor(sorted(set(exclude_list)), dtype=torch.long, device=invisible_verts.device)


        parts_to_include = [
        "leftShoulder","rightShoulder",
        "leftArm","rightArm",
        "leftHand", "rightHand", "rightHandIndex1", "leftHandIndex1",
        "leftForeArm", "rightForeArm",

        # "spine2","spine1","spine"
        # "left_side","right_side",
        # "leftUpLeg","rightUpLeg",
        ]
        # 构建 include_indices
        include_list = []
        for part in parts_to_include:
                include_list.extend(part_dict[part])
        include_indices = torch.tensor(sorted(set(include_list)), dtype=torch.long, device=invisible_verts.device)

        # 差集运算
        # filtered_invisible = torch.tensor(
        #     list(set(invisible_verts.tolist()) - set(exclude_indices.tolist())),
        #     dtype=torch.long,
        #     device=invisible_verts.device
        # )
        # filtered_invisible = torch.tensor(
        #     list(set(filtered_invisible.tolist()) | set(include_indices.tolist())),
        #     dtype=torch.long,
        #     device=invisible_verts.device
        # )
        filtered_invisible = torch.tensor(
            list(set(include_indices.tolist()) - set(exclude_indices.tolist())),
            dtype=torch.long,
            device=invisible_verts.device
        )




        parts_to_exclude = [
        "head", "leftEye", "rightEye", "eyeballs",
        "leftFoot", "rightFoot", "leftToeBase", "rightToeBase",
        "leftHand", "rightHand", "rightHandIndex1", "leftHandIndex1",
        "leftForeArm", "rightForeArm",
        'neck', 'leftShoulder','rightShoulder',
        # "leftHand", "rightHand",
        ]
        exclude_list = []
        for part in parts_to_exclude:
            if part in part_dict:
                exclude_list.extend(part_dict[part])
        shs_exclude_indices = torch.tensor(sorted(set(exclude_list)), dtype=torch.long, device=invisible_verts.device)
        # shs_parts_include = [
        #     "hips","spine",
        #     "spine1","spine2",
        #     "dang",
        # ]
        shs_parts_include = [
            # "left_side","right_side",
            # "leftShoulder","rightShoulder",
            # "long_dang",
            "short_dang",
        ]
        shs_include_list = []
        for part in shs_parts_include:
            shs_include_list.extend(part_dict[part])
        shs_include_indices = torch.tensor(sorted(set(shs_include_list)), dtype=torch.long, device=invisible_verts.device)

        shs_filtered_invisible = torch.tensor(
            list(set(invisible_verts.tolist()) - set(shs_exclude_indices.tolist())),
            dtype=torch.long,
            device=invisible_verts.device
        )
        shs_include_indices = torch.tensor(
            list(set(shs_include_indices.tolist()) | set(shs_filtered_invisible.tolist())),
            dtype=torch.long,
            device=invisible_verts.device
        )


        # import trimesh
        # # invisible_points = mesh_v[filtered_invisible.cpu().numpy()]
        # # pc_invisible = trimesh.points.PointCloud(invisible_points.cpu().numpy())
        # # out_path = "/data1/zhangbotao/LHM/tpami/mesh/new_invisible_points.ply"
        # # pc_invisible.export(out_path)

        # # pc_invisible = trimesh.points.PointCloud(points.cpu().numpy())
        # # out_path = "/data1/zhangbotao/LHM/tpami/mesh/gaussian_points.ply"
        # # pc_invisible.export(out_path)
        # # 加载SMPL网格
        # mesh = trimesh.load('/data1/zhangbotao/LHM/tpami/smplx_optimized.obj')
        # # 创建顶点颜色数组，默认浅灰色
        # vertex_colors = np.full((len(mesh.vertices), 3), [0.9, 0.9, 0.9])  # 浅灰色背景
        # # 将shs_include_indices标记为红色

        # # 使用红色标记目标区域
        # red_color = [144/255 , 238/255 , 144/255 ] # 纯红色
        # # 为shs_include_indices分配红色
        # vertex_colors[shs_include_indices.cpu().numpy()] = red_color
        # # 更新网格的顶点颜色
        # mesh.visual.vertex_colors = (vertex_colors * 255).astype(np.uint8)
        # # 保存结果
        # output_path = '/data1/zhangbotao/LHM/tpami/smplx_red_colored.obj'
        # mesh.export(output_path)


        mask_invisible = torch.isin(nearest_idx, filtered_invisible)
        mask_invisible = torch.nonzero(mask_invisible).squeeze(1)

        shs_mask = torch.isin(nearest_idx, shs_include_indices)
        shs_mask = torch.nonzero(shs_mask).squeeze(1)




        #ACAP
        acap_parts_to_include = [
        "head", "neck", "leftEye", "rightEye", "eyeballs",
        "leftFoot", "rightFoot", "leftToeBase", "rightToeBase", 
        # "leftLeg", "rightLeg",
        # "left_side","right_side",
        # "leftUpLeg","rightUpLeg",
        ]
        # 构建 include_indices
        acap_include_list = []
        for part in acap_parts_to_include:
            acap_include_list.extend(part_dict[part])
        acap_include_indices = torch.tensor(sorted(set(acap_include_list)), dtype=torch.long, device=invisible_verts.device)
        acap_mask = ~torch.isin(nearest_idx, acap_include_indices)
        acap_mask = torch.nonzero(acap_mask).squeeze(1)

        batch_list = [] 
        infer_batch_list = []
        LHM_infer_batch_list = []
        batch_size = 320  # avoid memeory out! 40 320

        for batch_i in range(0, camera_size, batch_size):
            # with torch.no_grad():
                # TODO check device and dtype
                # dict_keys(['comp_rgb', 'comp_rgb_bg', 'comp_mask', 'comp_depth', '3dgs'])

            print(f"batch: {batch_i}, total: {camera_size //batch_size +1} ")

            keys = [
                    "root_pose",
                    "body_pose",
                    "jaw_pose",
                    "leye_pose",
                    "reye_pose",
                    "lhand_pose",
                    "rhand_pose",
                    "trans",
                    "focal",
                    "princpt",
                    "img_size_wh",
                    "expr",
                ]


            batch_smplx_params = dict()
            batch_smplx_params["betas"] = beta.to(device)
            batch_smplx_params['transform_mat_neutral_pose'] = transform_mat_neutral_pose
            for key in keys:
                    batch_smplx_params[key] = motion_seq["smplx_params"][key][
                        :, batch_i : batch_i + batch_size
                    ].to(device)

            infer_smplx_params_list = []
            batch_infer_smplx_params = dict()
            batch_infer_smplx_params["betas"] = beta.to(device)
            batch_infer_smplx_params['transform_mat_neutral_pose'] = transform_mat_neutral_pose
            for key in keys:
                batch_infer_smplx_params[key] = infer_smplx_params[key][
                        :, batch_i : batch_i + batch_size
                    ].to(device)
            infer_smplx_params_list.append(batch_infer_smplx_params)

                # def animation_infer(self, gs_model_list, query_points, smplx_params, render_c2ws, render_intrs, render_bg_colors, render_h, render_w):
            res, infer_res_list, LHM_infer_res_list = self.model.animation_infer_train(gs_model_list, query_points, batch_smplx_params,
                    render_c2ws=motion_seq["render_c2ws"][
                        :, batch_i : batch_i + batch_size
                    ].to(device),
                    render_intrs=motion_seq["render_intrs"][
                        :, batch_i : batch_i + batch_size
                    ].to(device),
                    render_bg_colors=motion_seq["render_bg_colors"][
                        :, batch_i : batch_i + batch_size
                    ].to(device),
                    dataset_dir=dataset_dir,
                    K=K,
                    R=R,
                    T=T,
                    mask = mask_invisible,
                    shs_mask = shs_mask,
                    new_params_list = infer_smplx_params_list,
                    acap_mask = acap_mask,
                    )
            
            # res = infer_res_list[0]
            infer_res = infer_res_list[0]
            LHM_infer_res = LHM_infer_res_list[0]

            comp_rgb = res["comp_rgb"] # [Nv, H, W, 3], 0-1
            comp_mask = res["comp_mask"] # [Nv, H, W, 3], 0-1
            comp_mask[comp_mask < 0.5] = 0.0

            batch_rgb = comp_rgb * comp_mask + (1 - comp_mask) * 1
            batch_rgb = (batch_rgb.clamp(0,1) * 255).to(torch.uint8).detach().cpu().numpy()
            batch_list.append(batch_rgb)
            del res

            comp_rgb = infer_res["comp_rgb"] # [Nv, H, W, 3], 0-1
            comp_mask = infer_res["comp_mask"] # [Nv, H, W, 3], 0-1
            comp_mask[comp_mask < 0.5] = 0.0

            batch_rgb = comp_rgb * comp_mask + (1 - comp_mask) * 1
            batch_rgb = (batch_rgb.clamp(0,1) * 255).to(torch.uint8).detach().cpu().numpy()
            infer_batch_list.append(batch_rgb)
            del infer_res_list

            comp_rgb = LHM_infer_res["comp_rgb"] # [Nv, H, W, 3], 0-1
            comp_mask = LHM_infer_res["comp_mask"] # [Nv, H, W, 3], 0-1
            comp_mask[comp_mask < 0.5] = 0.0

            batch_rgb = comp_rgb * comp_mask + (1 - comp_mask) * 1
            batch_rgb = (batch_rgb.clamp(0,1) * 255).to(torch.uint8).detach().cpu().numpy()
            LHM_infer_batch_list.append(batch_rgb)
            del LHM_infer_res_list
            torch.cuda.empty_cache()
        
        rgb = np.concatenate(batch_list, axis=0)
        infer_rgb = np.concatenate(infer_batch_list, axis=0)
        LHM_infer_rgb = np.concatenate(LHM_infer_batch_list, axis=0)

        os.makedirs(os.path.dirname(dump_video_path), exist_ok=True)

        print(f"save video to {dump_video_path}")

        images_to_video(
            rgb,
            output_path=dump_video_path,
            fps=render_fps,
            gradio_codec=False,
            verbose=True,
        )
        # os.path.join(os.path.dirname(dump_video_path),"novel_view.mp4")
        images_to_video(
            infer_rgb,
            output_path=dump_video_path[:-4]+"_"+os.path.basename(os.path.dirname(infer_motion_seqs_dir))+".mp4",
            fps=render_fps,
            gradio_codec=False,
            verbose=True,
        )

        images_to_video(
            LHM_infer_rgb,
            output_path=dump_video_path[:-4]+"_"+os.path.basename(os.path.dirname(infer_motion_seqs_dir))+"_LHM"+".mp4",
            fps=render_fps,
            gradio_codec=False,
            verbose=True,
        )


    def infer(self):

        image_paths = []
        if os.path.isfile(self.cfg.image_input):
            omit_prefix = os.path.dirname(self.cfg.image_input)
            image_paths.append(self.cfg.image_input)
        else:
            omit_prefix = self.cfg.image_input
            suffixes = (".jpg", ".jpeg", ".png", ".webp", ".JPG")
            for root, dirs, files in os.walk(self.cfg.image_input):
                for file in files:
                    if file.endswith(suffixes):
                        image_paths.append(os.path.join(root, file))
            image_paths.sort()

        # alloc to each DDP worker
        image_paths = image_paths[
            self.accelerator.process_index :: self.accelerator.num_processes
        ]


        for image_path in tqdm(image_paths,
            disable=not self.accelerator.is_local_main_process,
        ):

            # prepare dump paths
            image_name = os.path.basename(image_path)
            uid = image_name.split(".")[0]
            subdir_path = os.path.dirname(image_path).replace(omit_prefix, "")
            subdir_path = (
                subdir_path[1:] if subdir_path.startswith("/") else subdir_path
            )
            print("subdir_path and uid:", subdir_path, uid)

            # setting config
            motion_seqs_dir = self.cfg.motion_seqs_dir
            motion_name = os.path.dirname(
                motion_seqs_dir[:-1] if motion_seqs_dir[-1] == "/" else motion_seqs_dir
            )
            motion_name = os.path.basename(motion_name)
            dump_video_path = os.path.join(
                self.cfg.video_dump,
                subdir_path,
                motion_name,
                f"{uid}.mp4",
            )
            dump_image_dir = os.path.join(
                self.cfg.image_dump,
                subdir_path,
            )
            dump_mesh_dir = os.path.join(
                self.cfg.mesh_dump,
                subdir_path,
            )
            dump_tmp_dir = os.path.join(self.cfg.image_dump, subdir_path, "tmp_res")
            os.makedirs(dump_image_dir, exist_ok=True)
            os.makedirs(dump_tmp_dir, exist_ok=True)
            os.makedirs(dump_mesh_dir, exist_ok=True)

            # shape_pose = self.pose_estimator(image_path)

            print("预测输入图片的SMPL")
            # shape_pose = self.pose_estimator.predict_smpl_orth2(image_path)
            shape_pose = self.pose_estimator.predict_smpl(image_path)
            
            # try:
            #     assert shape_pose.ratio>0.4, f"body ratio is too small: {shape_pose.ratio}"
            # except:
            #     continue



            # acts = ["jntm","doctor1","doctor2","girl2","ex5"]
            # acts = ["jntm","mimo2","taiji"]
            # acts = ["jntm"]
            # print("self.cfg.dataset_dir",self.cfg.dataset_dir)
            # for act in acts:
            #     infer_motion_seqs = os.path.join("./train_data/motion_video",act,"smplx_params")
            #     self.train_single(
            #             image_path,
            #             motion_seqs_dir=self.cfg.motion_seqs_dir,
            #             motion_img_dir=self.cfg.motion_img_dir,
            #             motion_video_read_fps=self.cfg.motion_video_read_fps,
            #             export_video=self.cfg.export_video,
            #             export_mesh=self.cfg.export_mesh,
            #             dump_tmp_dir=dump_tmp_dir,
            #             dump_image_dir=dump_image_dir,
            #             dump_video_path=dump_video_path,
            #             infer_motion_seqs_dir = infer_motion_seqs,
            #             dataset_dir = self.cfg.dataset_dir, # self.pose_estimator.data_dir, "/data1/zhangbotao/LHM/OUTPUT/TMP/frames_480P_7processnew"
            #             K = self.pose_estimator.K,
            #             R = self.pose_estimator.R,
            #             T = self.pose_estimator.T,
            #             visible_verts = self.pose_estimator.visible_verts,
            #             invisible_verts = self.pose_estimator.invisible_verts,
            #             shape_param=shape_pose,
            #         )
            if self.cfg.only_predict_smpl:
                return
            else:
                infer_motion_seqs = self.cfg.motion_seqs_dir
                self.train_single(
                            image_path,
                            motion_seqs_dir=self.cfg.motion_seqs_dir,
                            motion_img_dir=self.cfg.motion_img_dir,
                            motion_video_read_fps=self.cfg.motion_video_read_fps,
                            export_video=self.cfg.export_video,
                            export_mesh=self.cfg.export_mesh,
                            dump_tmp_dir=dump_tmp_dir,
                            dump_image_dir=dump_image_dir,
                            dump_video_path=dump_video_path,
                            infer_motion_seqs_dir = infer_motion_seqs,
                            dataset_dir = self.cfg.dataset_dir, # self.pose_estimator.data_dir, "/data1/zhangbotao/LHM/OUTPUT/TMP/frames_480P_7processnew"
                            K = self.pose_estimator.K,
                            R = self.pose_estimator.R,
                            T = self.pose_estimator.T,
                            visible_verts = self.pose_estimator.visible_verts,
                            invisible_verts = self.pose_estimator.invisible_verts,
                            shape_param=shape_pose,
                        )




            


    

            # if self.cfg.export_mesh is not None:
            #     self.infer_mesh(
            #         image_path,
            #         dump_tmp_dir=dump_tmp_dir,
            #         dump_mesh_dir=dump_mesh_dir,
            #         shape_param=shape_pose.beta,
            #     )
            # else:
            #     self.infer_single(
            #         image_path,
            #         motion_seqs_dir=self.cfg.motion_seqs_dir,
            #         motion_img_dir=self.cfg.motion_img_dir,
            #         motion_video_read_fps=self.cfg.motion_video_read_fps,
            #         export_video=self.cfg.export_video,
            #         export_mesh=self.cfg.export_mesh,
            #         dump_tmp_dir=dump_tmp_dir,
            #         dump_image_dir=dump_image_dir,
            #         dump_video_path=dump_video_path,
            #         shape_param=shape_pose,
            #     )


