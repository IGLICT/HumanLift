import torch
from diffsynth import ModelManager, WanVideoPipeline, save_video, VideoData, WanUniAnimateVideoPipeline
from modelscope import snapshot_download, dataset_snapshot_download
from PIL import Image
import os
import pickle
from PIL import Image
import numpy as np
import random
import pickle
import torch
from io import BytesIO
import torchvision
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image, ImageFilter
import  torch.nn  as nn
import cv2
import sys  
from crop_imgs import pad_480x480_to_832x480
sys.path.append("../../")  

# define hight and width
# height = 480#832
# width = 832#480
height = 832
width = 480
seed = 0
max_frames = 81
use_teacache = False

test_list_path= [
    # Format: [frame_interval, reference image, driving pose sequence]
    # [1, "data/images/WOMEN-Blouses_Shirts-id_00004955-01_4_full.jpg", "data/saved_pose/WOMEN-Blouses_Shirts-id_00004955-01_4_full"],
    # [1, "data/images/musk.jpg", "data/saved_pose/musk"],
    # [1, "data/images/WOMEN-Blouses_Shirts-id_00005125-03_4_full.jpg", "data/saved_pose/WOMEN-Blouses_Shirts-id_00005125-03_4_full"],
    # [1, "data/images/IMG_20240514_104337.jpg", "data/saved_pose/IMG_20240514_104337"],
    # [1, "data/images/10.jpg", "data/saved_pose/10"],
    # [1, "data/images/reference4.jpg", "data/driven_poses/s2-81"],
    # [1, "data/images/frame_51.png", "data/driven_poses/s1-81"],
    # [1, "data/images/002_00024_0007.png", "data/driven_poses/002_00024_0007"],
    # [1, "data/images/frame_52.png", "data/driven_poses/002_00026_0053"],
    # [1, "data/images/007_00090_0013.png", "data/driven_poses/007_00090_0013"],
    # [1, "data/images/007_00090_0024.png", "data/driven_poses/007_00090_0024"],
    # [1, "data/normal/4.jpg", "data/normal/seq"],
    [1, "data/data/data0118/normal/814f8c98-4d1f-454e-aabf-6001e640dc29.png", "./outputs515-data0118/frames_480P_814f8c98-4d1f-454e-aabf-6001e640dc29.jpg_images/jpg"],
    # [1, "data/images/sifudata2/1.jpg", "data/sifudata2/1/temp/smplx/normals"],
    # [1, "data/images/sifudata2/2.jpg", "data/sifudata2/2/temp/smplx/normals"],
    # [1, "data/images/sifudata2/3.jpg", "data/sifudata2/3/temp/smplx/normals"],
    # [1, "data/images/sifudata2/4.jpg", "data/sifudata2/4/temp/smplx/normals"],
    # [1, "data/images/sifudata2/5.jpg", "data/sifudata2/5/temp/smplx/normals"],
    # [1, "data/images/sifudata2/6.jpg", "data/sifudata2/6/temp/smplx/normals"],
    # [1, "data/images/sifudata2/7.jpg", "data/sifudata2/7/temp/smplx/normals"],
    # [1, "data/images/sifudata2/8.jpg", "data/sifudata2/8/temp/smplx/normals"],
    # # [1, "data/images/sifudata2/9.jpg", "data/sifudata2/9/temp/smplx/normals"],
    # [1, "data/images/sifudata2/10.jpg", "data/sifudata2/10/temp/smplx/normals"],
    # [1, "data/images/sifudata2/11.jpg", "data/sifudata2/11/temp/smplx/normals"],
    # [1, "data/images/sifudata2/12.jpg", "data/sifudata2/12/temp/smplx/normals"],
    # [1, "data/images/sifudata2/13.jpg", "data/sifudata2/13/temp/smplx/normals"],
    # [1, "data/images/sifudata2/14.jpg", "data/sifudata2/14/temp/smplx/normals"],
    # [1, "data/images/sifudata2/15.jpg", "data/sifudata2/15/temp/smplx/normals"],
    # [1, "data/images/sifudata2/trump_processed.png", "data/sifudata2/trump/temp/smplx/normals"],
]

xxx = "test1104"
image_dir = f"data/data/{xxx}"
pose_base_dir = f"outputs0826-1-out10-8-7-12000-{xxx}"
# pose_base_dir = f"outputs0826-1-out10-2-18-30000-{xxx}"

test_list_path = []

# Get all image files in the directory
for img_file in sorted(os.listdir(image_dir)):
    if img_file.lower().endswith('.png') or img_file.lower().endswith('.jpg') or img_file.lower().endswith('.jpeg'):
        
        base_name = os.path.splitext(img_file)[0]
        img_path = f"{image_dir}/normal/{base_name}.png"
        
        # 普通文件直接使用文件名作为目录名
        pose_dir = f"{pose_base_dir}/frames_480P_{base_name}"
        if os.path.exists(pose_dir):
            # if original version 480*480 need the following two lines
            # pad_480x480_to_832x480(pose_dir)
            # pose_dir = f"{pose_base_dir}/frames_480P_{img_file}_images/jpg"
            if not os.path.exists(os.path.join(pose_dir, "pose.jpg")):
                print(f"Pose image does not exist: {pose_dir}")
                # 使用pose_dir文件夹中的第一个图片复制到pose.jpg
                pose_imgs = os.path.join(pose_dir, os.listdir(pose_dir)[0])
                pose_imgt = os.path.join(pose_dir, "pose.jpg")
                cv2.imwrite(pose_imgt, cv2.imread(pose_imgs))
            
            test_list_path.append([1, img_path, pose_dir])
print(test_list_path)
print(len(test_list_path))
# test_list_path = test_list_path[19:]
# cc()
misc_size = [height,width]

# Download models
# snapshot_download("Wan-AI/Wan2.1-I2V-14B-720P", local_dir="./Wan2.1-I2V-14B-720P")

# Load models
model_manager = ModelManager(device="cpu")
model_manager.load_models(
    ["./Wan2.1-I2V-14B-720P/models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth"],
    torch_dtype=torch.float32, # Image Encoder is loaded with float32
)
model_manager.load_models(
    [
        [
            
            "./Wan2.1-I2V-14B-720P/diffusion_pytorch_model-00001-of-00007.safetensors",
            "./Wan2.1-I2V-14B-720P/diffusion_pytorch_model-00002-of-00007.safetensors",
            "./Wan2.1-I2V-14B-720P/diffusion_pytorch_model-00003-of-00007.safetensors",
            "./Wan2.1-I2V-14B-720P/diffusion_pytorch_model-00004-of-00007.safetensors",
            "./Wan2.1-I2V-14B-720P/diffusion_pytorch_model-00005-of-00007.safetensors",
            "./Wan2.1-I2V-14B-720P/diffusion_pytorch_model-00006-of-00007.safetensors",
            "./Wan2.1-I2V-14B-720P/diffusion_pytorch_model-00007-of-00007.safetensors",

        ],
        "./Wan2.1-I2V-14B-720P/models_t5_umt5-xxl-enc-bf16.pth",
        "./Wan2.1-I2V-14B-720P/Wan2.1_VAE.pth",
    ],
    torch_dtype=torch.bfloat16, # You can set `torch_dtype=torch.float8_e4m3fn` to enable FP8 quantization.
)

# rgb
# chpt_folder = 'models_out7'
# version_id = 'version_3'
# ckpt_id = 'epoch=4-step=7000.ckpt'

# normal
chpt_folder = 'models_out10_normal'
version_id = 'version_5' # 1
ckpt_id = 'epoch=16-step=27000.ckpt' # 7 12000

save_dir = "./outputs0902-normal-" + \
    "-" + chpt_folder.split('_')[1] + \
    "-" + version_id.split('_')[1] + \
    "-" + ckpt_id.split('=')[1].split('-')[0] + \
    "-" + ckpt_id.split('=')[2].split('.')[0] + "-" + xxx

# if you use deepspeed to train UniAnimate-Wan2.1, multiple checkpoints may be need to load, use the following form:
model_manager.load_lora_v2([
            "./"+chpt_folder+"/lightning_logs/"+version_id+"/checkpoints/"+ckpt_id+"/output_dir/model-00001-of-00011.safetensors",
            "./"+chpt_folder+"/lightning_logs/"+version_id+"/checkpoints/"+ckpt_id+"/output_dir/model-00002-of-00011.safetensors",
            "./"+chpt_folder+"/lightning_logs/"+version_id+"/checkpoints/"+ckpt_id+"/output_dir/model-00003-of-00011.safetensors",
            "./"+chpt_folder+"/lightning_logs/"+version_id+"/checkpoints/"+ckpt_id+"/output_dir/model-00004-of-00011.safetensors",
            "./"+chpt_folder+"/lightning_logs/"+version_id+"/checkpoints/"+ckpt_id+"/output_dir/model-00005-of-00011.safetensors",
            "./"+chpt_folder+"/lightning_logs/"+version_id+"/checkpoints/"+ckpt_id+"/output_dir/model-00006-of-00011.safetensors",
            "./"+chpt_folder+"/lightning_logs/"+version_id+"/checkpoints/"+ckpt_id+"/output_dir/model-00007-of-00011.safetensors",
            "./"+chpt_folder+"/lightning_logs/"+version_id+"/checkpoints/"+ckpt_id+"/output_dir/model-00008-of-00011.safetensors",
            "./"+chpt_folder+"/lightning_logs/"+version_id+"/checkpoints/"+ckpt_id+"/output_dir/model-00009-of-00011.safetensors",
            "./"+chpt_folder+"/lightning_logs/"+version_id+"/checkpoints/"+ckpt_id+"/output_dir/model-00010-of-00011.safetensors",
            "./"+chpt_folder+"/lightning_logs/"+version_id+"/checkpoints/"+ckpt_id+"/output_dir/model-00011-of-00011.safetensors",
            ], lora_alpha=1.0)

pipe = WanUniAnimateVideoPipeline.from_model_manager(model_manager, torch_dtype=torch.bfloat16, device="cuda")
pipe.enable_vram_management(num_persistent_param_in_dit=6*10**9) # You can set `num_persistent_param_in_dit` to a small number to reduce VRAM required.

def crop_center_opencv(image_path, output_path):
    try:
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError("无法加载图片，请检查路径")
            
        h, w = img.shape[:2]  # h = 高度（480），w = 宽度（832）
        
        # 检查尺寸是否符合预期（832x480）
        if w != 832 or h != 480:
            print(f"警告：图片尺寸不是832x480，当前尺寸为{w}x{h}")
        
        # 计算裁剪区域（宽度从中心裁剪480px，高度保持不变）
        start_x = (w - 480) // 2  # (832 - 480) / 2 = 176
        cropped_img = img[:, start_x : start_x + 480]  # 保持全部高度，裁剪宽度
        
        cv2.imwrite(output_path, cropped_img)
        print(f"成功保存裁剪后的图片到: {output_path}")
        
    except Exception as e:
        print(f"处理图片时出错: {e}")

def resize(image):
    
    image = torchvision.transforms.functional.resize(
        image,
        (height, width),
        interpolation=torchvision.transforms.InterpolationMode.BILINEAR
    )
    return torch.from_numpy(np.array(image))


for path_dir_per in test_list_path:
    sample_fps = path_dir_per[0] # frame interval for sampling
    pose_file_path = path_dir_per[2]
    ref_image_path = path_dir_per[1]
    if not os.path.exists(ref_image_path) or not os.path.exists(pose_file_path):
        print(f"文件不存在: {ref_image_path} 或 {pose_file_path}")
        continue
    if os.path.exists(save_dir + "/video_480P_{}_{}.mp4".format(ref_image_path.split('/')[-1], pose_file_path.split('/')[-1])):
        continue

    dwpose_all = {}
    frames_all = {}
    
    for ii_index in sorted(os.listdir(pose_file_path)):
        if ii_index != "pose.jpg":
            dwpose_all[ii_index] = Image.open(os.path.join(pose_file_path, ii_index))
            frames_all[ii_index] = Image.fromarray(cv2.cvtColor(cv2.imread(ref_image_path), cv2.COLOR_BGR2RGB))

    frames_pose_ref = Image.open(os.path.join(pose_file_path, "pose.jpg"))

    stride = sample_fps
    _total_frame_num = len(frames_all)
    cover_frame_num = (stride * max_frames)
    
    if _total_frame_num < cover_frame_num + 1:
        start_frame = 0
        end_frame = _total_frame_num-1
        stride = max((_total_frame_num//max_frames),1)
        end_frame = min(stride*max_frames, _total_frame_num)
    else:
        start_frame = random.randint(0, _total_frame_num-cover_frame_num-1)
        end_frame = start_frame + cover_frame_num
    frame_list = []
    dwpose_list = []

    random_ref_frame = frames_all[list(frames_all.keys())[0]]
    if random_ref_frame.mode != 'RGB':
        random_ref_frame = random_ref_frame.convert('RGB')
    random_ref_dwpose = frames_pose_ref
    if random_ref_dwpose.mode != 'RGB':
        random_ref_dwpose = random_ref_dwpose.convert('RGB')
    
    # sample pose sequence
    for i_index in range(start_frame, end_frame, stride):
        if i_index < len(frames_all):  # Check index within bounds
            i_key = list(frames_all.keys())[i_index]
            i_frame = frames_all[i_key]
            if i_frame.mode != 'RGB':
                i_frame = i_frame.convert('RGB')
            
            i_dwpose = dwpose_all[i_key]
            if i_dwpose.mode != 'RGB':
                i_dwpose = i_dwpose.convert('RGB')
            frame_list.append(i_frame)
            dwpose_list.append(i_dwpose)
    
    if (end_frame-start_frame) < max_frames:
        for _ in range(max_frames-(end_frame-start_frame)):
            i_key = list(frames_all.keys())[end_frame-1]
            
            i_frame = frames_all[i_key]
            if i_frame.mode != 'RGB':
                i_frame = i_frame.convert('RGB')
            i_dwpose = dwpose_all[i_key]
            
    
            frame_list.append(i_frame)
            dwpose_list.append(i_dwpose)

    have_frames = len(frame_list)>0
    middle_indix = 0

    if have_frames:

        l_hight = random_ref_frame.size[1]
        l_width = random_ref_frame.size[0]
        
        ref_frame = random_ref_frame 
        
        random_ref_frame_tmp = torchvision.transforms.functional.resize(
            random_ref_frame,
            (height, width),
            interpolation=torchvision.transforms.InterpolationMode.BILINEAR
        )
        random_ref_dwpose_tmp = resize(random_ref_dwpose) 
        
        dwpose_data_tmp = torch.stack([resize(ss).permute(2,0,1) for ss in dwpose_list], dim=0)

    
    dwpose_data = torch.zeros(max_frames, 3, misc_size[0], misc_size[1])

    if have_frames:
        
        dwpose_data[:len(frame_list), ...] = dwpose_data_tmp
        
    
    dwpose_data = dwpose_data.permute(1,0,2,3)

    def image_compose_width(imag, imag_1):
        # read the size of image1
        rom_image = imag
        width, height = imag.size
        # read the size of image2
        rom_image_1 = imag_1
        
        width1 = rom_image_1.size[0]
        # create a new image
        to_image = Image.new('RGB', (width+width1, height))
        # paste old images
        to_image.paste(rom_image, (0, 0))
        to_image.paste(rom_image_1, (width, 0))
        return to_image

    caption = "A camera smoothly circles around the person horizontally, completing a full rotation, capturing them from all sides. The person remains still throughout the video."
    video_out_condition = []
    for ii in range(dwpose_data_tmp.shape[0]):
        ss = dwpose_list[ii]
        video_out_condition.append(image_compose_width(random_ref_frame_tmp, torchvision.transforms.functional.resize(
            ss,
            (height, width),
            interpolation=torchvision.transforms.InterpolationMode.BILINEAR
        )))

    # Image-to-video
    video = pipe(
        prompt="A camera smoothly circles around the person horizontally, completing a full rotation, capturing them from all sides. The person remains still throughout the video.",
        negative_prompt="细节模糊不清，字幕，作品，画作，画面，静止，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，杂乱的背景，三条腿，背景人很多，倒着走",
        input_image=ref_frame,
        num_inference_steps=50,
        cfg_scale=1.5, # slow
        # cfg_scale=1.0, # fast
        seed=seed, tiled=True,
        dwpose_data=dwpose_data,
        random_ref_dwpose=random_ref_dwpose_tmp,
        height=height,
        width=width,
        tea_cache_l1_thresh=0.3 if use_teacache else None,
        tea_cache_model_id="Wan2.1-I2V-14B-720P" if use_teacache else None,

    )

    video_out = []
    # frames_savepath = os.path.join(save_dir, "frames_480P_{}_{}".format(ref_image_path.split('/')[-1], pose_file_path.split('/')[-1]))
    frames_savepath = os.path.join(save_dir, "frames_480P_{}".format(ref_image_path.split('/')[-1].split('.')[0]))
    os.makedirs(frames_savepath, exist_ok=True)
    for ii in range(len(video)):
        ss = video[ii]
        ss.save(os.path.join(frames_savepath, f"frame_{ii:02d}.png"))
        crop_center_opencv(os.path.join(frames_savepath, f"frame_{ii:02d}.png"), os.path.join(frames_savepath, f"frame_{ii:02d}.png"))
        video_out.append(image_compose_width(video_out_condition[ii], ss))
    os.makedirs(save_dir, exist_ok=True)
    # save_video(video_out, save_dir + "/video_480P_{}_{}.mp4".format(ref_image_path.split('/')[-1], pose_file_path.split('/')[-1]), fps=15, quality=5,save_frames_dir=None) #os.path.join(save_dir, "frames_480P_{}_{}".format(ref_image_path.split('/')[-1], pose_file_path.split('/')[-1])) ) # 720P
    save_video(video_out, save_dir + "/video_480P_{}.mp4".format(ref_image_path.split('/')[-1].split('.')[0]), fps=15, quality=5,save_frames_dir=None)
    # CUDA_VISIBLE_DEVICES="0" python examples/unianimate_wan/inference_unianimate_wan_480p.py