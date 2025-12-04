import os
# os.environ['OPENBLAS_NUM_THREADS'] = '1'
import torch, imageio, argparse
from torchvision.transforms import v2
from einops import rearrange
import lightning as pl
import pandas as pd
from diffsynth import WanVideoPipeline, ModelManager, load_state_dict, load_state_dict_from_folder
from peft import LoraConfig, inject_adapter_in_model
import torchvision
from PIL import Image
import numpy as np
import random
import pickle
from io import BytesIO
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image, ImageFilter
import  torch.nn  as nn
os.environ["TOKENIZERS_PARALLELISM"] = "false"


class TextVideoDataset(torch.utils.data.Dataset):
    def __init__(self, base_path, metadata_path, max_num_frames=81, frame_interval=1, num_frames=81, height=480, width=832, is_i2v=False):
        metadata = pd.read_csv(metadata_path)
        self.path = [os.path.join(base_path, "train", file_name) for file_name in metadata["file_name"]]
        self.text = metadata["text"].to_list()
        
        self.max_num_frames = max_num_frames
        self.frame_interval = frame_interval
        self.num_frames = num_frames
        self.height = height
        self.width = width
        self.is_i2v = is_i2v
            
        self.frame_process = v2.Compose([
            v2.CenterCrop(size=(height, width)),
            v2.Resize(size=(height, width), antialias=True),
            v2.ToTensor(),
            v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])
        
        
    def crop_and_resize(self, image):
        width, height = image.size
        scale = max(self.width / width, self.height / height)
        image = torchvision.transforms.functional.resize(
            image,
            (round(height*scale), round(width*scale)),
            interpolation=torchvision.transforms.InterpolationMode.BILINEAR
        )
        return image

    def resize(self, image):
        width, height = image.size
        # scale = max(self.width / width, self.height / height)
        image = torchvision.transforms.functional.resize(
            image,
            (self.height, self.width),
            interpolation=torchvision.transforms.InterpolationMode.BILINEAR
        )
        return torch.from_numpy(np.array(image))


    def load_frames_using_imageio(self, file_path, max_num_frames, start_frame_id, interval, num_frames, frame_process):
        reader = imageio.get_reader(file_path)
        if reader.count_frames() < max_num_frames or reader.count_frames() - 1 < start_frame_id + (num_frames - 1) * interval:
            reader.close()
            return None
        
        frames = []
        first_frame = None
        for frame_id in range(num_frames):
            frame = reader.get_data(start_frame_id + frame_id * interval)
            frame = Image.fromarray(frame)
            frame = self.crop_and_resize(frame)
            if first_frame is None:
                first_frame = np.array(frame)
            frame = frame_process(frame)
            frames.append(frame)
        reader.close()

        frames = torch.stack(frames, dim=0)
        frames = rearrange(frames, "T C H W -> C T H W")

        if self.is_i2v:
            return frames, first_frame
        else:
            return frames


    def load_video(self, file_path):
        start_frame_id = torch.randint(0, self.max_num_frames - (self.num_frames - 1) * self.frame_interval, (1,))[0]
        frames = self.load_frames_using_imageio(file_path, self.max_num_frames, start_frame_id, self.frame_interval, self.num_frames, self.frame_process)
        return frames
    
    
    def is_image(self, file_path):
        file_ext_name = file_path.split(".")[-1]
        if file_ext_name.lower() in ["jpg", "jpeg", "png", "webp"]:
            return True
        return False
    
    
    def load_image(self, file_path):
        frame = Image.open(file_path).convert("RGB")
        frame = self.crop_and_resize(frame)
        first_frame = frame
        frame = self.frame_process(frame)
        frame = rearrange(frame, "C H W -> C 1 H W")
        return frame


    def __getitem__(self, data_id):
        text = self.text[data_id]
        path = self.path[data_id]
        if self.is_image(path):
            if self.is_i2v:
                raise ValueError(f"{path} is not a video. I2V model doesn't support image-to-image training.")
            video = self.load_image(path)
        else:
            video = self.load_video(path)
        if self.is_i2v:
            video, first_frame = video
            data = {"text": text, "video": video, "path": path, "first_frame": first_frame}
        else:
            data = {"text": text, "video": video, "path": path}
        return data
    

    def __len__(self):
        return len(self.path)


class TextVideoDataset_onestage1(torch.utils.data.Dataset):
    def __init__(self, base_path, metadata_path, max_num_frames=81, frame_interval=2, num_frames=81, height=480, width=832, is_i2v=True,steps_per_epoch=1,typea='images',typeb='images',caption='a person is dancing'):
        # metadata = pd.read_csv(metadata_path)
        # self.path = [os.path.join(base_path, "train", file_name) for file_name in metadata["file_name"]]
        # self.text = metadata["text"].to_list()
        
        self.max_num_frames = max_num_frames
        self.frame_interval = frame_interval
        self.num_frames = num_frames
        self.height = height
        self.width = width
        self.is_i2v = is_i2v
        self.steps_per_epoch = steps_per_epoch
        self.typea = typea
        self.typeb = typeb
        self.caption = caption
        
        # data_list = ['UBC_Fashion', 'self_collected_videos_pose', 'TikTok']
        data_list = ['self_collected_videos_pose']
        self.sample_fps = frame_interval
        self.max_frames = max_num_frames
        self.misc_size = [height, width]
        self.video_list = []

        
        if 'TikTok' in data_list:
            
            self.pose_dir = "./data/example_dataset/TikTok/"
            file_list = os.listdir(self.pose_dir)
            print("!!! all dataset length: ", len(file_list))
            # 
            for iii_index in file_list:
                    self.video_list.append(self.pose_dir+iii_index)

            self.use_pose = True
            print("!!! dataset length: ", len(self.video_list))
        
        if 'UBC_Fashion' in data_list:
            self.pose_dir = "path_of_UBC_Fashion"
            file_list = os.listdir(self.pose_dir)
            print("!!! all dataset length (UBC_Fashion): ", len(file_list))
            
            for iii_index in file_list:
            #     
                    self.video_list.append(self.pose_dir + iii_index)

            self.use_pose = True
            print("!!! dataset length: ", len(self.video_list))
        if 'self_collected_videos_pose' in data_list:
            
            self.pose_dir = "/home/jovyan/data2/yangjie/human4dit-3d-81-orth-pkl-2/"
            file_list = os.listdir(self.pose_dir)
            print("!!! all dataset length (self_collected_videos_pose): ", len(file_list))
            # 
            for iii_index in file_list:
                
                self.video_list.append(self.pose_dir+iii_index)
            
            self.pose_dir = "/home/jovyan/data3/yangjie/2k2k-3d-81-orth-pkl/"
            file_list = os.listdir(self.pose_dir)
            print("!!! all dataset length (self_collected_videos_pose): ", len(file_list))
            # 
            for iii_index in file_list:
                
                self.video_list.append(self.pose_dir+iii_index)

            self.use_pose = True
            print("!!! dataset length: ", len(self.video_list))

        random.shuffle(self.video_list)
            
        self.frame_process = v2.Compose([
            # v2.CenterCrop(size=(height, width)),
            v2.Resize(size=(height, width), antialias=True),
            v2.ToTensor(),
            v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])

    def resize(self, image):
        width, height = image.size
        # print(width, height)
        # 
        image = torchvision.transforms.functional.resize(
            image,
            (self.height, self.width),
            interpolation=torchvision.transforms.InterpolationMode.BILINEAR
        )
        # print(image.size)
        # return torch.from_numpy(np.array(image))
        return image
        
    def crop_and_resize(self, image):
        width, height = image.size
        scale = max(self.width / width, self.height / height)
        image = torchvision.transforms.functional.resize(
            image,
            (round(height*scale), round(width*scale)),
            interpolation=torchvision.transforms.InterpolationMode.BILINEAR
        )
        return image


    def load_frames_using_imageio(self, file_path, max_num_frames, start_frame_id, interval, num_frames, frame_process):
        reader = imageio.get_reader(file_path)
        if reader.count_frames() < max_num_frames or reader.count_frames() - 1 < start_frame_id + (num_frames - 1) * interval:
            reader.close()
            return None
        
        frames = []
        first_frame = None
        for frame_id in range(num_frames):
            frame = reader.get_data(start_frame_id + frame_id * interval)
            frame = Image.fromarray(frame)
            frame = self.crop_and_resize(frame)
            if first_frame is None:
                first_frame = np.array(frame)
            frame = frame_process(frame)
            frames.append(frame)
        reader.close()

        frames = torch.stack(frames, dim=0)
        frames = rearrange(frames, "T C H W -> C T H W")

        if self.is_i2v:
            return frames, first_frame
        else:
            return frames


    def load_video(self, file_path):
        start_frame_id = torch.randint(0, self.max_num_frames - (self.num_frames - 1) * self.frame_interval, (1,))[0]
        frames = self.load_frames_using_imageio(file_path, self.max_num_frames, start_frame_id, self.frame_interval, self.num_frames, self.frame_process)
        return frames
    
    
    def is_image(self, file_path):
        file_ext_name = file_path.split(".")[-1]
        if file_ext_name.lower() in ["jpg", "jpeg", "png", "webp"]:
            return True
        return False
    
    
    def load_image(self, file_path):
        frame = Image.open(file_path).convert("RGB")
        frame = self.crop_and_resize(frame)
        first_frame = frame
        frame = self.frame_process(frame)
        frame = rearrange(frame, "C H W -> C 1 H W")
        return frame


    # def __getitem__(self, data_id):
    def __getitem__(self, index):
        index = index % len(self.video_list)
        success = False
        for _try in range(5):
            try:
                if _try >0:
                    
                    index = random.randint(1,len(self.video_list))
                    index = index % len(self.video_list)
                
                clean = True
                path_dir = self.video_list[index]

                frames_all = pickle.load(open(path_dir+'/'+self.typea+'.pkl','rb'))
                
                dwpose_all = pickle.load(open(path_dir+'/'+self.typeb+'.pkl','rb'))
                # 
                # random sample fps
                stride = random.randint(1, self.sample_fps)  
                
                _total_frame_num = len(frames_all)
                cover_frame_num = (stride * self.max_frames)
                max_frames = self.max_frames
                if _total_frame_num < cover_frame_num + 1:
                    start_frame = 0
                    end_frame = _total_frame_num-1
                    stride = max((_total_frame_num//max_frames),1)
                    end_frame = min(stride*max_frames, _total_frame_num-1)
                else:
                    start_frame = random.randint(0, _total_frame_num-cover_frame_num-5)
                    end_frame = start_frame + cover_frame_num
                frame_list = []
                dwpose_list = []

                random_ref = random.randint(0,_total_frame_num-1)
                i_key = list(frames_all.keys())[random_ref]
                random_ref_frame = Image.open(BytesIO(frames_all[i_key]))
                if random_ref_frame.mode != 'RGB':
                    random_ref_frame = random_ref_frame.convert('RGB')
                random_ref_dwpose = Image.open(BytesIO(dwpose_all[i_key]))
                
                first_frame = None
                for i_index in range(start_frame, end_frame, stride):
                    i_key = list(frames_all.keys())[i_index]
                    i_frame = Image.open(BytesIO(frames_all[i_key]))
                    if i_frame.mode != 'RGB':
                        i_frame = i_frame.convert('RGB')
                    i_dwpose = Image.open(BytesIO(dwpose_all[i_key]))
                    
                    if first_frame is None:
                        first_frame=i_frame

                        frame_list.append(i_frame)
                        dwpose_list.append(i_dwpose)

                    else:
                        frame_list.append(i_frame)
                        dwpose_list.append(i_dwpose)

                if (end_frame-start_frame) < max_frames:
                    for _ in range(max_frames-(end_frame-start_frame)):
                        i_key = list(frames_all.keys())[end_frame-1]
                        
                        i_frame = Image.open(BytesIO(frames_all[i_key]))
                        if i_frame.mode != 'RGB':
                            i_frame = i_frame.convert('RGB')
                        i_dwpose = Image.open(BytesIO(dwpose_all[i_key]))
                        
                        frame_list.append(i_frame)
                        dwpose_list.append(i_dwpose)

                have_frames = len(frame_list)>0
                middle_indix = 0

                if have_frames:

                    l_hight = random_ref_frame.size[1]
                    l_width = random_ref_frame.size[0]

                    # random crop
                    x1 = 0#random.randint(0, l_width//14)
                    x2 = 0#random.randint(0, l_width//14)
                    y1 = 0#random.randint(0, l_hight//14)
                    y2 = 0#random.randint(0, l_hight//14)
                    
                    
                    random_ref_frame = random_ref_frame.crop((x1, y1,l_width-x2, l_hight-y2))
                    ref_frame = random_ref_frame 
                    # 
                    
                    random_ref_frame_tmp = torch.from_numpy(np.array(self.resize(random_ref_frame)))
                    random_ref_dwpose_tmp = torch.from_numpy(np.array(self.resize(random_ref_dwpose.crop((x1, y1,l_width-x2, l_hight-y2))))) # [3, 512, 320]
                    
                    video_data_tmp = torch.stack([self.frame_process(self.resize(ss.crop((x1, y1,l_width-x2, l_hight-y2)))) for ss in frame_list], dim=0) # self.transforms(frames)
                    dwpose_data_tmp = torch.stack([torch.from_numpy(np.array(self.resize(ss.crop((x1, y1,l_width-x2, l_hight-y2))))).permute(2,0,1) for ss in dwpose_list], dim=0)

                video_data = torch.zeros(self.max_frames, 3, self.misc_size[0], self.misc_size[1])
                dwpose_data = torch.zeros(self.max_frames, 3, self.misc_size[0], self.misc_size[1])
                
                if have_frames:
                    video_data[:len(frame_list), ...] = video_data_tmp      
                    
                    dwpose_data[:len(frame_list), ...] = dwpose_data_tmp
                    
                video_data = video_data.permute(1,0,2,3)
                dwpose_data = dwpose_data.permute(1,0,2,3)
                
                # caption = "a person is dancing"
                 
                caption = self.caption#"A camera smoothly circles around the person horizontally, completing a full rotation, capturing them from all sides. The person remains still throughout the video."
                break
            except Exception as e:
                # 
                caption = self.caption#"A camera smoothly circles around the person horizontally, completing a full rotation, capturing them from all sides. The person remains still throughout the video."
                # 
                video_data = torch.zeros(3, self.max_frames, self.misc_size[0], self.misc_size[1])  
                random_ref_frame_tmp = torch.zeros(self.misc_size[0], self.misc_size[1], 3)
                vit_image = torch.zeros(3,self.misc_size[0], self.misc_size[1])
                
                dwpose_data = torch.zeros(3, self.max_frames, self.misc_size[0], self.misc_size[1])  
                # 
                random_ref_dwpose_data = torch.zeros(3, self.max_frames, self.misc_size[0], self.misc_size[1])  
                print('{} read video frame failed with error: {}'.format(path_dir, e))
                continue


        text = caption 
        path = path_dir 
        
        if self.is_i2v:
            video, first_frame = video_data, random_ref_frame_tmp
            data = {"text": text, "video": video, "path": path, "first_frame": first_frame, "dwpose_data": dwpose_data, "random_ref_dwpose_data": random_ref_dwpose_tmp}
        else:
            data = {"text": text, "video": video, "path": path}
        return data
    

    def __len__(self):
        
        return len(self.video_list)
 



class LightningModelForTrain_onestage(pl.LightningModule):
    def __init__(
        self,
        dit_path,
        learning_rate=1e-5,
        lora_rank=4, lora_alpha=4, train_architecture="lora", lora_target_modules="q,k,v,o,ffn.0,ffn.2", init_lora_weights="kaiming",
        use_gradient_checkpointing=True, use_gradient_checkpointing_offload=False,
        pretrained_lora_path=None,
        model_VAE=None,
        # 
    ):
        super().__init__()
        model_manager = ModelManager(torch_dtype=torch.bfloat16, device="cpu")
        if os.path.isfile(dit_path):
            model_manager.load_models([dit_path])
        else:
            dit_path = dit_path.split(",")
            model_manager.load_models([dit_path])
        
        self.pipe = WanVideoPipeline.from_model_manager(model_manager)
        self.pipe.scheduler.set_timesteps(1000, training=True)

        self.pipe_VAE = model_VAE.pipe.eval()
        self.tiler_kwargs = model_VAE.tiler_kwargs

        concat_dim = 4
        self.dwpose_embedding = nn.Sequential(
                    nn.Conv3d(3, concat_dim * 4, (3,3,3), stride=(1,1,1), padding=(1,1,1)),
                    nn.SiLU(),
                    nn.Conv3d(concat_dim * 4, concat_dim * 4, (3,3,3), stride=(1,1,1), padding=(1,1,1)),
                    nn.SiLU(),
                    nn.Conv3d(concat_dim * 4, concat_dim * 4, (3,3,3), stride=(1,1,1), padding=(1,1,1)),
                    nn.SiLU(),
                    nn.Conv3d(concat_dim * 4, concat_dim * 4, (3,3,3), stride=(1,2,2), padding=(1,1,1)),
                    nn.SiLU(),
                    nn.Conv3d(concat_dim * 4, concat_dim * 4, 3, stride=(2,2,2), padding=1),
                    nn.SiLU(),
                    nn.Conv3d(concat_dim * 4, concat_dim * 4, 3, stride=(2,2,2), padding=1),
                    nn.SiLU(),
                    nn.Conv3d(concat_dim * 4, 5120, (1,2,2), stride=(1,2,2), padding=0))

        randomref_dim = 20
        self.randomref_embedding_pose = nn.Sequential(
                    nn.Conv2d(3, concat_dim * 4, 3, stride=1, padding=1),
                    nn.SiLU(),
                    nn.Conv2d(concat_dim * 4, concat_dim * 4, 3, stride=1, padding=1),
                    nn.SiLU(),
                    nn.Conv2d(concat_dim * 4, concat_dim * 4, 3, stride=1, padding=1),
                    nn.SiLU(),
                    nn.Conv2d(concat_dim * 4, concat_dim * 4, 3, stride=2, padding=1),
                    nn.SiLU(),
                    nn.Conv2d(concat_dim * 4, concat_dim * 4, 3, stride=2, padding=1),
                    nn.SiLU(),
                    nn.Conv2d(concat_dim * 4, randomref_dim, 3, stride=2, padding=1),
                    
                    )
        self.freeze_parameters()

        # self.freeze_parameters()
        if train_architecture == "lora":
            self.add_lora_to_model(
                self.pipe.denoising_model(),
                lora_rank=lora_rank,
                lora_alpha=lora_alpha,
                lora_target_modules=lora_target_modules,
                init_lora_weights=init_lora_weights,
                pretrained_lora_path=pretrained_lora_path,
            )
        else:
            self.pipe.denoising_model().requires_grad_(True)
        
        self.learning_rate = learning_rate
        self.use_gradient_checkpointing = use_gradient_checkpointing
        self.use_gradient_checkpointing_offload = use_gradient_checkpointing_offload
        

        
        
    def freeze_parameters(self):
        # Freeze parameters
        self.pipe.requires_grad_(False)
        self.pipe.eval()
        self.pipe.denoising_model().train()
        self.pipe_VAE.requires_grad_(False)
        self.pipe_VAE.eval()
        self.randomref_embedding_pose.train()
        self.dwpose_embedding.train()
        
        
    def add_lora_to_model(self, model, lora_rank=4, lora_alpha=4, lora_target_modules="q,k,v,o,ffn.0,ffn.2", init_lora_weights="kaiming", pretrained_lora_path=None, state_dict_converter=None):
        # Add LoRA to UNet
        self.lora_alpha = lora_alpha
        if init_lora_weights == "kaiming":
            init_lora_weights = True
            
        lora_config = LoraConfig(
            r=lora_rank,
            lora_alpha=lora_alpha,
            init_lora_weights=init_lora_weights,
            target_modules=lora_target_modules.split(","),
        )
        model = inject_adapter_in_model(lora_config, model)
        for param in model.parameters():
            # Upcast LoRA parameters into fp32
            if param.requires_grad:
                param.data = param.to(torch.float32)
                
        # Lora pretrained lora weights
        if pretrained_lora_path is not None:
            # 
            try:
                state_dict = load_state_dict(pretrained_lora_path)
            except:
                state_dict = load_state_dict_from_folder(pretrained_lora_path)
            # 
            state_dict_new = {}
            state_dict_new_module = {}
            for key in state_dict.keys():
                
                if 'pipe.dit.' in key:
                    key_new = key.split("pipe.dit.")[1]
                    state_dict_new[key_new] = state_dict[key]
                if "dwpose_embedding" in key or "randomref_embedding_pose" in key:
                    state_dict_new_module[key] = state_dict[key]
            state_dict = state_dict_new
            state_dict_new = {}

            for key in state_dict_new_module:
                if "dwpose_embedding" in key:
                    state_dict_new[key.split("dwpose_embedding.")[1]] = state_dict_new_module[key]
            self.dwpose_embedding.load_state_dict(state_dict_new, strict=True)

            state_dict_new = {}
            for key in state_dict_new_module:
                if "randomref_embedding_pose" in key:
                    state_dict_new[key.split("randomref_embedding_pose.")[1]] = state_dict_new_module[key]
            self.randomref_embedding_pose.load_state_dict(state_dict_new,strict=True)

            if state_dict_converter is not None:
                state_dict = state_dict_converter(state_dict)
            missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
            all_keys = [i for i, _ in model.named_parameters()]
            num_updated_keys = len(all_keys) - len(missing_keys)
            num_unexpected_keys = len(unexpected_keys)
            print(f"{num_updated_keys} parameters are loaded from {pretrained_lora_path}. {num_unexpected_keys} parameters are unexpected.")
    
    

    def training_step(self, batch, batch_idx):
        # batch["dwpose_data"]/255.: [1, 3, 81, 832, 480], batch["random_ref_dwpose_data"]/255.: [1, 832, 480, 3]
        text, video, path = batch["text"][0], batch["video"], batch["path"][0]
        # 'A person is dancing',  [1, 3, 81, 832, 480], 'data/example_dataset/train/[DLPanda.com][]7309800480371133711.mp4'
        self.pipe_VAE.device = self.device
        dwpose_data = self.dwpose_embedding((torch.cat([batch["dwpose_data"][:,:,:1].repeat(1,1,3,1,1), batch["dwpose_data"]], dim=2)/255.).to(self.device))
        random_ref_dwpose_data = self.randomref_embedding_pose((batch["random_ref_dwpose_data"]/255.).to(torch.bfloat16).to(self.device).permute(0,3,1,2)).unsqueeze(2) # [1, 20, 104, 60]
        with torch.no_grad():
            if video is not None:
                # prompt
                prompt_emb = self.pipe_VAE.encode_prompt(text)
                # video
                video = video.to(dtype=self.pipe_VAE.torch_dtype, device=self.pipe_VAE.device)
                latents = self.pipe_VAE.encode_video(video, **self.tiler_kwargs)[0]
                # image
                if "first_frame" in batch: # [1, 853, 480, 3]
                    first_frame = Image.fromarray(batch["first_frame"][0].cpu().numpy())
                    _, _, num_frames, height, width = video.shape
                    image_emb = self.pipe_VAE.encode_image(first_frame, num_frames, height, width)
                else:
                    image_emb = {}
                
                batch = {"latents": latents.unsqueeze(0), "prompt_emb": prompt_emb, "image_emb": image_emb}
        
        # Data
        p1 = random.random()
        p = random.random()
        if p1 < 0.05:
            
            dwpose_data = torch.zeros_like(dwpose_data)
            random_ref_dwpose_data = torch.zeros_like(random_ref_dwpose_data)
        latents = batch["latents"].to(self.device)  # [1, 16, 21, 60, 104]
        prompt_emb = batch["prompt_emb"] # batch["prompt_emb"]["context"]:  [1, 1, 512, 4096]
        
        prompt_emb["context"] = prompt_emb["context"].to(self.device)
        image_emb = batch["image_emb"]
        if "clip_feature" in image_emb:
            image_emb["clip_feature"] = image_emb["clip_feature"].to(self.device) # [1, 257, 1280]
            if p < 0.1:
                image_emb["clip_feature"] = torch.zeros_like(image_emb["clip_feature"]) # [1, 257, 1280]
        if "y" in image_emb:
            
            if p < 0.1:
                image_emb["y"] = torch.zeros_like(image_emb["y"])
            image_emb["y"] = image_emb["y"].to(self.device) + random_ref_dwpose_data  # [1, 20, 21, 104, 60]
        
        condition =  dwpose_data
        # 
        condition = rearrange(condition, 'b c f h w -> b (f h w) c').contiguous()
        # Loss
        self.pipe.device = self.device
        noise = torch.randn_like(latents)
        timestep_id = torch.randint(0, self.pipe.scheduler.num_train_timesteps, (1,))
        timestep = self.pipe.scheduler.timesteps[timestep_id].to(dtype=self.pipe.torch_dtype, device=self.pipe.device)
        extra_input = self.pipe.prepare_extra_input(latents)
        noisy_latents = self.pipe.scheduler.add_noise(latents, noise, timestep)
        training_target = self.pipe.scheduler.training_target(latents, noise, timestep)

        # Compute loss
        noise_pred = self.pipe.denoising_model()(
            noisy_latents, timestep=timestep, **prompt_emb, **extra_input, **image_emb,
            use_gradient_checkpointing=self.use_gradient_checkpointing,
            use_gradient_checkpointing_offload=self.use_gradient_checkpointing_offload,
            add_condition = condition,
        )
        loss = torch.nn.functional.mse_loss(noise_pred.float(), training_target.float())
        loss = loss * self.pipe.scheduler.training_weight(timestep)

        # Record log
        self.log("train_loss", loss, prog_bar=True)
        return loss


    def configure_optimizers(self):
        # trainable_modules = filter(lambda p: p.requires_grad, self.pipe.denoising_model().parameters())
        # optimizer = torch.optim.AdamW(trainable_modules, lr=self.learning_rate)
        # return optimizer
        trainable_modules = [
            {'params': filter(lambda p: p.requires_grad, self.pipe.denoising_model().parameters())},
            {'params': self.dwpose_embedding.parameters()},
            {'params': self.randomref_embedding_pose.parameters()},
        ]
        optimizer = torch.optim.AdamW(trainable_modules, lr=self.learning_rate)
        return optimizer
    

    def on_save_checkpoint(self, checkpoint):
        checkpoint.clear()
        # trainable_param_names = list(filter(lambda named_param: named_param[1].requires_grad, self.pipe.denoising_model().named_parameters())) + \
        #                         list(filter(lambda named_param: named_param[1].requires_grad, self.dwpose_embedding.named_parameters())) + \
        #                         list(filter(lambda named_param: named_param[1].requires_grad, self.randomref_embedding_pose.named_parameters()))
        trainable_param_names = list(filter(lambda named_param: named_param[1].requires_grad, self.named_parameters())) 
        
        trainable_param_names = set([named_param[0] for named_param in trainable_param_names])
        # state_dict = self.pipe.denoising_model().state_dict()
        state_dict = self.state_dict()
        # state_dict.update()
        lora_state_dict = {}
        for name, param in state_dict.items():
            if name in trainable_param_names:
                lora_state_dict[name] = param
        checkpoint.update(lora_state_dict)



def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "--task",
        type=str,
        default="data_process",
        required=True,
        choices=["data_process", "train"],
        help="Task. `data_process` or `train`.",
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        default=None,
        required=True,
        help="The path of the Dataset.",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="./",
        help="Path to save the model.",
    )
    parser.add_argument(
        "--typea",
        type=str,
        default="frame_data_images",
        help="to data",
    )
    parser.add_argument(
        "--typeb",
        type=str,
        default="smplx_with_foot_wo_face_images",
        help="from data",
    )
    parser.add_argument(
        "--caption",
        type=str,
        default="a person is dancing",
        help="text description",
    )
    parser.add_argument(
        "--text_encoder_path",
        type=str,
        default="./Wan2.1-I2V-14B-720P/models_t5_umt5-xxl-enc-bf16.pth",
        help="Path of text encoder.",
    )
    parser.add_argument(
        "--image_encoder_path",
        type=str,
        default=None,
        help="Path of image encoder.",
    )
    parser.add_argument(
        "--vae_path",
        type=str,
        default="./Wan2.1-I2V-14B-720P/Wan2.1_VAE.pth",
        help="Path of VAE.",
    )
    parser.add_argument(
        "--dit_path",
        type=str,
        default=None,
        help="Path of DiT.",
    )
    parser.add_argument(
        "--tiled",
        # default=False,
        default=True,
        action="store_true",
        help="Whether enable tile encode in VAE. This option can reduce VRAM required.",
    )
    parser.add_argument(
        "--tile_size_height",
        type=int,
        default=34,
        help="Tile size (height) in VAE.",
    )
    parser.add_argument(
        "--tile_size_width",
        type=int,
        default=34,
        help="Tile size (width) in VAE.",
    )
    parser.add_argument(
        "--tile_stride_height",
        type=int,
        default=18,
        help="Tile stride (height) in VAE.",
    )
    parser.add_argument(
        "--tile_stride_width",
        type=int,
        default=16,
        help="Tile stride (width) in VAE.",
    )
    parser.add_argument(
        "--steps_per_epoch",
        type=int,
        default=500,
        help="Number of steps per epoch.",
    )
    parser.add_argument(
        "--num_frames",
        type=int,
        default=81,
        help="Number of frames.",
    )
    # parser.add_argument(
    #     "--height",
    #     type=int,
    #     default=480,
    #     help="Image height.",
    # )
    # parser.add_argument(
    #     "--width",
    #     type=int,
    #     default=832,
    #     help="Image width.",
    # )
    parser.add_argument(
        "--height",
        type=int,
        default=832,
        help="Image height.",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=480,
        help="Image width.",
    )
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=1,
        help="Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-5,
        help="Learning rate.",
    )
    parser.add_argument(
        "--accumulate_grad_batches",
        type=int,
        default=1,
        help="The number of batches in gradient accumulation.",
    )
    parser.add_argument(
        "--max_epochs",
        type=int,
        default=1,
        help="Number of epochs.",
    )
    parser.add_argument(
        "--lora_target_modules",
        type=str,
        default="q,k,v,o,ffn.0,ffn.2",
        help="Layers with LoRA modules.",
    )
    parser.add_argument(
        "--init_lora_weights",
        type=str,
        default="kaiming",
        choices=["gaussian", "kaiming"],
        help="The initializing method of LoRA weight.",
    )
    parser.add_argument(
        "--training_strategy",
        type=str,
        default="auto",
        choices=["auto", "deepspeed_stage_1", "deepspeed_stage_2", "deepspeed_stage_3"],
        help="Training strategy",
    )
    parser.add_argument(
        "--lora_rank",
        type=int,
        default=4,
        help="The dimension of the LoRA update matrices.",
    )
    parser.add_argument(
        "--lora_alpha",
        type=float,
        default=4.0,
        help="The weight of the LoRA update matrices.",
    )
    parser.add_argument(
        "--use_gradient_checkpointing",
        default=False,
        action="store_true",
        help="Whether to use gradient checkpointing.",
    )
    parser.add_argument(
        "--use_gradient_checkpointing_offload",
        default=False,
        action="store_true",
        help="Whether to use gradient checkpointing offload.",
    )
    parser.add_argument(
        "--train_architecture",
        type=str,
        default="lora",
        choices=["lora", "full"],
        help="Model structure to train. LoRA training or full training.",
    )
    parser.add_argument(
        "--pretrained_lora_path",
        type=str,
        default=None,
        help="Pretrained LoRA path. Required if the training is resumed.",
    )
    parser.add_argument(
        "--use_swanlab",
        default=False,
        action="store_true",
        help="Whether to use SwanLab logger.",
    )
    parser.add_argument(
        "--swanlab_mode",
        default=None,
        help="SwanLab mode (cloud or local).",
    )
    args = parser.parse_args()
    return args


def data_process(args):
    dataset = TextVideoDataset(
        args.dataset_path,
        os.path.join(args.dataset_path, "metadata.csv"),
        max_num_frames=args.num_frames,
        frame_interval=1,
        num_frames=args.num_frames,
        height=args.height,
        width=args.width,
        is_i2v=args.image_encoder_path is not None
    )
    dataloader = torch.utils.data.DataLoader(
        dataset,
        shuffle=False,
        batch_size=1,
        num_workers=args.dataloader_num_workers
    )
    model = LightningModelForDataProcess(
        text_encoder_path=args.text_encoder_path,
        image_encoder_path=args.image_encoder_path,
        vae_path=args.vae_path,
        tiled=args.tiled,
        tile_size=(args.tile_size_height, args.tile_size_width),
        tile_stride=(args.tile_stride_height, args.tile_stride_width),
    )
    trainer = pl.Trainer(
        accelerator="gpu",
        devices="auto",
        default_root_dir=args.output_path,
    )
    trainer.test(model, dataloader)



class LightningModelForDataProcess(pl.LightningModule):
    def __init__(self, text_encoder_path, vae_path, image_encoder_path=None, tiled=False, tile_size=(34, 34), tile_stride=(18, 16)):
        super().__init__()
        model_path = [text_encoder_path, vae_path]
        if image_encoder_path is not None:
            model_path.append(image_encoder_path)
        model_manager = ModelManager(torch_dtype=torch.bfloat16, device="cpu")
        model_manager.load_models(model_path)
        self.pipe = WanVideoPipeline.from_model_manager(model_manager)

        self.tiler_kwargs = {"tiled": tiled, "tile_size": tile_size, "tile_stride": tile_stride}
        
    def test_step(self, batch, batch_idx):
        text, video, path = batch["text"][0], batch["video"], batch["path"][0]
        
        self.pipe.device = self.device
        if video is not None:
            # prompt
            prompt_emb = self.pipe.encode_prompt(text)
            # video
            video = video.to(dtype=self.pipe.torch_dtype, device=self.pipe.device)
            latents = self.pipe.encode_video(video, **self.tiler_kwargs)[0]
            # image
            if "first_frame" in batch:
                first_frame = Image.fromarray(batch["first_frame"][0].cpu().numpy())
                _, _, num_frames, height, width = video.shape
                image_emb = self.pipe.encode_image(first_frame, num_frames, height, width)
            else:
                image_emb = {}
            data = {"latents": latents, "prompt_emb": prompt_emb, "image_emb": image_emb}
            torch.save(data, path + ".tensors.pth")


    
def train(args):
    dataset = TensorDataset(
        args.dataset_path,
        os.path.join(args.dataset_path, "metadata.csv"),
        steps_per_epoch=args.steps_per_epoch,
    )
    
    dataloader = torch.utils.data.DataLoader(
        dataset,
        shuffle=True,
        batch_size=1,
        num_workers=args.dataloader_num_workers
    )
    model_VAE = LightningModelForDataProcess(
        text_encoder_path=args.text_encoder_path,
        image_encoder_path=args.image_encoder_path,
        vae_path=args.vae_path,
        tiled=args.tiled,
        tile_size=(args.tile_size_height, args.tile_size_width),
        tile_stride=(args.tile_stride_height, args.tile_stride_width),
    )
    model = LightningModelForTrain(
        dit_path=args.dit_path,
        learning_rate=args.learning_rate,
        train_architecture=args.train_architecture,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_target_modules=args.lora_target_modules,
        init_lora_weights=args.init_lora_weights,
        use_gradient_checkpointing=args.use_gradient_checkpointing,
        use_gradient_checkpointing_offload=args.use_gradient_checkpointing_offload,
        pretrained_lora_path=args.pretrained_lora_path,
        model_VAE = model_VAE,
        
    )
    if args.use_swanlab:
        from swanlab.integration.pytorch_lightning import SwanLabLogger
        swanlab_config = {"UPPERFRAMEWORK": "DiffSynth-Studio"}
        swanlab_config.update(vars(args))
        swanlab_logger = SwanLabLogger(
            project="wan", 
            name="wan",
            config=swanlab_config,
            mode=args.swanlab_mode,
            logdir=os.path.join(args.output_path, "swanlog"),
        )
        logger = [swanlab_logger]
    else:
        logger = None
    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        accelerator="gpu",
        devices="auto",
        precision="bf16",
        strategy=args.training_strategy,
        default_root_dir=args.output_path,
        accumulate_grad_batches=args.accumulate_grad_batches,
        callbacks=[pl.pytorch.callbacks.ModelCheckpoint(save_top_k=-1)],
        logger=logger,
    )
    trainer.fit(model, dataloader)


def train_onestage(args):
    
    dataset = TextVideoDataset_onestage1(
        args.dataset_path,
        os.path.join(args.dataset_path, "metadata.csv"),
        max_num_frames=args.num_frames,
        frame_interval=1,
        num_frames=args.num_frames,
        height=args.height,
        width=args.width,
        is_i2v=args.image_encoder_path is not None,
        steps_per_epoch=args.steps_per_epoch,
        typea=args.typea,
        typeb=args.typeb,
        caption=args.caption
    )
    
    dataloader = torch.utils.data.DataLoader(
        dataset,
        shuffle=True,
        batch_size=1,
        num_workers=args.dataloader_num_workers
    )
    model_VAE = LightningModelForDataProcess(
        text_encoder_path=args.text_encoder_path,
        image_encoder_path=args.image_encoder_path,
        vae_path=args.vae_path,
        tiled=args.tiled,
        tile_size=(args.tile_size_height, args.tile_size_width),
        tile_stride=(args.tile_stride_height, args.tile_stride_width),
    )
    model = LightningModelForTrain_onestage(
        dit_path=args.dit_path,
        learning_rate=args.learning_rate,
        train_architecture=args.train_architecture,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_target_modules=args.lora_target_modules,
        init_lora_weights=args.init_lora_weights,
        use_gradient_checkpointing=args.use_gradient_checkpointing,
        use_gradient_checkpointing_offload=args.use_gradient_checkpointing_offload,
        pretrained_lora_path=args.pretrained_lora_path,
        model_VAE = model_VAE,
        
    )
    if args.use_swanlab:
        from swanlab.integration.pytorch_lightning import SwanLabLogger
        swanlab_config = {"UPPERFRAMEWORK": "DiffSynth-Studio"}
        swanlab_config.update(vars(args))
        swanlab_logger = SwanLabLogger(
            project="wan", 
            name="wan",
            config=swanlab_config,
            mode=args.swanlab_mode,
            logdir=os.path.join(args.output_path, "swanlog"),
        )
        logger = [swanlab_logger]
    else:
        logger = None
    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        accelerator="gpu",
        devices="auto",
        precision="bf16",
        strategy=args.training_strategy,
        default_root_dir=args.output_path,
        accumulate_grad_batches=args.accumulate_grad_batches,
        callbacks=[pl.pytorch.callbacks.ModelCheckpoint(save_top_k=-1, every_n_train_steps=3000)], # save checkpoints every_n_train_steps 
        logger=logger,
    )
    trainer.fit(model, dataloader)

if __name__ == '__main__':
    args = parse_args()
    if args.task == "data_process":
        data_process(args)
    elif args.task == "train":
        # support VAE and DiT in a single stage
        train_onestage(args)

