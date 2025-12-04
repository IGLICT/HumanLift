# # high-res human body rgb
CUDA_VISIBLE_DEVICES="4,5,6,7" python ./train_wan.py  \
   --task train   --train_architecture lora \
   --lora_rank 128 --lora_alpha 128 --num_frames 81  \
   --dataset_path /home/jovyan/data2/yangjie/human4dit-3d-81-orth-pkl-2/   \
   --output_path ./models_out10_1  --dit_path "./Wan2.1-I2V-14B-720P/diffusion_pytorch_model-00001-of-00007.safetensors,./Wan2.1-I2V-14B-720P/diffusion_pytorch_model-00002-of-00007.safetensors,./Wan2.1-I2V-14B-720P/diffusion_pytorch_model-00003-of-00007.safetensors,./Wan2.1-I2V-14B-720P/diffusion_pytorch_model-00004-of-00007.safetensors,./Wan2.1-I2V-14B-720P/diffusion_pytorch_model-00005-of-00007.safetensors,./Wan2.1-I2V-14B-720P/diffusion_pytorch_model-00006-of-00007.safetensors,./Wan2.1-I2V-14B-720P/diffusion_pytorch_model-00007-of-00007.safetensors"     \
   --max_epochs 20   --learning_rate 1e-4   \
   --accumulate_grad_batches 1   \
   --use_gradient_checkpointing \
   --image_encoder_path "./Wan2.1-I2V-14B-720P/models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth" \
   --use_gradient_checkpointing_offload \
   --training_strategy "deepspeed_stage_2" \
   --pretrained_lora_path="./checkpoints/Wan2.1-14B-Lora-12000.ckpt" \
   --typea "frame_data_images" \
   --typeb "smplx_with_foot_wo_face_images" \
   --caption "A camera smoothly circles around the person horizontally, completing a full rotation, capturing them from all sides. The person remains still throughout the video. Rendering quality is very high, with a focus on realistic details and textures. The background is a simple, white color to emphasize the person."


# # high-res human body normals
# CUDA_VISIBLE_DEVICES="4,5,6,7" python ./train_wan.py  \
#    --task train   --train_architecture lora \
#    --lora_rank 128 --lora_alpha 128 --num_frames 81  \
#    --dataset_path /home/jovyan/data2/yangjie/human4dit-3d-81-orth-pkl-2/   \
#    --output_path ./models_out10_normal1  --dit_path "./Wan2.1-I2V-14B-720P/diffusion_pytorch_model-00001-of-00007.safetensors,./Wan2.1-I2V-14B-720P/diffusion_pytorch_model-00002-of-00007.safetensors,./Wan2.1-I2V-14B-720P/diffusion_pytorch_model-00003-of-00007.safetensors,./Wan2.1-I2V-14B-720P/diffusion_pytorch_model-00004-of-00007.safetensors,./Wan2.1-I2V-14B-720P/diffusion_pytorch_model-00005-of-00007.safetensors,./Wan2.1-I2V-14B-720P/diffusion_pytorch_model-00006-of-00007.safetensors,./Wan2.1-I2V-14B-720P/diffusion_pytorch_model-00007-of-00007.safetensors"     \
#    --max_epochs 20   --learning_rate 1e-4   \
#    --accumulate_grad_batches 1   \
#    --use_gradient_checkpointing \
#    --image_encoder_path "./Wan2.1-I2V-14B-720P/models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth" \
#    --use_gradient_checkpointing_offload \
#    --training_strategy "deepspeed_stage_2" \
#    --pretrained_lora_path="./checkpoints/Wan2.1-14B-Lora-12000.ckpt" \
#    --typea "frame_data_normals" \
#    --typeb "frame_data_images" \
#    --caption "A camera smoothly circles around the person horizontally, completing a full rotation, capturing them from all sides. The person remains still throughout the video."

