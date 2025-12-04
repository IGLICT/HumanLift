import os
from PIL import Image, ImageOps

def crop_center_832x480_to_480x480(folder_path):
    supported_formats = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp')
    
    for filename in os.listdir(folder_path):
        if filename.lower().endswith(supported_formats):
            file_path = os.path.join(folder_path, filename)
            
            try:
                img = Image.open(file_path)
                width, height = img.size
                
                if width == 832 and height == 480:
                    # 计算裁剪区域（左右各裁剪 (832-480)/2 = 176）
                    left = (width - 480) // 2  # (832-480)/2 = 176
                    upper = 0
                    right = left + 480
                    lower = height
                    
                    # 裁剪并覆盖原文件
                    img.crop((left, upper, right, lower)).save(file_path)
                    print(f"✅ 已处理: {filename}")
                else:
                    print(f"⚠️ 跳过: {filename}（尺寸不符，需 832x480，实际 {width}x{height}）")
                    
            except Exception as e:
                print(f"❌ 处理 {filename} 时出错: {str(e)}")

def pad_480x480_to_832x480(folder_path):
    """
    将480x480的图像两侧填充白色，变成832x480
    
    Args:
        folder_path: 包含图片的文件夹路径
    """
    supported_formats = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.webp')
    
    for filename in os.listdir(folder_path):
        if filename.lower().endswith(supported_formats):
            file_path = os.path.join(folder_path, filename)
            
            try:
                img = Image.open(file_path)
                width, height = img.size
                
                if width == 480 and height == 480:
                    # 计算需要填充的宽度 (832-480)/2 = 176 每边
                    padding = (176, 0, 176, 0)  # 左、上、右、下
                    
                    # 创建白色背景的新图像
                    new_img = ImageOps.expand(img, padding, fill='white')
                    
                    # 覆盖原文件
                    if not os.path.exists(folder_path+'/jpg'):
                        os.makedirs(folder_path+'/jpg', exist_ok=True)
                    new_img.save(os.path.join(folder_path+'/jpg', filename.split('.')[0]+'.jpg'))
                    print(f"✅ 已处理: {filename} (480x480 → 832x480)")
                elif width == 832 and height == 480:
                    if not os.path.exists(folder_path+'/jpg'):
                        os.makedirs(folder_path+'/jpg', exist_ok=True)
                    img.save(os.path.join(folder_path+'/jpg', filename.split('.')[0]+'.jpg'))
                    print(f"✅ 已处理: {filename} (不需要填充，已保存为832x480)")
                else:
                    print(f"⚠️ 跳过: {filename}（尺寸不符，需 480x480，实际 {width}x{height}）")
                    
            except Exception as e:
                print(f"❌ 处理 {filename} 时出错: {str(e)}")


import os
import cv2
import numpy as np
from PIL import Image

def video_to_frames(video_path, output_folder='frames', target_size=(832, 480)):
    """
    Convert video to JPG frames with specific requirements
    
    Args:
        video_path: Path to input video file
        output_folder: Folder to save frames (default: 'frames')
        target_size: Target resolution (width, height) (default: 832x480)
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)
    
    # Open video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"❌ Error opening video file: {video_path}")
        return
    
    frame_count = 0
    first_frame = None
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Convert BGR (OpenCV) to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(frame_rgb)
        
        # Process image size
        if pil_img.size != target_size:
            pil_img = resize_with_padding(pil_img, target_size)
        
        # Save frame
        frame_filename = os.path.join(output_folder, f'frame_{frame_count:02d}.jpg')
        pil_img.save(frame_filename, quality=95)
        
        # Keep first frame for pose.jpg
        if frame_count == 0:
            first_frame = pil_img
        
        frame_count += 1
    
    cap.release()
    
    # Save pose.jpg (copy of first frame)
    if first_frame is not None:
        pose_path = os.path.join(output_folder, 'pose.jpg')
        first_frame.save(pose_path, quality=95)
        print(f"✅ Saved pose.jpg from first frame")
    
    print(f"✅ Converted {frame_count} frames to JPG in '{output_folder}'")

def resize_with_padding(img, target_size, bg_color=(0, 0, 0)):
    """
    Resize image with padding to maintain aspect ratio
    
    Args:
        img: PIL Image object
        target_size: (width, height) tuple
        bg_color: Background color (default: black)
    
    Returns:
        Resized PIL Image with padding
    """
    target_w, target_h = target_size
    img_w, img_h = img.size
    
    # Calculate scaling ratio
    ratio = min(target_w / img_w, target_h / img_h)
    new_w = int(img_w * ratio)
    new_h = int(img_h * ratio)
    
    # Resize image
    img = img.resize((new_w, new_h), Image.LANCZOS)
    
    # Create new image with target size and background color
    new_img = Image.new('RGB', (target_w, target_h), bg_color)
    
    # Calculate position to center the image
    x = (target_w - new_w) // 2
    y = (target_h - new_h) // 2
    
    # Paste the resized image onto the center
    new_img.paste(img, (x, y))
    
    return new_img

# Example usage:
# video_to_frames('smplimages3.mp4', './smplimage3')

# if __name__ == "__main__":
#     folder_path = "./outputs514-sifudata2/frames_480P_4.jpg_images/"
    
#     pad_480x480_to_832x480(folder_path)
# pad_480x480_to_832x480('./outputs520-comp/frames_480P_test.png_images')