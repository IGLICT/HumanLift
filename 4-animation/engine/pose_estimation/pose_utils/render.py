import os
import imageio
import numpy as np
import torch
from tqdm import tqdm
import torchvision
from PIL import ImageColor
from pytorch3d.renderer import (
    BlendParams,
    FoVOrthographicCameras,
    PerspectiveCameras,
    TexturesVertex,
    PointLights,
    Materials,
    RasterizationSettings,
    MeshRenderer,
    MeshRasterizer,
    blending,
    SoftPhongShader,
    look_at_view_transform,
)
import trimesh
from pytorch3d.renderer.mesh.shader import ShaderBase
from pytorch3d.structures import Meshes
from mmhuman3d.core.renderer.torch3d_renderer.meshes import ParametricMeshes
from mmhuman3d.core.visualization.visualize_smpl import _prepare_colors
def tensor_to_list(tensor):
    return tensor.squeeze().detach().cpu().numpy().tolist()
import matplotlib.pyplot as plt
class NormalShader(ShaderBase):
    def __init__(self, device = "cpu", **kwargs):
        super().__init__(device=device, **kwargs)

    def forward(self, fragments, meshes, **kwargs):
        blend_params = kwargs.get("blend_params", self.blend_params)
        texels = fragments.bary_coords.clone()
        texels = texels.permute(0, 3, 1, 2, 4)
        texels = texels * 2 - 1  # 将 bary_coords 映射到 [-1, 1]

        # 获取法线
        verts_normals = meshes.verts_normals_packed()
        faces_normals = verts_normals[meshes.faces_packed()]
        bary_coords = fragments.bary_coords

        pixel_normals = (bary_coords[..., None] * faces_normals[fragments.pix_to_face]).sum(dim=-2)
        pixel_normals = pixel_normals / pixel_normals.norm(dim=-1, keepdim=True)

        # 将法线映射到颜色空间
        # colors = (pixel_normals + 1) / 2  # 将法线映射到 [0, 1]
        colors = torch.clamp(pixel_normals, -1, 1)
        print(colors.shape)
        mask = (fragments.pix_to_face > 0).float()
        colors = torch.cat([colors, mask.unsqueeze(-1)], dim=-1)
        # colors[fragments.pix_to_face < 0] = 0

        # 混合颜色
        # images = self.blend(texels, colors, fragments, blend_params)
        return colors

def overlay_image_onto_background(image, mask, bbox, background):
    if isinstance(image, torch.Tensor):
        image = image.detach().cpu().numpy()
    if isinstance(mask, torch.Tensor):
        mask = mask.detach().cpu().numpy()
    out_image = np.zeros_like(background)  # 如果 background 是 (H,W,C)，则生成全0数组
    # out_image = background.copy()
    bbox = bbox[0].int().cpu().numpy().copy()
    roi_image = out_image[bbox[1]:bbox[3], bbox[0]:bbox[2]]
    if len(roi_image) < 1 or len(roi_image[1]) < 1:
        return out_image
    try:
        roi_image[mask] = image[mask]
    except Exception as e:
        raise e
    out_image[bbox[1]:bbox[3], bbox[0]:bbox[2]] = roi_image
    return out_image


def overlay_image_onto_background_rgba(image, mask, bbox, background):
    # 转为 NumPy 格式
    if isinstance(image, torch.Tensor):
        image = image.detach().cpu().numpy()
    if isinstance(mask, torch.Tensor):
        mask = mask.detach().cpu().numpy()
    if isinstance(background, torch.Tensor):
        background = background.detach().cpu().numpy()
    # 保证 image 是 (H, W, 3)
    if image.shape[-1] != 3:
        raise ValueError("image 必须是 RGB 图像 (H, W, 3)")
    
    # 创建 RGBA 背景 (全透明黑背景)
    h, w = background.shape[:2]
    out_image = np.zeros((h, w, 4), dtype=np.uint8)

    # 如果需要可以使用 background 作为底图
    # out_image[..., :3] = background  # 背景 RGB
    # out_image[..., 3] = 255  # 背景 alpha 全不透明（可选）

    # 处理 bounding box 和 ROI
    bbox = bbox[0].int().cpu().numpy().copy()
    x1, y1, x2, y2 = bbox

    # 边界检查
    if y2 <= y1 or x2 <= x1 or y2 > h or x2 > w:
        return out_image  # 空区域，返回全透明图

    roi_h = y2 - y1
    roi_w = x2 - x1

    # 确保 image 和 mask 大小匹配 bbox
    if image.shape[0] != roi_h or image.shape[1] != roi_w:
        raise ValueError("image 尺寸与 bbox 不一致")

    # 获取 ROI 区域
    roi_mask = mask.astype(bool)
    roi_image_rgb = image

    # 设置 ROI 区域 RGB
    out_image[y1:y2, x1:x2, :3][roi_mask] = roi_image_rgb[roi_mask]
    
    # 设置 ROI 区域 alpha
    out_image[y1:y2, x1:x2, 3][roi_mask[..., 0]] = 255  # mask 通常是 (H, W, 1) 或 (H, W)

    return out_image

def update_intrinsics_from_bbox(K_org, bbox):
    '''
    update intrinsics for cropped images
    '''
    device, dtype = K_org.device, K_org.dtype
    
    K = torch.zeros((K_org.shape[0], 4, 4)
    ).to(device=device, dtype=dtype)
    K[:, :3, :3] = K_org.clone()
    K[:, 2, 2] = 0
    K[:, 2, -1] = 1
    K[:, -1, 2] = 1
    
    image_sizes = []
    for idx, bbox in enumerate(bbox):
        left, upper, right, lower = bbox
        cx, cy = K[idx, 0, 2], K[idx, 1, 2]

        new_cx = cx - left
        new_cy = cy - upper
        new_height = max(lower - upper, 1)
        new_width = max(right - left, 1)
        new_cx = new_width - new_cx
        new_cy = new_height - new_cy

        K[idx, 0, 2] = new_cx
        K[idx, 1, 2] = new_cy
        image_sizes.append((int(new_height), int(new_width)))

    return K, image_sizes


def perspective_projection(x3d, K, R=None, T=None):
    if R != None:
        x3d = torch.matmul(R, x3d.transpose(1, 2)).transpose(1, 2)
    if T != None:
        x3d = x3d + T.transpose(1, 2)

    x2d = torch.div(x3d, x3d[..., 2:])
    x2d = torch.matmul(K, x2d.transpose(-1, -2)).transpose(-1, -2)[..., :2]
    return x2d


def compute_bbox_from_points(X, img_w, img_h, scaleFactor=1.2):
    left = torch.clamp(X.min(1)[0][:, 0], min=0, max=img_w)
    right = torch.clamp(X.max(1)[0][:, 0], min=0, max=img_w)
    top = torch.clamp(X.min(1)[0][:, 1], min=0, max=img_h)
    bottom = torch.clamp(X.max(1)[0][:, 1], min=0, max=img_h)

    cx = (left + right) / 2
    cy = (top + bottom) / 2
    width = (right - left)
    height = (bottom - top)

    new_left = torch.clamp(cx - width/2 * scaleFactor, min=0, max=img_w-1)
    new_right = torch.clamp(cx + width/2 * scaleFactor, min=1, max=img_w)
    new_top = torch.clamp(cy - height / 2 * scaleFactor, min=0, max=img_h-1)
    new_bottom = torch.clamp(cy + height / 2 * scaleFactor, min=1, max=img_h)

    bbox = torch.stack((new_left.detach(), new_top.detach(),
                        new_right.detach(), new_bottom.detach())).int().float().T
    return bbox


color_mapping = {
    # tab20b
    10:0, 11:1, 23:2, 27:3, # body
    12:4, 19:5, 3:6, 26:7, 16:7, # left hand
    13:8, 20:9, 15:10, 1:11, 18:11, # right hand
    24:12, 7:13, 9:14, 8:14, # left leg
    2:16, 17:17, 14:18, 22:18, # right leg
    # tab20c
    4:20, 5:20, 6:20, 25:20, # head
    21:21 , # neck
} 

def get_semantic_colors():
    # tab20b
    cmap = plt.cm.get_cmap("tab20b", 20)
    colors = [cmap(i)[:3] for i in range(20)]
    # tab20c
    cmap = plt.cm.get_cmap("tab20c", 20)
    colors.append(cmap(0)[:3])
    colors.append(cmap(1)[:3])
    # black
    colors = [(0.,0.,0.)] + colors
    return torch.Tensor(colors).float()  
class cleanShader(torch.nn.Module):
    def __init__(self, blend_params=None):
        super().__init__()
        self.blend_params = blend_params if blend_params is not None else BlendParams()

    def forward(self, fragments, meshes, **kwargs):
        # get renderer output
        blend_params = kwargs.get("blend_params", self.blend_params)
        texels = meshes.sample_textures(fragments)
        images = blending.softmax_rgb_blend(texels, fragments, blend_params, znear=-256, zfar=256)
        return images

class Renderer():
    def __init__(self, width, height, K, device, faces=None):

        self.width = width
        self.height = height
        self.K = K

        self.device = device

        if faces is not None:
            self.faces = torch.from_numpy(
                (faces).astype('int')
            ).unsqueeze(0).to(self.device)

        self.initialize_camera_params()
        self.lights = PointLights(device=device, location=[[0.0, 0.0, -10.0]])
        self.create_renderer()

    def create_camera(self, R=None, T=None):
        if R is not None:
            self.R = R.clone().view(1, 3, 3).to(self.device)
        if T is not None:
            self.T = T.clone().view(1, 3).to(self.device)

        return PerspectiveCameras(
            device=self.device,
            R=self.R.mT,
            T=self.T,
            K=self.K_full,
            image_size=self.image_sizes,
            in_ndc=False)

    def create_circle_camera(self,num = 81):
        dis = 1.0
        azim_list = []
        elev_list = []
        for i in range(num):
            azim = -float(i*(360/num))
            azim_list.append(azim)
            elev_list.append(0)
        R, T = look_at_view_transform(
            dist = dis,
            azim = azim_list,
            elev = elev_list,
        )
        # T = T + torch.tensor([0,0,1.0])
        R = R.to(self.device)
        T = T.to(self.device)
        return PerspectiveCameras(
            device=self.device,
            R=R.mT,
            T=T,
            K=self.K_full,
            image_size=self.image_sizes,
            in_ndc=False)

    def create_renderer(self):
        bg="black"
        blendparam = BlendParams(1e-4, 1e-8, np.array(ImageColor.getrgb(bg)) / 255.0)
        self.raster_settings_mesh = RasterizationSettings(
                image_size=self.image_sizes[0],
                blur_radius=np.log(1.0 / 1e-4) * 1e-7,
                bin_size=-1, # -1
                faces_per_pixel=1, #1
            ) 
        self.meshRas = MeshRasterizer(raster_settings=self.raster_settings_mesh)
        self.renderer = MeshRenderer(
                    rasterizer=self.meshRas,
                    shader=cleanShader(blend_params=blendparam),
                )

    def create_normal_renderer(self):
        bg="gray"
        blendparam = BlendParams(1e-4, 1e-8, np.array(ImageColor.getrgb(bg)) / 255.0)
        self.raster_settings_mesh = RasterizationSettings(
                image_size=self.image_sizes[0],
                blur_radius=np.log(1.0 / 1e-4) * 1e-7,
                bin_size=-1, # -1
                faces_per_pixel=50, #1
            ) 
        self.meshRas = MeshRasterizer(cameras=self.cameras,raster_settings=self.raster_settings_mesh)
        self.renderer = MeshRenderer(
                    rasterizer=self.meshRas,
                    shader=cleanShader(blend_params=blendparam),
                )

        # self.renderer = MeshRenderer(
        #     rasterizer=MeshRasterizer(
        #         raster_settings=RasterizationSettings(
        #             image_size=self.image_sizes[0],
        #             blur_radius=1e-5,),
        #     ),
        #     shader=SoftPhongShader(
        #         device=self.device,
        #         lights=self.lights,
        #     )
        # )

    # def create_normal_renderer(self):
    #     normal_renderer = MeshRenderer(
    #         rasterizer=MeshRasterizer(
    #             cameras=self.cameras,
    #             raster_settings=RasterizationSettings(
    #                 image_size=self.image_sizes[0],
    #             ),
    #         ),
    #         shader=NormalShader(device=self.device),
    #     )
    #     return normal_renderer

    def initialize_camera_params(self):
        """Hard coding for camera parameters
        TODO: Do some soft coding"""

        # Extrinsics
        self.R = torch.diag(
            torch.tensor([1, 1, 1])
        ).float().to(self.device).unsqueeze(0)

        self.T = torch.tensor(
            [0, 0, 0]
        ).unsqueeze(0).float().to(self.device)

        # Intrinsics
        self.K = self.K.unsqueeze(0).float().to(self.device)
        self.bboxes = torch.tensor([[0, 0, self.width, self.height]]).float()
        self.K_full, self.image_sizes = update_intrinsics_from_bbox(self.K, self.bboxes)
        self.cameras = self.create_camera()
        self.circle_cameras = self.create_circle_camera()

    def render_normal(self, vertices):
        vertices = vertices.unsqueeze(0)
        mesh = Meshes(verts=vertices, faces=self.faces)
        # mesh.textures = TexturesVertex(
        #         verts_features=(mesh.verts_normals_padded() + 1.0) * 0.5
        #     )
        mesh.textures = TexturesVertex(
                verts_features=(mesh.verts_normals_packed() + 1.0) * 0.5
            )
        normal_renderer = self.create_normal_renderer()
        results = normal_renderer(mesh)
        results = torch.flip(results, [1, 2])
        return results

    def render_mesh2(self, vertices, background, colors=[0.8, 0.8, 0.8]):

        self.update_bbox(vertices[::50], scale=1.2)
        vertices = vertices.unsqueeze(0)

        if colors[0] > 1: colors = [c / 255. for c in colors]
        verts_features = torch.tensor(colors).reshape(1, 1, 3).to(device=vertices.device, dtype=vertices.dtype)
        verts_features = verts_features.repeat(1, vertices.shape[1], 1)
        textures = TexturesVertex(verts_features=verts_features)

        mesh = Meshes(verts=vertices,
                      faces=self.faces,
                      textures=textures,)

        materials = Materials(
            device=self.device,
            specular_color=(colors, ),
            shininess=0
            )
        results = torch.flip(
            self.renderer(mesh, materials=materials, cameras=self.cameras, lights=self.lights),
            [1, 2]
        )
        image = results[0, ..., :3] * 255
        mask = results[0, ..., -1] > 1e-3
        image = overlay_image_onto_background(image, mask, self.bboxes, background.copy())
        self.reset_bbox()
        return image

    def render_mesh(self, vertices, background, colors=[0, 0, 0]):

        # self.update_bbox(vertices[::50], scale=1.2)
        vertices = vertices.unsqueeze(0)
        if colors[0] > 1: colors = [c / 255. for c in colors]
        colors_all = _prepare_colors(palette=['white'],
                                    render_choice='part_silhouette', 
                                    num_person=1, 
                                    num_verts=vertices.shape[1], 
                                    model_type='smplx')
        colors_all = colors_all.view(-1, vertices.shape[1], 3)
        # color mapping
        color_mapping_tensor = torch.tensor([color_mapping[i] if i in color_mapping else -1 for i in range(max(color_mapping)+1)])
        color_mapping_tensor[0] = 21
        B, N, C = colors_all.shape
        colors_all = color_mapping_tensor[colors_all.view(-1).round().long()].view(B, N, C).float()
        colors_all = colors_all + 1 

        import trimesh



        n_views = 81
        # centroid = torch.mean(vertices[0], dim=0)
        # x_obj, y_obj, z_obj = centroid
        # 计算顶点坐标的最小值和最大值
        min_coords = torch.min(vertices[0], dim=0).values
        max_coords = torch.max(vertices[0], dim=0).values
        # 计算包围盒中心点
        centroid = (min_coords + max_coords) / 2.0
        x_min, y_min, z_min = min_coords
        x_max, y_max, z_max = max_coords
        x_obj, y_obj, z_obj = centroid
        r = z_obj
        # print(r)
        # newvertices = vertices - torch.tensor([0,0,r]).to(self.device)
        newvertices = vertices
        azim_list = []
        elev_list = []
        for i in range(n_views):
            azim = float(i*(360/n_views))
            azim_list.append(azim-180)
            elev_list.append(0)
            
        # R_tensor, T_tensor = look_at_view_transform(
        #     dist = r,
        #     azim = azim_list,
        #     elev = elev_list,
        #     at = ((x_obj, y_obj, z_obj),)
        # )

        R_tensor, T_tensor = look_at_view_transform(
            dist = r,
            azim = azim_list,
            elev = elev_list,
            at = ((0, 0, z_obj),)
        )

        # circle = PerspectiveCameras(
        #     device=self.device,
        #     R=R_tensor,
        #     T=T_tensor,
        #     K=self.K_full,
        #     image_size=self.image_sizes,
        #     in_ndc=False)

        dis =  0.8 * max(1.733*(x_obj-x_min),y_obj-y_min)
        image_width = 832
        image_height = 480
        # print(x_obj-dis)
        # print(x_obj+dis)
        # print(y_obj-dis)
        # print(y_obj+dis)
        # print("zbt")
        aspect_ratio = image_width / image_height  # 约1.733
        
        circle = FoVOrthographicCameras(
            device=self.device,
            R=R_tensor,
            T=T_tensor,
            znear=0.1,
            zfar=100.0,
            min_x = -dis,
            max_x = dis,
            min_y = -dis,
            max_y = dis,
        )
        
        K_tensor = torch.transpose(circle.get_projection_transform().get_matrix(), 1, 2)
        # circle = FoVOrthographicCameras(
        #     device=self.device,
        #     R=R_tensor,
        #     T=T_tensor,
        #     K=K_tensor,
        # )
        # print("dis",dis)
        # print(K_tensor)
        R = [r.cpu().numpy() for r in R_tensor]
        T = [t.cpu().numpy() for t in T_tensor]
        K = [k.cpu().numpy() for k in K_tensor]
        semantic_mesh = ParametricMeshes(
            verts=newvertices,
            faces=self.faces,
            N_individual_overdide=1,
            model_type="smplx",
            use_nearest=True, # True
            vertex_color=colors_all).to(vertices.device) 

        import trimesh
        def setdiff1d(a, b):
            mask = ~torch.isin(a, b)
            return a[mask]
        vertices_np = newvertices[0].cpu().numpy()
        faces_np = self.faces[0].cpu().numpy()
        # 构建 mesh
        mesh = trimesh.Trimesh(vertices=vertices_np, faces=faces_np, process=False)

        fragments = self.meshRas(semantic_mesh.extend(len(self.circle_cameras)), cameras=circle)
        faces = semantic_mesh.faces_packed()  # (F, 3)
        pix_to_face = fragments.pix_to_face  # (N, H, W, K)

        verts = semantic_mesh.verts_packed()
        print(verts.shape)
        verts_h = torch.cat([verts, torch.ones_like(verts[:, :1])], dim=-1)  # (V,4)

        verts_cam = verts_h @ (circle.get_projection_transform().get_matrix().transpose(1,2))[0]
        verts_ndc = verts_cam[:, :3] / verts_cam[:, 3:4]  # 归一化设备坐标 [-1,1]

        # Step 2: 可见面索引
        # visible_faces = (pix_to_face[pix_to_face >= 0] % faces.shape[0]).unique()
        # 每个面出现的次数
        face_ids = pix_to_face[pix_to_face >= 0] % faces.shape[0]
        counts = torch.bincount(face_ids, minlength=faces.shape[0])
        # 选出出现次数 >= min_count 的面
        visible_faces = torch.nonzero(counts >= 40).squeeze(1)
        # visible_verts = torch.unique(faces[visible_faces].reshape(-1))
        # Step 3: 面索引映射到顶点索引
        all_faces = torch.arange(faces_np.shape[0], device=semantic_mesh.device)
        invisible_faces = setdiff1d(all_faces, visible_faces)

        
        def triangle_area_2d(verts_2d, faces):
            """
            计算 2D 投影下的三角形面积
            verts_2d: (V, 2)
            faces: (F, 3)
            return: (F,) 每个三角形的面积
            """
            v0 = verts_2d[faces[:, 0]]
            v1 = verts_2d[faces[:, 1]]
            v2 = verts_2d[faces[:, 2]]
            area = 0.5 * torch.abs(
                (v1[:, 0] - v0[:, 0]) * (v2[:, 1] - v0[:, 1]) -
                (v2[:, 0] - v0[:, 0]) * (v1[:, 1] - v0[:, 1])
            )
            return area
        # 假设 verts_ndc 是 (V, 3)，取前两维作为屏幕坐标
        verts_2d = verts_ndc[:, :2]
        print(verts_2d.shape)
        print(faces.shape)
        areas = triangle_area_2d(verts_2d, faces)
        # 面积阈值过滤
        print(areas.shape)
        print(areas)
        tiny_area_faces = torch.nonzero(areas < 1e-9).squeeze(1)

        invisible_faces = torch.cat([invisible_faces, tiny_area_faces]).unique()

        # Step 4: 面-》顶点
        invisible_verts = faces[invisible_faces].unique()
        all_verts = torch.arange(vertices_np.shape[0], device=semantic_mesh.device)
        visible_verts = setdiff1d(all_verts, invisible_verts)

        # bad_faces = torch.cat([rarely_visible_faces, tiny_area_faces]).unique()


        print(f"可见顶点数量: {len(visible_verts)}")
        print(f"不可见顶点数量: {len(invisible_verts)}")

        # 给每个顶点一个颜色 (默认白色)
        colors = torch.ones_like(semantic_mesh.verts_packed()) * 255
        colors = colors.cpu().numpy().astype("uint8")
        # 可见点 = 绿色，不可见点 = 红色
        colors[visible_verts.cpu().numpy()] = [0, 255, 0]
        colors[invisible_verts.cpu().numpy()] = [255, 0, 0]
        # Step 6: 用 trimesh 创建并导出
        # canomesh = trimesh.load("./tpami/mesh/cano.obj", process=False)
        # canoverts = np.asarray(canomesh.vertices)  # (N, 3)
        # mesh_trimesh = trimesh.Trimesh(vertices=canoverts, faces=faces_np, vertex_colors=colors)
        # out_path = './tpami/mesh/mesh.obj'
        # mesh_trimesh.export(out_path)
        # print(f"结果已导出到: {out_path}")

        # 提取可见顶点
        # invisible_points = canoverts[invisible_verts.cpu().numpy()]

        # pc_invisible = trimesh.points.PointCloud(invisible_points)
        # out_path = "./tpami/mesh/invisible_points.ply"
        # pc_invisible.export(out_path)


        results = torch.flip(
            self.renderer(semantic_mesh.extend(len(self.circle_cameras)), cameras=circle),
            [1, 2]
        )
        image_list = []
        print(results.shape)
        cou = results.shape[0]
        for i in range(cou):
            print("number:",i)
            images = results[i:i+1,:,:,0] # B H W index
            color_palette = get_semantic_colors().to(vertices.device)
            B, H, W = images.shape
            images = color_palette[images.view(-1).round().long()].view(B, H, W, 3) # B H W C
            # images = images.permute(0, 3, 1, 2) # B C H W
            image = images[0, ..., :3] * 255
            # mask = results[0, ..., -1] > 1e-3
            mask = results[i, ..., -1] > 1e-3
            alpha = torch.where(mask,255,0)
            image = torch.cat([image,alpha.unsqueeze(-1)],dim=-1)
            image_list.append(image.detach().cpu().numpy())
            # image = overlay_image_onto_background_rgba(image, mask, self.bboxes, background.copy())
            # image_list.append(image)

        # image = image.detach().cpu().numpy()
        # print(image.shape)
        self.reset_bbox()

        return image_list,R,T,K,visible_verts,invisible_verts

    def update_bbox(self, x3d, scale=2.0, mask=None):

        """ Update bbox of cameras from the given 3d points

        x3d: input 3D keypoints (or vertices), (num_frames, num_points, 3)
        """
        if x3d.size(-1) != 3:
            x2d = x3d.unsqueeze(0)
        else:
            x2d = perspective_projection(x3d.unsqueeze(0), self.K, self.R[0:1], self.T[0:1].reshape(1, 3, 1))

        if mask is not None:
            x2d = x2d[:, ~mask]
        bbox = compute_bbox_from_points(x2d, self.width, self.height, scale)
        self.bboxes = bbox

        self.K_full, self.image_sizes = update_intrinsics_from_bbox(self.K, bbox)
        self.cameras = self.create_camera()
        self.create_renderer()
        
    
    def reset_bbox(self):
        bbox = torch.zeros((1, 4)).float().to(self.device)
        bbox[0, 2] = self.width
        bbox[0, 3] = self.height
        self.bboxes = bbox
        self.K_full, self.image_sizes = update_intrinsics_from_bbox(self.K, bbox)
        self.cameras = self.create_camera()
        self.create_renderer()
        pass
# class Renderer():
#     def __init__(self, width, height, K, device, faces=None):

#         self.width = width
#         self.height = height
#         self.K = K

#         self.device = device

#         if faces is not None:
#             self.faces = torch.from_numpy(
#                 (faces).astype('int')
#             ).unsqueeze(0).to(self.device)

#         self.initialize_camera_params()
#         self.lights = PointLights(device=device, location=[[0.0, 0.0, -10.0]])
#         self.create_renderer()

#     def create_camera(self, R=None, T=None):
#         if R is not None:
#             self.R = R.clone().view(1, 3, 3).to(self.device)
#         if T is not None:
#             self.T = T.clone().view(1, 3).to(self.device)

#         return PerspectiveCameras(
#             device=self.device,
#             R=self.R.mT,
#             T=self.T,
#             K=self.K_full,
#             image_size=self.image_sizes,
#             in_ndc=False)

#     def create_renderer(self):
#         self.renderer = MeshRenderer(
#             rasterizer=MeshRasterizer(
#                 raster_settings=RasterizationSettings(
#                     image_size=self.image_sizes[0],
#                     blur_radius=1e-5,),
#             ),
#             shader=SoftPhongShader(
#                 device=self.device,
#                 lights=self.lights,
#             )
#         )

#     def create_normal_renderer(self):
#         normal_renderer = MeshRenderer(
#             rasterizer=MeshRasterizer(
#                 cameras=self.cameras,
#                 raster_settings=RasterizationSettings(
#                     image_size=self.image_sizes[0],
#                 ),
#             ),
#             shader=NormalShader(device=self.device),
#         )
#         return normal_renderer

#     def initialize_camera_params(self):
#         """Hard coding for camera parameters
#         TODO: Do some soft coding"""

#         # Extrinsics
#         self.R = torch.diag(
#             torch.tensor([1, 1, 1])
#         ).float().to(self.device).unsqueeze(0)

#         self.T = torch.tensor(
#             [0, 0, 0]
#         ).unsqueeze(0).float().to(self.device)

#         # Intrinsics
#         self.K = self.K.unsqueeze(0).float().to(self.device)
#         self.bboxes = torch.tensor([[0, 0, self.width, self.height]]).float()
#         self.K_full, self.image_sizes = update_intrinsics_from_bbox(self.K, self.bboxes)
#         self.cameras = self.create_camera()

#     def render_normal(self, vertices):
#         vertices = vertices.unsqueeze(0)

#         mesh = Meshes(verts=vertices, faces=self.faces)
#         normal_renderer = self.create_normal_renderer()
#         results = normal_renderer(mesh)
#         results = torch.flip(results, [1, 2])
#         return results

#     def render_mesh(self, vertices, background, colors=[0.8, 0.8, 0.8]):

#         self.update_bbox(vertices[::50], scale=1.2)
#         vertices = vertices.unsqueeze(0)

#         if colors[0] > 1: colors = [c / 255. for c in colors]
#         verts_features = torch.tensor(colors).reshape(1, 1, 3).to(device=vertices.device, dtype=vertices.dtype)
#         verts_features = verts_features.repeat(1, vertices.shape[1], 1)
#         textures = TexturesVertex(verts_features=verts_features)

#         mesh = Meshes(verts=vertices,
#                       faces=self.faces,
#                       textures=textures,)

#         materials = Materials(
#             device=self.device,
#             specular_color=(colors, ),
#             shininess=0
#             )

#         results = torch.flip(
#             self.renderer(mesh, materials=materials, cameras=self.cameras, lights=self.lights),
#             [1, 2]
#         )
#         image = results[0, ..., :3] * 255
#         mask = results[0, ..., -1] > 1e-3

#         image = overlay_image_onto_background(image, mask, self.bboxes, background.copy())
#         self.reset_bbox()
#         return image

#     def update_bbox(self, x3d, scale=2.0, mask=None):
#         """ Update bbox of cameras from the given 3d points

#         x3d: input 3D keypoints (or vertices), (num_frames, num_points, 3)
#         """
#         if x3d.size(-1) != 3:
#             x2d = x3d.unsqueeze(0)
#         else:
#             x2d = perspective_projection(x3d.unsqueeze(0), self.K, self.R, self.T.reshape(1, 3, 1))

#         if mask is not None:
#             x2d = x2d[:, ~mask]
#         bbox = compute_bbox_from_points(x2d, self.width, self.height, scale)
#         self.bboxes = bbox

#         self.K_full, self.image_sizes = update_intrinsics_from_bbox(self.K, bbox)
#         self.cameras = self.create_camera()
#         self.create_renderer()

#     def reset_bbox(self,):
#         bbox = torch.zeros((1, 4)).float().to(self.device)
#         bbox[0, 2] = self.width
#         bbox[0, 3] = self.height
#         self.bboxes = bbox

#         self.K_full, self.image_sizes = update_intrinsics_from_bbox(self.K, bbox)
#         self.cameras = self.create_camera()
#         self.create_renderer()

class RendererUtil():
    def __init__(self, K, w, h, device, faces, keep_origin=True):
        self.keep_origin = keep_origin
        self.default_R = torch.eye(3)
        self.default_T = torch.zeros(3)
        self.device = device
        self.renderer =  Renderer(w, h, K, device, faces)

    def set_extrinsic(self, R, T):
        self.default_R = R
        self.default_T = T

    def render_normal(self, verts_list):
        if not len(verts_list) == 1:
            return None
        
        self.renderer.create_camera(self.default_R, self.default_T)
        normal_map = self.renderer.render_normal(verts_list[0])
        return normal_map[0, :, :, 0]

    def render_frame(self, humans, pred_rend_array, verts_list=None, color_list=None):
        if not isinstance(pred_rend_array, np.ndarray):
            pred_rend_array = np.asarray(pred_rend_array)
        self.renderer.create_camera(self.default_R, self.default_T)
        _img = pred_rend_array
        if humans is not None:
            for human in humans:
                _img = self.renderer.render_mesh(human['v3d'].to(self.device), _img)
        else:
            for i, verts in enumerate(verts_list):
                if color_list is None:
                    _img = self.renderer.render_mesh(verts.to(self.device), _img)
                else:
                    _img = self.renderer.render_mesh(verts.to(self.device), _img, color_list[i])
        if self.keep_origin:
            _img = np.concatenate([np.asarray(pred_rend_array), _img],1).astype(np.uint8)
        return _img


    def render_video(self, results, pil_bis_frames, fps, out_path):
        writer = imageio.get_writer(
             out_path,
             fps=fps, mode='I', format='FFMPEG', macro_block_size=1
        )
        for i, humans in enumerate(tqdm(results)):
            pred_rend_array = pil_bis_frames[i]
            _img = self.render_frame( humans, pred_rend_array)
            try:
                writer.append_data(_img)
            except:
                print('Error in writing video')
                print(type(_img))
        writer.close()


def render_frame(renderer, humans, pred_rend_array, default_R, default_T, device, keep_origin=True):
    
    if not isinstance(pred_rend_array, np.ndarray):
        pred_rend_array = np.asarray(pred_rend_array)
    renderer.create_camera(default_R, default_T)
    renderer.create_circle_camera(num=81)
    _img = pred_rend_array
    if humans is None:
        humans = []
    if isinstance(humans, dict):
        humans = [humans]
    def aabb(ref_v):
        return np.min(ref_v, axis=0), np.max(ref_v, axis=0)
    for human in humans:
        if isinstance(human, dict):
            v3d = human['v3d'].to(device)
        else:
            v3d = human
        # print("v3d",v3d.shape)
        
    new_v3d = v3d.detach().cpu().numpy()
    if False:
        new_v3d = v3d-torch.mean(v3d, dim=0, keepdim=True)
        # 应用变换矩阵（注意：PyTorch 的矩阵乘法是 @ 或 matmul）
        transform = torch.tensor([[1, 0, 0], 
                                [0, -1, 0], 
                                [0, 0, -1]], dtype=torch.float32).to(device)
        new_v3d = new_v3d @ transform
        new_v3d = new_v3d.cpu().numpy()
        # new_mesh = trimesh.Trimesh(vertices=new_v3d, faces=renderer.faces[0].cpu().numpy())
        # new_mesh.export('newbasketball.obj')
        # vmin, vmax = aabb(new_v3d)
        # vgtmin, vgtmax = aabb(gt_v3d)

        # scale = ((vgtmax-vgtmin)/(vmax-vmin))[1]
        # print("scale:",scale.shape)
        # shift1 = ((vgtmin+vgtmax)/2)-((vmin+vmax)* scale/2)
        # shift = ((vgtmin+vgtmax)/2)-((vmin+vmax)* scale/2)
        # new_v3d = (new_v3d * scale + shift) 
    new_mesh = trimesh.Trimesh(vertices=new_v3d, faces=renderer.faces[0].cpu().numpy())
    new_mesh.export("./tmp/1-old.obj")
    _img_list,R,T,K,visible_verts,invisible_verts = renderer.render_mesh(v3d, _img)

    
    return new_mesh, _img_list,R,T,K,visible_verts,invisible_verts


def render_frame2(renderer, humans, pred_rend_array, default_R, default_T, device, keep_origin=True):
    
    if not isinstance(pred_rend_array, np.ndarray):
        pred_rend_array = np.asarray(pred_rend_array)
    renderer.create_camera(default_R, default_T)
    renderer.create_circle_camera(num=81)
    _img = pred_rend_array
    if humans is None:
        humans = []
    if isinstance(humans, dict):
        humans = [humans]
    for human in humans:
        if isinstance(human, dict):
            v3d = human['v3d'].to(device)
        else:
            v3d = human
        _img = renderer.render_mesh2(v3d, _img)
    if keep_origin:
        _img = np.concatenate([np.asarray(pred_rend_array), _img],1).astype(np.uint8)
    return _img


def render_video(results, faces, K, pil_bis_frames, fps, out_path, device, keep_origin=True):    
    # results [F, N, ...]
    if isinstance(pil_bis_frames[0], np.ndarray):
        height, width, _ = pil_bis_frames[0].shape
    else:
        shape = pil_bis_frames[0].size
        width, height = shape[1], shape[0]
    renderer = Renderer(width, height, K[0], device, faces)
    
    
    # build default camera
    default_R, default_T = torch.eye(3), torch.zeros(3)
    
    writer = imageio.get_writer(
             out_path,
             fps=fps, mode='I', format='FFMPEG', macro_block_size=1
        )
    for i, humans in enumerate(tqdm(results)):
        pred_rend_array = pil_bis_frames[i]
        _img = render_frame2(renderer, humans, pred_rend_array, default_R, default_T, device, keep_origin)
        try:
            writer.append_data(_img)
        except:
            print('Error in writing video')
            print(type(_img))
    writer.close()

def render_video2(results, faces, K, pil_bis_frames, fps, out_path1, out_path2, device, keep_origin=True,gt_mesh_path=""):
        num = 81
        if isinstance(pil_bis_frames[0], np.ndarray):
            height, width, _ = pil_bis_frames[0].shape
        else:
            shape = pil_bis_frames[0].size
            width, height = shape[1], shape[0]
        renderer = Renderer(width, height, K[0], device, faces)
        # writer2 = imageio.get_writer(
        #      out_path2,
        #      fps=fps, mode='I', format='FFMPEG', macro_block_size=1
        # )
        output_dir = os.path.join(os.path.dirname(out_path2), "smplimages")
        os.makedirs(output_dir, exist_ok=True)
        output_dir2 = os.path.join(os.path.dirname(out_path2), "smplimagesrgb")
        os.makedirs(output_dir2, exist_ok=True)
        # build default camera
        default_R, default_T = torch.eye(3), torch.zeros(3)
        print("outputdir:",output_dir)
        for i, humans in enumerate(tqdm(results)):
            pred_rend_array = pil_bis_frames[i]
            new_mesh, _img_list,R,T,K,visible_verts,invisible_verts = render_frame(renderer, humans, pred_rend_array, default_R, default_T, device, keep_origin)
            # writer1.append_data(np.asarray(pred_rend_array))
            # imageio.imwrite(out_path1, np.asarray(pred_rend_array))
            print(len(_img_list))
            frame_count = 0
            for _img in _img_list:
                frame_filename = os.path.join(output_dir, f"{frame_count:06d}.png")
                frame_filename2 = os.path.join(output_dir2, f"{frame_count:06d}.png")
                imageio.imwrite(frame_filename, _img.astype(np.uint8)) #img:
                imageio.imwrite(frame_filename2, _img[:,:,0:3].astype(np.uint8)) 
                frame_count += 1  # 递增计数器
        #         writer2.append_data(_img.astype(np.uint8))
        # writer2.close()
        # imageio.imwrite(out_path1[:-4]+"1.png", np.asarray(_img_list[0]).astype(np.uint8))
            # frame_filename = os.path.join(out_path1, f"frame_{i}.png")
            # smpl_filename = os.path.join(out_path2, f"frame_{i}.png")
            # imageio.imwrite(frame_filename, np.asarray(pred_rend_array))
            # imageio.imwrite(smpl_filename, _img)
        return output_dir,R,T,K,visible_verts,invisible_verts

def get_mesh(results, faces, K, pil_bis_frames, fps, out_path1, out_path2, device, keep_origin=True,gt_mesh_path=""):
        num = 81
        if isinstance(pil_bis_frames[0], np.ndarray):
            height, width, _ = pil_bis_frames[0].shape
        else:
            shape = pil_bis_frames[0].size
            width, height = shape[1], shape[0]
        renderer = Renderer(width, height, K[0], device, faces)

        # build default camera
        default_R, default_T = torch.eye(3), torch.zeros(3)
        for i, humans in enumerate(tqdm(results)):
            pred_rend_array = pil_bis_frames[i]
            new_mesh = render_frame(renderer, humans, pred_rend_array, default_R, default_T, device, keep_origin)
            # new_mesh.export(out_path1)
            # frame_filename = os.path.join(out_path1, f"frame_{i}.png")
            # smpl_filename = os.path.join(out_path2, f"frame_{i}.png")
            # imageio.imwrite(frame_filename, np.asarray(pred_rend_array))
            # imageio.imwrite(smpl_filename, _img)

