from pytorch3d.structures import Meshes
from pytorch3d.renderer.mesh.rasterizer import Fragments
from pytorch3d.renderer.mesh.shading import interpolate_face_attributes
from pytorch3d.renderer import (
    look_at_view_transform,
    sigmoid_alpha_blend,
    softmax_rgb_blend,
    BlendParams,
    FoVPerspectiveCameras,
    PerspectiveCameras,
    RasterizationSettings,
    MeshRenderer,
    TexturesVertex,
    MeshRasterizer,
    FoVOrthographicCameras,
    look_at_view_transform,
    blending,
)
import torch
import torch.nn as nn
import numpy as np
from PIL import Image

from mmhuman3d.core.renderer.torch3d_renderer.meshes import ParametricMeshes
from mmhuman3d.core.visualization.visualize_smpl import _prepare_colors

import matplotlib.pyplot as plt
from PIL import ImageColor

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

class PlainColorShader(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.blend_params = BlendParams(background_color=(0., 0., 0.), sigma=1e-20, gamma=1e-20)

    def forward(self, fragments: Fragments, meshes: Meshes, **kwargs) -> torch.Tensor:
        """
        Only want to render the silhouette so RGB values can be ones.
        There is no need for lighting or texturing
        """
        #vert_colors = meshes.verts_normals_packed() * 0.5 + 0.5
        vert_colors = kwargs.get("vert_colors")
        faces = meshes.faces_packed()
        face_colors = vert_colors[faces]
        colors = interpolate_face_attributes(
            fragments.pix_to_face, fragments.bary_coords, face_colors
        )

        blend_params = kwargs.get("blend_params", self.blend_params)
        images = softmax_rgb_blend(colors, fragments, blend_params)
        return images

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

def render_smpl_vcano_map(smpl_verts, smpl_faces, device, cam_R=torch.eye(3), cam_t=torch.zeros((3,1)), cam_K=torch.zeros((3,3)), img_w=1024, img_h=1024):
    cam_f = torch.stack([cam_K[0, 0], cam_K[1, 1]], dim=-1)
    cam_c = torch.stack([img_w - cam_K[0, 2], img_h - cam_K[1, 2]], dim=-1)
    smpl_verts = torch.from_numpy(smpl_verts.astype(np.float32)).unsqueeze(0).to(device)
    smpl_faces = torch.from_numpy(smpl_faces.astype(np.float32)).to(device)

    cameras = PerspectiveCameras(
        focal_length=cam_f.reshape(-1, 2), principal_point=cam_c.reshape(-1, 2),
        R=cam_R.reshape(-1, 3, 3).permute(0, 2, 1), T=cam_t.reshape(-1, 3),
        # R=cam_R.reshape(-1, 3, 3), T=cam_t.reshape(-1, 3),
        in_ndc=False, image_size=((img_h, img_w),),
        device=smpl_verts.device)


    raster_settings = RasterizationSettings(
        image_size=(img_h, img_w),
        blur_radius=0.0,
        faces_per_pixel=1,
        max_faces_per_bin=len(smpl_faces),
    )
    
    render_meshes = Meshes(verts=smpl_verts, faces=smpl_faces.unsqueeze(0))

    
    vert_colors = render_meshes.verts_normals_packed() * -1
    vert_colors = cam_R.reshape(-1, 3, 3).to(device) @ vert_colors.transpose(0, 1).unsqueeze(0)
    vert_colors = vert_colors[0].transpose(0, 1)
    vert_colors = vert_colors* 0.5 + 0.5

    renderer = MeshRenderer(
        rasterizer=MeshRasterizer(
            cameras=cameras,
            raster_settings=raster_settings
        ),
        shader=PlainColorShader()
    )

    cano_vert_render = renderer(render_meshes, vert_colors=vert_colors)
    cano_vert_render = torch.flip(cano_vert_render, [1, 2])
    return cano_vert_render

def render_smpl_vcano_map_orth(smpl_verts, smpl_faces, device, azim_list=[0.0], elev_list=[0.0], img_w=1024, img_h=1024, random_color = None):
    # cam_f = torch.stack([cam_K[0, 0], cam_K[1, 1]], dim=-1)
    # cam_c = torch.stack([img_w - cam_K[0, 2], img_h - cam_K[1, 2]], dim=-1)
    smpl_verts = torch.from_numpy(smpl_verts.astype(np.float32)).unsqueeze(0).to(device)
    smpl_faces = torch.from_numpy(smpl_faces.astype(np.float32)).to(device)


    R, T = look_at_view_transform(
            dist = 1.0,
            azim = azim_list,
            elev = elev_list,
        )
    R = R.clone().to(device)
    T = T.clone().to(device)
    R[:, :3, 0:2] *= -1
    
    cameras = FoVOrthographicCameras(
        device=device,
        R=R,
        T=T,
    )

    raster_settings = RasterizationSettings(
        image_size=(img_h, img_w),
        blur_radius=0.0,
        faces_per_pixel=1,
        bin_size=None,
        max_faces_per_bin=len(smpl_faces),
    )
    
    render_meshes = Meshes(verts=smpl_verts, faces=smpl_faces.unsqueeze(0))

    
    vert_colors = render_meshes.verts_normals_packed() * -1
    # vert_colors = R.reshape(-1, 3, 3).to(device) @ vert_colors.transpose(0, 1).unsqueeze(0)
    # vert_colors = vert_colors[0].transpose(0, 1)
    vert_colors = vert_colors* 0.5 + 0.5
    if random_color is not None:
        vert_colors = random_color

    renderer = MeshRenderer(
        rasterizer=MeshRasterizer(
            cameras=cameras,
            raster_settings=raster_settings
        ),
        shader=PlainColorShader()
    )

    cano_vert_render = renderer(render_meshes, vert_colors=vert_colors)
    cano_vert_render = torch.flip(cano_vert_render, [1, 2])
    return cano_vert_render

def render_smpl_vcano_map_semantic_orth(smpl_verts, smpl_faces, device, azim_list=[0.0], elev_list=[0.0], img_w=1024, img_h=1024, semantic = True, random_color = None):
    # cam_f = torch.stack([cam_K[0, 0], cam_K[1, 1]], dim=-1)
    # cam_c = torch.stack([img_w - cam_K[0, 2], img_h - cam_K[1, 2]], dim=-1)
    if not isinstance(smpl_faces, torch.Tensor):
        smpl_faces = torch.from_numpy(smpl_faces.astype(np.float32))
        smpl_verts = torch.from_numpy(smpl_verts.astype(np.float32))
    smpl_verts = smpl_verts.unsqueeze(0).to(device)
    smpl_faces = smpl_faces.to(device)


    R, T = look_at_view_transform(
            dist = 2.0,
            azim = azim_list,
            elev = elev_list,
        )
    R = R.clone().to(device)
    T = T.clone().to(device)

    # cameras_semantic = FoVOrthographicCameras(
    #     device=device,
    #     R=R,
    #     T=T,
    # )

    R[:, :3, 0:2] *= -1
    cameras = FoVOrthographicCameras(
        device=device,
        R=R,
        T=T,
    )
    # cameras = FoVPerspectiveCameras(
    #     znear = 0.01,
    #     zfar = 100.0,
    #     fov = 1.0808390378952026/np.pi *180,
    #     R = R,
    #     T = T, 
    #     device=device
    # )
    raster_settings = RasterizationSettings(
        image_size=(img_h, img_w),
        blur_radius=0.0,
        faces_per_pixel=1,
        bin_size=None,
        max_faces_per_bin=len(smpl_faces),
    )
    
    render_meshes = Meshes(verts=smpl_verts, faces=smpl_faces.unsqueeze(0))
    render_meshes.textures = TexturesVertex(
                verts_features=(render_meshes.verts_normals_padded() + 1.0) * 0.5
            )

    ## semantic mesh
    if semantic:
        colors_all = _prepare_colors(palette=['white'],
                                    render_choice='part_silhouette', 
                                    num_person=1, 
                                    num_verts=smpl_verts.shape[1], 
                                    model_type='smplx')
        colors_all = colors_all.view(-1, smpl_verts.shape[1], 3)
        # color mapping
        color_mapping_tensor = torch.tensor([color_mapping[i] if i in color_mapping else -1 for i in range(max(color_mapping)+1)])
        color_mapping_tensor[0] = 21
        B, N, C = colors_all.shape
        colors_all = color_mapping_tensor[colors_all.view(-1).round().long()].view(B, N, C).float()
        colors_all = colors_all + 1 
        # print(smpl_verts.shape)
        render_semantic_mesh = ParametricMeshes(
            verts=smpl_verts,
            faces=smpl_faces,
            N_individual_overdide=1,
            model_type="smplx",
            use_nearest=True, # True
            vertex_color=colors_all).to(device) 

    # normal render
    vert_colors = render_meshes.verts_normals_packed() * -1
    # vert_colors = R.reshape(-1, 3, 3).to(device) @ vert_colors.transpose(0, 1).unsqueeze(0)
    # vert_colors = vert_colors[0].transpose(0, 1)
    vert_colors = vert_colors* 0.5 + 0.5

    renderer = MeshRenderer(
        rasterizer=MeshRasterizer(
            cameras=cameras,
            raster_settings=raster_settings
        ),
        shader=PlainColorShader()
    )
    if random_color is not None:
        vert_colors = random_color
    cano_vert_render = renderer(render_meshes, vert_colors=vert_colors)
    cano_vert_render = torch.flip(cano_vert_render, [1, 2])

    # semantic render
    if semantic:
        raster_settings_mesh = RasterizationSettings(
            image_size=(img_h, img_w),
            blur_radius=np.log(1.0 / 1e-4) * 1e-7,
            bin_size=None, # -1
            faces_per_pixel=1, #1
        ) 
        blendparam = BlendParams(1e-4, 1e-8, np.array(ImageColor.getrgb("black")) / 255.0)
        renderer1 = MeshRenderer(
                rasterizer=MeshRasterizer(cameras=cameras, raster_settings=raster_settings_mesh),
                shader=cleanShader(blend_params=blendparam),
            )

        semantic_vert_render = renderer1(render_semantic_mesh.extend(1)) 
        semantic_vert_render = semantic_vert_render[:,:,:,0]
        color_palette = get_semantic_colors().to(device)
        B, H, W = semantic_vert_render.shape
        semantic_vert_render = color_palette[semantic_vert_render.view(-1).round().long()].view(B, H, W, 3) # B H W C
        semantic_vert_render = torch.flip(semantic_vert_render, [1, 2]) # B C H W
    else:
        semantic_vert_render = cano_vert_render
    
    return cano_vert_render, semantic_vert_render