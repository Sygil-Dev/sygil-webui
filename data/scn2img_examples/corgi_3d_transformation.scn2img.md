initial_seed: 2

// select background and img2img over it
mask_by_color_at: 64, 64
mask_invert: True

prompt: corgi
ddim_steps: 50
seed: 242886303

mask_mode: 0
denoising_strength: 0.8
//cfg_scale: 15
mask_restore: True
image_editor_mode:Mask

# estimate depth and transform the corgi in 3d
transform3d: True
transform3d_depth_near: 0.5
transform3d_depth_scale: 10
transform3d_from_hfov: 45
transform3d_to_hfov: 45
transform3d_from_pose: 0,0,0,  0,0,0
transform3d_to_pose: 0.5,0,0,  0,-5,0
transform3d_min_mask: 0
transform3d_max_mask: 255
transform3d_inpaint_radius: 1
transform3d_inpaint_method: 0

## put foreground onto background 
size: 512, 512


### create foreground
size: 512, 512

mask_depth: True
mask_depth_model: 1
mask_depth_min: -0.05
mask_depth_max: 0.5
mask_depth_invert:False

####
prompt: corgi
ddim_steps: 25
seed: 242886303

### background
size: 512,512
color: #9F978D
