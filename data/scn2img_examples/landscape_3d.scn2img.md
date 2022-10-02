size: 512,512
mask_blur: 6

prompt: fantasy landscape with castle and forest and waterfalls, trending on artstation
denoising_strength: 0.6
seed: 1
image_editor_mode: Mask
mask_mode: 0
mask_restore: True

# mask the left which contains artifacts
color: 255,255,255,0
blend:multiply
size: 100,512
pos: 50,256

# mask the top-left which contains lots of artifacts
color: 255,255,255,0
blend:multiply
size: 280,128
pos: 128,64

# go forward and turn head left to look at the left waterfalls
transform3d: True
transform3d_depth_scale: 10000
transform3d_from_hfov: 60
transform3d_to_hfov: 60
transform3d_from_pose: 0,0,0,  0,0,0
transform3d_to_pose: 4000,0,2000,  0,-50,0
transform3d_min_mask: 0
transform3d_max_mask: 255
transform3d_inpaint_radius: 5
transform3d_inpaint_method: 1

##
prompt: fantasy landscape with castle and forest and waterfalls, trending on artstation
seed: 1
