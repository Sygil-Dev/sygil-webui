// blend it together and finish it with some details
prompt: cute corgi at beach, trending on artstation
ddim_steps: 50
denoising_strength: 0.5
initial_seed: 2

# put foreground onto background 
size: 512, 512

## create foreground
size: 512, 512

// estimate depth from image and select mask by depth
// https://huggingface.co/spaces/atsantiago/Monocular_Depth_Filter
mask_depth: True
mask_depth_min: -0.05
mask_depth_max: 0.4
mask_depth_invert:False

###
prompt: corgi
ddim_steps: 25

## create background
prompt:beach landscape, beach with ocean in background, photographic, beautiful:1 red:-0.4
