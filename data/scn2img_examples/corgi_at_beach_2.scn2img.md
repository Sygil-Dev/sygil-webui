// give it some polish and details
size: 512, 512
prompt: cute corgi at beach, intricate details, photorealistic, trending on artstation
variation: 0
seed: 1360051694
initial_seed: 5

# blend it together
prompt: beautiful corgi:1.5 cute corgi at beach, trending on artstation:1 photorealistic:1.5
ddim_steps: 50
denoising_strength: 0.5
variation: 0

## put foreground in front of background
size: 512, 512

### select foreground
size: 512, 512

// estimate depth from image and select mask by depth
// https://huggingface.co/spaces/atsantiago/Monocular_Depth_Filter
mask_depth: True
mask_depth_min: -0.05
mask_depth_max: 0.4
mask_depth_invert:False

#### create foreground
prompt: corgi
ddim_steps: 25
seed: 242886303

### create background
prompt:beach landscape, beach with ocean in background, photographic, beautiful:1 red:-0.4
variation: 3
