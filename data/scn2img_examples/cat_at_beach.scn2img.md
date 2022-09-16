// blend it together and finish it with details
prompt: cute happy orange cat sitting at beach, beach in background, trending on artstation:1 cute happy cat:1
sampler_name:k_euler_a
ddim_steps: 35
denoising_strength: 0.55
variation: 3
initial_seed: 1

# put foreground onto background 
size: 512, 512
color: 0,0,0

## create foreground
size:512,512
color:0,0,0,0
resize: 300, 300
pos: 256, 350

// select mask by probing some pixels from the image 
mask_by_color_at: 15, 15,   15, 256,   85, 465,  100, 480
mask_by_color_threshold:80
mask_by_color_space: HLS

// some pixels inside the cat may be selected, remove them with mask_open
mask_open: 15

// there is still some background pixels left at the edge between cat and background
// grow the mask to get them as well
mask_grow: 15

// we want to remove whatever is masked:
mask_invert: True

####
prompt: cute happy orange cat, white background
ddim_steps: 25
variation: 1

## create background
prompt:beach landscape, beach with ocean in background, photographic, beautiful:1 red:-0.4
