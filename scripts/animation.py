import code
import torch
import cv2
import math
import numpy as np
# Note: in order for this to work, need the patched class KDiffusionSampler which includes new function set_sigma_start_end_indices 

def int_log2(x):
    return int(math.log2(x))
        
def animation_sample(prompt_start, prompt_end, num_animation_frames, steps, maybe_modelCS, func_sample, sampler, x, init_data, sampler_name, batch_size):
    uc = maybe_modelCS.get_learned_conditioning([""])
    MIN_SQUARE_DIFF_FACTOR = 0.5 # TODO: want this as a parameter? Increasing this can result in smoother animation.
    DONT_AVERAGE_LAST_LEVEL = True # TODO: want it as parameter? should prevent some small artifacts. 
    batch_uc = torch.cat([uc for i in range(batch_size)])
    
    c_start = maybe_modelCS.get_learned_conditioning(prompt_start)
    c_end = maybe_modelCS.get_learned_conditioning(prompt_end)
    
    max_tree_height = int_log2(num_animation_frames)
    if num_animation_frames != (1<<max_tree_height):
        max_tree_height+=1
        num_animation_frames = (1<<max_tree_height)
        print(f"changing num animation frames to a power of 2: {num_animation_frames}")
        
    current_samples = x
    previous_conditioning_ratios = [0]
    levels_sizes = [(1 << i) for i in range(max_tree_height + 1)] # TODO: want those as parameters?
    num_levels = len(levels_sizes)
    if num_levels == 1: # For single picture
        steps_per_level = [0]
    else:
        steps_per_level = list(range(0, steps, steps // num_levels))
    
    for current_level in range(num_levels):
        print(f"animation: level {current_level+1}/{num_levels}")
        print(f"current_samples.shape={current_samples.shape}")
        current_level_size = levels_sizes[current_level]
        
        new_sample_inputs = []
        new_sample_inputs_indices = []
        new_sample_ratios = []
        if current_level_size <= 2:
            new_sample_inputs = [current_samples[0]] * current_level_size
            new_sample_inputs_indices = [0] * current_level_size
            new_sample_ratios = [1] * current_level_size
            if current_level_size == 1:
                conditioning_ratios = [0.5]
            else:
                conditioning_ratios = [0, 1]
        else:
            squared_diffs = torch.sqrt(
                torch.sum(
                    torch.square(
                    torch.diff(current_samples, dim=0)
                    ),
                    list(range(1, len(current_samples.shape))))
            ).cpu()
            min_square_diff = torch.min(squared_diffs)
            squared_diffs -= min_square_diff * MIN_SQUARE_DIFF_FACTOR # Try to ignore differences from random
            squared_diffs_sums = torch.cat((torch.zeros(1), torch.cumsum(squared_diffs, dim=0)))
            squared_diffs_sums = previous_conditioning_ratios[0] + squared_diffs_sums * ( (previous_conditioning_ratios[-1] - previous_conditioning_ratios[0]) / squared_diffs_sums[-1])
            print(f"squared_diffs sums: {list(enumerate(squared_diffs_sums))}")
            
            conditioning_ratios = []
            j = 0
            max_j = len(previous_conditioning_ratios) - 1
            for i in range(current_level_size):
                wanted_ratio = i / (current_level_size - 1)
                
                while j <= max_j and wanted_ratio >= squared_diffs_sums[j]:
                    j+=1
                #print(f"i={i}, wanted_ratio={wanted_ratio}, j={j}")
                if j <= 0:
                    new_sample_inputs.append(current_samples[0])
                    new_sample_ratios.append(1)
                    conditioning_ratios.append(wanted_ratio)
                elif j > max_j:
                    j = max_j
                    new_sample_inputs.append(current_samples[j])
                    conditioning_ratios.append(wanted_ratio)
                    new_sample_ratios.append(1)
                else:
                    j -= 1
                    #print(f"i={i}, wanted_ratio={wanted_ratio}, j={j}, squared_diffs_sums[{j}]={squared_diffs_sums[j]}, squared_diffs_sums[{j+1}]={squared_diffs_sums[j+1]}")
                    first_ratio = 1 - (wanted_ratio - squared_diffs_sums[j]) / (squared_diffs_sums[j+1] - squared_diffs_sums[j])
                    new_sample_ratios.append(first_ratio)
                    conditioning_ratio = previous_conditioning_ratios[j] * first_ratio + (1-first_ratio) * previous_conditioning_ratios[j+1]
                    conditioning_ratios.append(conditioning_ratio)
                    if current_level == num_levels - 1 and DONT_AVERAGE_LAST_LEVEL: # Don't average in the last level
                        if first_ratio >= 0.5:
                            new_sample = current_samples[j]
                        else:
                            new_sample = current_samples[j+1]
                    else:
                        new_sample = current_samples[j] * first_ratio + (1-first_ratio) * current_samples[j+1]
                    new_sample_inputs.append(new_sample)
                new_sample_inputs_indices.append(j)
                
            
        conditioning_tensor = torch.cat([c_start * (1-ratio) + c_end * ratio for ratio in conditioning_ratios])
            
        print(f"new samples indices taken from: {new_sample_inputs_indices}")
        print(f"conditioning_ratios: {conditioning_ratios}")
        print(f"[(i, ratio, parent_sample_index, sample_ratio)] = {list(zip(range(current_level_size), conditioning_ratios, new_sample_inputs_indices, new_sample_ratios))}")
        new_sample_inputs = torch.cat([sample[None] for sample in new_sample_inputs])
        sigma_start = steps_per_level[current_level]
        if current_level == num_levels - 1:
            sigma_end = None
        else:
            sigma_end = steps_per_level[current_level + 1] + 1 # +1 to sigma_end because of a bug where it always skips the last sigma.
        print(f"animation: level {current_level+1}/{num_levels}, sigma_start={sigma_start}, sigma_end={sigma_end}")
        print(f"new_sample_inputs shape={new_sample_inputs.shape}, current_samples.shape = {current_samples.shape}")
        sampler.set_sigma_start_end_indices(sigma_start, sigma_end)
        for batch_offset in range(0, current_level_size, batch_size):
            batch_results = func_sample(init_data=init_data, x=new_sample_inputs[batch_offset:batch_offset+batch_size], conditioning=conditioning_tensor[batch_offset:batch_offset+batch_size], unconditional_conditioning=batch_uc[:min(batch_size, current_level_size-batch_offset)], sampler_name=sampler_name)
            if batch_offset == 0:
                current_samples = batch_results
            else:
                current_samples = torch.cat((current_samples, batch_results))
        previous_conditioning_ratios = conditioning_ratios
    
    return current_samples

def save_animation(output_frames, output_path, codec=None, fps = None, animation_duration = 5): # TODO: fps / frames interpolation.
    if fps is None:
        fps = max(1, len(output_frames) // animation_duration) # At least 1 FPS.
    if output_path.endswith("avi") and codec is None:
        codec = "XVID"
    elif output_path.endswith("mp4") and codec is None:
        codec = "avc1" # Need to download binaries from cisco for this to work.
    print(f"saving animation to '{output_path}', codec={codec}, fps={fps}, number of frames: {len(output_frames)}")
    out = cv2.VideoWriter(output_path,cv2.VideoWriter_fourcc(*codec), fps, (output_frames[0].width, output_frames[0].height))
    for frame in output_frames:
        cv_im = np.array(frame)
        out.write(cv_im[:,:,::-1])
    out.release()
