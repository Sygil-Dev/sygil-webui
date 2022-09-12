import numpy as np
import torch
import k_diffusion as K
from tqdm.auto import trange, tqdm


def find_noise_for_image(model, device, init_image, prompt, steps=200, cond_scale=2.0, verbose=False, normalize=False, generation_callback=None):
	image = np.array(init_image).astype(np.float32) / 255.0
	image = image[None].transpose(0, 3, 1, 2)
	image = torch.from_numpy(image)
	image = 2. * image - 1.
	image = image.to(device)
	x = model.get_first_stage_encoding(model.encode_first_stage(image))

	uncond = model.get_learned_conditioning([''])
	cond = model.get_learned_conditioning([prompt])

	s_in = x.new_ones([x.shape[0]])
	dnw = K.external.CompVisDenoiser(model)
	sigmas = dnw.get_sigmas(steps).flip(0)

	if verbose:
		print(sigmas)

	for i in trange(1, len(sigmas)):
		x_in = torch.cat([x] * 2)
		sigma_in = torch.cat([sigmas[i - 1] * s_in] * 2)
		cond_in = torch.cat([uncond, cond])

		c_out, c_in = [K.utils.append_dims(k, x_in.ndim) for k in dnw.get_scalings(sigma_in)]
		
		if i == 1:
			t = dnw.sigma_to_t(torch.cat([sigmas[i] * s_in] * 2))
		else:
			t = dnw.sigma_to_t(sigma_in)
			
		eps = model.apply_model(x_in * c_in, t, cond=cond_in)
		denoised_uncond, denoised_cond = (x_in + eps * c_out).chunk(2)
		
		denoised = denoised_uncond + (denoised_cond - denoised_uncond) * cond_scale
		
		if i == 1:
			d = (x - denoised) / (2 * sigmas[i])
		else:
			d = (x - denoised) / sigmas[i - 1]

		if generation_callback is not None:
			generation_callback(x, i)

		dt = sigmas[i] - sigmas[i - 1]
		x = x + d * dt
	
	if normalize:
		# multiplying sigmas seems to break things pretty bad...
		return (x / x.std())# * sigmas[-1]
	else:
		return x
