from util.imports import *
from util.generation_callback import *

class CFGMaskedDenoiser(nn.Module):
	def __init__(self, model):
		super().__init__()
		self.inner_model = model

	def forward(self, x, sigma, uncond, cond, cond_scale, mask, x0, xi):
		x_in = x
		x_in = torch.cat([x_in] * 2)
		sigma_in = torch.cat([sigma] * 2)
		cond_in = torch.cat([uncond, cond])
		uncond, cond = self.inner_model(x_in, sigma_in, cond=cond_in).chunk(2)
		denoised = uncond + (cond - uncond) * cond_scale

		if mask is not None:
			assert x0 is not None
			img_orig = x0
			mask_inv = 1. - mask
			denoised = (img_orig * mask_inv) + (mask * denoised)

		return denoised

class CFGDenoiser(nn.Module):
	def __init__(self, model):
		super().__init__()
		self.inner_model = model

	def forward(self, x, sigma, uncond, cond, cond_scale):
		x_in = torch.cat([x] * 2)
		sigma_in = torch.cat([sigma] * 2)
		cond_in = torch.cat([uncond, cond])
		uncond, cond = self.inner_model(x_in, sigma_in, cond=cond_in).chunk(2)
		return uncond + (cond - uncond) * cond_scale
def append_zero(x):
	return torch.cat([x, x.new_zeros([1])])
def append_dims(x, target_dims):
	"""Appends dimensions to the end of a tensor until it has target_dims dimensions."""
	dims_to_append = target_dims - x.ndim
	if dims_to_append < 0:
		raise ValueError(f'input has {x.ndim} dims but target_dims is {target_dims}, which is less')
	return x[(...,) + (None,) * dims_to_append]
def get_sigmas_karras(n, sigma_min, sigma_max, rho=7., device='cpu'):
	"""Constructs the noise schedule of Karras et al. (2022)."""
	ramp = torch.linspace(0, 1, n)
	min_inv_rho = sigma_min ** (1 / rho)
	max_inv_rho = sigma_max ** (1 / rho)
	sigmas = (max_inv_rho + ramp * (min_inv_rho - max_inv_rho)) ** rho
	return append_zero(sigmas).to(device)


def get_sigmas_exponential(n, sigma_min, sigma_max, device='cpu'):
	"""Constructs an exponential noise schedule."""
	sigmas = torch.linspace(math.log(sigma_max), math.log(sigma_min), n, device=device).exp()
	return append_zero(sigmas)


def get_sigmas_vp(n, beta_d=19.9, beta_min=0.1, eps_s=1e-3, device='cpu'):
	"""Constructs a continuous VP noise schedule."""
	t = torch.linspace(1, eps_s, n, device=device)
	sigmas = torch.sqrt(torch.exp(beta_d * t ** 2 / 2 + beta_min * t) - 1)
	return append_zero(sigmas)


def to_d(x, sigma, denoised):
	"""Converts a denoiser output to a Karras ODE derivative."""
	return (x - denoised) / append_dims(sigma, x.ndim)
def linear_multistep_coeff(order, t, i, j):
	if order - 1 > i:
		raise ValueError(f'Order {order} too high for step {i}')
	def fn(tau):
		prod = 1.
		for k in range(order):
			if j == k:
				continue
			prod *= (tau - t[i - k]) / (t[i - j] - t[i - k])
		return prod
	return integrate.quad(fn, t[i], t[i + 1], epsrel=1e-4)[0]

class KDiffusionSampler:
	def __init__(self, m, sampler):
		self.model = m
		self.model_wrap = K.external.CompVisDenoiser(m)
		self.schedule = sampler
	def get_sampler_name(self):
		return self.schedule
	def sample(self, S, conditioning, batch_size, shape, verbose, unconditional_guidance_scale, unconditional_conditioning, eta, x_T, img_callback=None, log_every_t=None):
		sigmas = self.model_wrap.get_sigmas(S)
		x = x_T * sigmas[0]
		model_wrap_cfg = CFGDenoiser(self.model_wrap)
		samples_ddim = None
		samples_ddim = K.sampling.__dict__[f'sample_{self.schedule}'](model_wrap_cfg, x, sigmas,
                                                                              extra_args={'cond': conditioning, 'uncond': unconditional_conditioning,
                                                                                          'cond_scale': unconditional_guidance_scale}, disable=False, callback=generation_callback)
		#
		return samples_ddim, None


@torch.no_grad()
def log_likelihood(model, x, sigma_min, sigma_max, extra_args=None, atol=1e-4, rtol=1e-4):
	extra_args = {} if extra_args is None else extra_args
	s_in = x.new_ones([x.shape[0]])
	v = torch.randint_like(x, 2) * 2 - 1
	fevals = 0
	def ode_fn(sigma, x):
		nonlocal fevals
		with torch.enable_grad():
			x = x[0].detach().requires_grad_()
			denoised = model(x, sigma * s_in, **extra_args)
			d = to_d(x, sigma, denoised)
			fevals += 1
			grad = torch.autograd.grad((d * v).sum(), x)[0]
			d_ll = (v * grad).flatten(1).sum(1)
		return d.detach(), d_ll
	x_min = x, x.new_zeros([x.shape[0]])
	t = x.new_tensor([sigma_min, sigma_max])
	sol = odeint(ode_fn, x_min, t, atol=atol, rtol=rtol, method='dopri5')
	latent, delta_ll = sol[0][-1], sol[1][-1]
	ll_prior = torch.distributions.Normal(0, sigma_max).log_prob(latent).flatten(1).sum(1)
	return ll_prior + delta_ll, {'fevals': fevals}


def create_random_tensors(shape, seeds):
	xs = []
	for seed in seeds:
		torch.manual_seed(seed)

		# randn results depend on device; gpu and cpu get different results for same seed;
		# the way I see it, it's better to do this on CPU, so that everyone gets same result;
		# but the original script had it like this so i do not dare change it for now because
		# it will break everyone's seeds.
		xs.append(torch.randn(shape, device=defaults.general.gpu))
	x = torch.stack(xs)
	return x