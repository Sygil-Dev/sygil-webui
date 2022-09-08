from util.imports import *

def slerp(device, t, v0:torch.Tensor, v1:torch.Tensor, DOT_THRESHOLD=0.9995):
	v0 = v0.detach().cpu().numpy()
	v1 = v1.detach().cpu().numpy()

	dot = np.sum(v0 * v1 / (np.linalg.norm(v0) * np.linalg.norm(v1)))
	if np.abs(dot) > DOT_THRESHOLD:
		v2 = (1 - t) * v0 + t * v1
	else:
		theta_0 = np.arccos(dot)
		sin_theta_0 = np.sin(theta_0)
		theta_t = theta_0 * t
		sin_theta_t = np.sin(theta_t)
		s0 = np.sin(theta_0 - theta_t) / sin_theta_0
		s1 = sin_theta_t / sin_theta_0
		v2 = s0 * v0 + s1 * v1

	v2 = torch.from_numpy(v2).to(device)

	return v2
