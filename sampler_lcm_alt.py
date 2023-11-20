import comfy.samplers
from comfy.k_diffusion.sampling import default_noise_sampler
from tqdm.auto import trange, tqdm
import torch

@torch.no_grad()
def sample_lcm_alt(model, x, sigmas, extra_args=None, callback=None, disable=None, noise_sampler=None, sigma_limit=0.115):
    extra_args = {} if extra_args is None else extra_args
    noise_sampler = default_noise_sampler(x) if noise_sampler is None else noise_sampler
    s_in = x.new_ones([x.shape[0]])
    sigma_min = float(model.inner_model.inner_model.model_sampling.sigma_min)
    sigma_max = float(model.inner_model.inner_model.model_sampling.sigma_max)
    sigma_limit = sigma_min + sigma_limit * (sigma_max - sigma_min)
    for i in trange(len(sigmas) - 1, disable=disable):
        denoised = model(x, sigmas[i] * s_in, **extra_args)
        if callback is not None:
            callback({'x': x, 'i': i, 'sigma': sigmas[i], 'sigma_hat': sigmas[i], 'denoised': denoised})

        if sigmas[i+1] > sigma_limit:
            noise = (x - denoised) / sigmas[i]
        elif sigmas[i + 1] > 0:
            noise = noise_sampler(sigmas[i], sigmas[i + 1])
        else:
            noise = None
        x = denoised
        if sigmas[i + 1] > 0 and torch.is_tensor(noise):
            x += sigmas[i + 1] * noise
    return x

class SamplerLCMAlternative:
    @classmethod
    def INPUT_TYPES(s):
        return {"required":
                    {"sigma_limit": ("FLOAT", {"default": 0.115, "min": 0.0, "max": 1.0, "step":0.001, "round": False}),
                      }
               }
    RETURN_TYPES = ("SAMPLER",)
    CATEGORY = "sampling/custom_sampling/samplers"

    FUNCTION = "get_sampler"

    def get_sampler(self, sigma_limit):
        sampler = comfy.samplers.KSAMPLER(sample_lcm_alt, extra_options={"sigma_limit": sigma_limit})
        return (sampler, )


NODE_CLASS_MAPPINGS = {
    "SamplerLCMAlternative": SamplerLCMAlternative,
}
