# ComfyUI-sampler-lcm-alternative
ComfyUI Custom Sampler nodes that add a new improved LCM scheduler function

This custom node repository adds two new nodes for ComfyUI to the Custom Sampler category. SamplerLCMAlternative and LCMScheduler (just to save a few clicks, as you could also the BasicScheduler and choose smg_uniform).
Just clone it into your custom_nodes folder and you can start using it as soon as you restart ComfyUI.

SamplerLCMAlternative has two extra parameters.
- `euler_steps`, which is an integer and allows you tell the sampler to use Euler sampling for the first n steps (or skip euler only for last n steps when n is negative).
- `ancestral`, no clue if the way I add randomness actually matches Euler ancestral sampler, but regardless, if you give this a value above 0.0, the Euler steps get some fresh randomness injected each step. The value controls how much.

With default parameters, this sampler acts exactly like the original LCM sampler from ComfyUI. When you start tuning, I recommend starting by setting `euler_steps` to half of the total step count this sampler will be handling. going higher will increase details/sharpness and lower will decrease both.
