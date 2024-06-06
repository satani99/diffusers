import torch 
from PIL import Image 
from typing import List, Optional, Tuple, Union
from diffusers import StableDiffusionImg2ImgPipeline, StableDiffusionXLImg2ImgPipeline, DDIMScheduler, EulerAncestralDiscreteScheduler
from diffusers.utils import BaseOutput 
from diffusers.utils.torch_utils import randn_tensor
import numpy as np 

class SchedulerOutput(BaseOutput):
    """
    Output class for the scheduler's `step` function output.

    Args:
        prev_sample (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)` for images):
            Computed sample `(x_{t-1})` of previous timestep. `prev_sample` should be used as next model input in the
            denoising loop.
        pred_original_sample (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)` for images):
            The predicted denoised sample `(x_{0})` based on the model output from the current timestep.
            `pred_original_sample` can be used to preview progress or for guidance.
    """

    prev_sample: torch.FloatTensor
    pred_original_sample: Optional[torch.FloatTensor] = None 

class MyDDIMScheduler(DDIMScheduler):

    def inv_step(
            self,
            model_output: torch.FloatTensor,
            timestep: int,
            sample: torch.FloatTensor,
            eta: float = 0.0,
            use_clipped_model_output: bool = False,
            generator=None,
            variance_noise: Optional[torch.FloatTensor] = None,
            return_dict: bool = True,
    ) -> Union[SchedulerOutput, Tuple]:
        """
        Predict the sample from the previous timestep by reversing the SDE. This function propagates the diffusion 
        process from the learned model outputs (most often the predicted noise).

        Args:
            model_output (`torch.FloatTensor`):
                The direct output from learned diffusion model.
            timestep (`float`):
                The current discrete timestep in the diffusion chain.
            sample (`torch.FloatTensor`):
                A current instance of a sample created by the diffusion process.
            eta (`float`):
                The weight of noise for added noise in diffusion step.
            use_clipped_model_output (`bool`, defaults to `False`):
                If `True`, computes "corrected" `model_output` from the clipped predicted original sample. Necessary 
                because predicted original sample is clipped to [-1, 1] when `self.config.clip_sample` is `True`. If np
                clipping has happened, "corrected" `model_output` would coincide with the one provided as input and
                `use_clipped_model_output` has no effect.
            generator (`torch.Generator`, *optional*):
                A random number generator.
            variance_noise (`torch.FloatTensor`):
                Alternative to generating noise with `generator` by directing providing the noise for the variance
                itself. Useful for methods such as [`CycleDiffusion`].
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`SchedulerOutput`] or `tuple`.
        
        Returns:
            [`SchedulerOutput`] or `tuple`:
                If return_dict is `True`, [`SchedulerOutput`] is returned, otherwise a
                tuple is returned where the first element is the sample tensor.
        """
        if self.num_inference_steps is None:
            raise ValueError(
                "Number of influence steps is 'None', you need to run 'set_timesteps' after creating the scheduler"
            )
        # 1. get previous step value (=t-1)
        prev_timestep = timestep - self.config.num_train_timesteps // self.num_inference_steps 

        # 2. compute alphas, betas
        alpha_prod_t = self.alphas_cumprod[timestep]
        alpha_prod_t_prev = self.alphas_cumprod[prev_timestep] if prev_timestep >= 0 else self.final_alpha_cumprod 

        beta_prod_t = 1 - alpha_prod_t 

        # 3. compute predicted original sample from predicted noise also called
        assert self.config.prediction_type == "epsilon"
        if self.config.prediction_type == "epsilon":
            pred_original_sample = (sample - beta_prod_t ** (0.5) * model_output) / alpha_prod_t ** (0.5)
            pred_epsilon = model_output 
        elif self.config.prediction_type == "sample":
            pred_original_sample = model_output 
            pred_epsilon = (sample - alpha_prod_t ** (0.5) * pred_original_sample) / beta_prod_t ** (0.5)
        elif self.config.prediction_type == "v_prediction":
            pred_original_sample = (alpha_prod_t**0.5) * sample - (beta_prod_t**0.5) * model_output 
            pred_epsilon = (alpha_prod_t**0.5) * model_output + (beta_prod_t**0.5) * sample 
        else:
            raise ValueError(
                f"prediction_type given as {self.config.prediction_type} must be one of `epsilon`, `sample`, or"
                " `v_prediction`"
            )

        # 4. Clip or threshold "predicted x_0"
        if self.config.thresholding:
            pred_original_sample = self._threshold_sample(pred_original_sample)
        elif self.config.clip_sample:
            pred_original_sample = pred_original_sample.clamp(
                self.config.clip_sample_range, self.config.clip_sample_range 
            )
        
        # 5. compute variance 
        variance = self._get_variance(timestep, prev_timestep)
        std_dev_t = eta * variance ** (0.5)

        if use_clipped_model_output:
            # the pred_epsilon is always re-derived from the clipped x_0 in Glide
            pred_epsilon = (sample - alpha_prod_t ** (0.5) * pred_original_sample) / beta_prod_t ** (0.5)

        # 6. compute "direction pointing to x_t"
        pred_sample_direction = (1 - alpha_prod_t_prev - std_dev_t**2) ** (0.5) * pred_epsilon 

        # 7. compute x_t without "random noise" 
        # prev_sample = alpha_prod_t_prev ** (0.5) * pred_original_sample + pred_sample_direction 

        prev_sample = (alpha_prod_t ** (0.5) * sample) / alpha_prod_t_prev ** (0.5) + (alpha_prod_t_prev ** (0.5) * beta_prod_t ** (0.5) * model_output) / alpha_prod_t_prev ** (0.5) - (alpha_prod_t ** (0.5) * pred_sample_direction) / alpha_prod_t_prev ** (0.5)

        if eta > 0:
            if variance_noise is not None and generator is not None:
                raise ValueError(
                    "Cannot pass both generator and variance_noise. Please make sure that either `generator` or"
                    " `variance_noise` stays `None`."
                )
            
            if variance_noise is None:
                variance_noise = randn_tensor(
                    model_output.shape, generator=generator, device=model_output.device, dtype=model_output.dtype
                )
            variance = std_dev_t * variance_noise 

            prev_sample = prev_sample + variance 

        if not return_dict:
            return (prev_sample,)

        return SchedulerOutput(prev_sample=prev_sample, pred_original_sample=pred_original_sample)


class MyEulerAncestralDiscreteScheduler(EulerAncestralDiscreteScheduler):
    def set_noise_list(self, noise_list):
        self.noise_list = noise_list 
    
    def get_noise_to_remove(self):
        sigma_from = self.sigmas[self.step_index]
        sigma_to = self.sigmas[self.step_index + 1]
        sigma_up = (sigma_to**2 * (sigma_from**2 - sigma_to**2) / sigma_from**2) ** 0.5 

        return self.noise_list[self.step_index] * sigma_up

    def scale_model_input(
            self, sample: torch.FloatTensor, timestep: Union[float, torch.FloatTensor]
    ) -> torch.FloatTensor:
        """
        Ensures interchangeability with schedulers that need to scale the denoising model input depending on the
        current timestep. Scales the denoising model input by `(sigma**2 + 1) ** 0.5` to match the Euler algorithm.

        Args:
            sample (`torch.FloatTensor`):
                The input sample.
            timestep (`int`, *optional*):
                The current timestep in the diffusion chain.
        
        Returns:
            `torch.FloatTensor`:
                A scaled input sample.
        """

        self._init_step_index(timestep.view((1)))
        return EulerAncestralDiscreteScheduler.scale_model_input(self, sample, timestep)
    
    def step(
            self,
            model_output: torch.FloatTensor,
            timestep: Union[float, torch.FloatTensor],
            sample: torch.FloatTensor,
            generator: Optional[torch.Generator] = None,
            return_dict: bool = True,
    ) -> Union[SchedulerOutput, Tuple]:
        """
        Predict the sample from the previous timestep by reversing the SDE. This function propagates the diffusion 
        process from the learned model outputs (most often the predicted noise).

        Args:
            model_output (`torch.FloatTensor`):
                The direct output from learned diffusion model.
            timestep (`float`):
                The current discrete timestep in the diffusion chain.
            sample (`torch.FloatTensor`):
                A current instance of a sample created by the diffusion process.
            generator (`torch.Generator`, *optional*):
                A random number generator.
            return_dict (`bool`):
                Whether or not to return a 
                [`SchedulerOutput`] or tuple.

        Returns:
            [`~schedulers.scheduling_euler_ancestral_discrete.EulerAncestralDiscreteSchedulerOutput`] or `tuple`:
                If return_dict is `True`,
                [`SchedulerOutput`] is returned,
                otherwise a tuple is returned where the first element is the sample tensor.

        """

        if (
            isinstance(timestep, int)
            or isinstance(timestep, torch.IntTensor)
            or isinstance(timestep, torch.LongTensor)
        ):
            raise ValueError(
                (
                    "Passing integer indices (e.g. from `enumerate(timesteps)`) as timesteps to"
                    " `EulerDiscreteScheduler.step()` is not supported. Make sure to pass"
                    " one of the `scheduler.timesteps` as a timestep."
                ),
            )
        
        if not self.is_scale_input_called:
            logger.warning(
                "The `scale_model_input` function should be called before `step` to ensure correct denoising. "
                "See `StableDiffusionPipeline` for a usage example."
            )
        
        self._init_step_index(timestep.view(1))

        sigma = self.sigmas[self.step_index]

        # Upcast to avoid precision issues when computing prev_sample
        sample = sample.to(torch.float32)

        # 1. compute predicted original sample (x_0) from sigma-scaled predicted noise
        if self.config.prediction_type == "epsilon":
            pred_original_sample = sample - sigma * model_output 
        elif self.config.prediction_type == "v_prediction":
            # * c_out + input * c_skip
            pred_original_sample = model_output * (-sigma / (sigma**2 +1) **0.5) + (sample / (sigma**2 + 1))
        elif self.config.prediction_type == "sample":
            raise NotImplementedError("prediction_type not implemented yet: sample")
        else:
            raise ValueError(
                f"prediction_type given as {self.config.prediction_type} must be one of `epsilon`, or `v_prediction`"
            )
    
        sigma_from = self.sigmas[self.step_index]
        sigma_to = self.sigmas[self.step_index + 1]
        sigma_up = (sigma_to**2 * (sigma_from**2 - sigma_to**2) / sigma_from**2) ** 0.5
        sigma_down = (sigma_to**2 - sigma_up**2) ** 0.5 

        # 2. Convert to an ODE derivative
        # derivative = (sample - pred_original_sample) / sigma
        derivative = model_output 

        dt = sigma_down - sigma 

        prev_sample = sample + derivative * dt 

        device = model_output.device 
        # noise = randn_tensor(model_output.shape, dtype=model_output.dtype, device=device, generator=generator)
        # prev_sample = prev_sample + noise * sigma_up 

        prev_sample = prev_sample + self.noise_list[self.step_index] * sigma_up 

        # Cast sample back to model compatible dtype 
        prev_sample = prev_sample.to(model_output.dtype)

        # upon completion increase step index by one 
        self._step_index += 1 

        if not return_dict:
            return (prev_sample,)
        
        return SchedulerOutput(
            prev_sample=prev_sample, pred_original_sample=pred_original_sample
        )
    
    def step_and_update_noise(
            self,
            model_output: torch.FloatTensor,
            timestep: Union[float, torch.FloatTensor],
            sample: torch.FloatTensor,
            expected_prev_sample: torch.FloatTensor,
            optimize_epsilon_type: bool = False,
            generator: Optional[torch.Generator] = None,
            return_dict: bool = True,
    ) -> Union[SchedulerOutput, Tuple]:
        """
        Predict the sample from the previous timestep by reversing the SDE. This function propagates the diffusion
        process from the learned model outputs (most often the predicted noise).

        Args:
            model_output (`torch.FloatTensor`):
                The direct output from learned diffusion model.
            timestep (`float`):
                The current discrete timestep in the diffusion chain.
            sample (`torch.FloatTensor`):
                A current instance of a sample created by the diffision process.
            generator (`torch.Generator`, *optional*):
                A random number generator.
            return_dict (`bool`):
                Whether or not to return a 
                [`SchedulerOutput`] or `tuple`.
            
        Returns:
            [`~SchedulerOutput`] or `tuple`:
                If return_dict is `True`,
                [`SchedulerOutput`] is returned,
                otherwise a tuple is returned where the first element is the sample tensor.

        """

        if(
            isinstance(timestep, int)
            or isinstance(timestep, torch.IntTensor)
            or isinstance(timestep, torch.LongTensor)
        ):
            raise ValueError(
                (
                    "Passing integer indices (e.g. from `enumerate(timesteps)`) as timesteps to"
                    " `EulerDiscreteScheduler.step()` is not supported. Make sure to pass"
                    "one of the `scheduler.timesteps` as a timestep."
                ),
            )
        
        if not self.is_scale_input_called:
            logger.warning(
                "The `scale_model_input` function should be called before `step` to ensure correct denoising. "
                "See `StableDiffusionPipeline` for a usage example."
            )

        self._init_step_index(timestep.view(1))

        sigma = self.sigmas[self.step_index]

        # Upcast to avoid precision issues when computing prev_sample 
        sample = sample.to(torch.float32)

        # 1. compute predicted original sample (x_0) from sigma-scaled predicted noise 
        if self.config.prediction_type == "epsilon":
            pred_original_sample = sample - sigma * model_output 
        elif self.config.prediction_type == "v_prediction":
            # * c_out + input * c_skip 
            pred_original_sample = model_output * (-sigma / (sigma**2 + 1) ** 0.5) + (sample / (sigma**2 + 1))
        elif self.config.prediction_type == "sample":
            raise NotImplementedError("prediction_type not implemented yet: sample")
        else:
            raise ValueError(
                f"prediction_type given as {self.config.prediction_type} must be one of `epsilon`, or `v_prediction`"
            )
        
        sigma_from = self.sigmas[self.step_index]
        sigma_to = self.sigmas[self.step_index + 1]
        sigma_up = (sigma_to**2 * (sigma_from**2 - sigma_to**2) / sigma_from**2) ** 0.5 
        sigma_down = (sigma_to**2 - sigma_up**2) ** 0.5 

        # 2. Convert to an ODE derivative 
        # derivative = (sample - pred_original_sample) / sigma 
        derivative = model_output 

        dt = sigma_down - sigma 

        prev_sample = sample + derivative * dt 

        device = model_output.device 
        # noise = randn_tensor(model_output.shape, dtype=model_output.dtype, device=device, generator=generator)
        # prev_sample = prev_sample + noise * sigma_up 

        if sigma_up > 0:
            req_noise = (expected_prev_sample - prev_sample) / sigma_up 
            if not optimize_epsilon_type:
                self.noise_list[self.step_index] = req_noise 
            else:
                for i in range(10):
                    n = torch.autograd.Variable(self.noise_list[self.step_index].detach().clone(), requires_grad=True)
                    loss = torch.norm(n - req_noise.detach())
                    loss.backward()
                    self.noise_list[self.step_index] -= n.grad.detach() * 1.8 

        prev_sample = prev_sample + self.noise_list[self.step_index] * sigma_up 

        # Cast sample back to model compatible dtype 
        prev_sample = prev_sample.to(model_output.dtype)

        # upon completion increase step index by one 
        self._step_index += 1

        if not return_dict:
            return (prev_sample,)
        
        return SchedulerOutput(
            prev_sample=prev_sample, pred_original_sample=pred_original_sample 
        )
    
    def inv_step(
            self,
            model_output: torch.FloatTensor,
            timestep: Union[float, torch.FloatTensor],
            sample: torch.FloatTensor,
            generator: Optional[torch.Generator] = None, 
            return_dict: bool = True,
    ) -> Union[SchedulerOutput, Tuple]:
        """
        Predict the sample from the previous timestep by reversing the SDE. This function propagates the diffusion
        process from the learned model outputs (most often the predicted noise).
        """




        
    

