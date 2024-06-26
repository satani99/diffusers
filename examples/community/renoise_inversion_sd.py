import torch 
import torch.nn.functional as F 
from PIL import Image 
from typing import List, Optional, Tuple, Union, Any, Callable, Dict
from diffusers import StableDiffusionImg2ImgPipeline, DDIMScheduler, EulerAncestralDiscreteScheduler, LCMScheduler, StableDiffusionXLImg2ImgPipeline
from diffusers.utils import BaseOutput 
from diffusers.utils.torch_utils import randn_tensor
import numpy as np 

from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion import (
    StableDiffusionPipelineOutput,
    retrieve_timesteps,
    PipelineImageInput
)

from diffusers.pipelines.stable_diffusion_xl.pipeline_stable_diffusion_xl import (
    StableDiffusionXLPipelineOutput,
    retrieve_timesteps,
    PipelineImageInput 
)


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
            [`SchedulerOutput`] or `tuple`:
                If return_dict is `True`,
                [`ScheulerOutput`] is returned,
                otherwise a tuple is returned where the first element is the sample tensor.

        """

        if (
            isinstance(timestep, int)
            or isinstance(timestep, torch.IntTensor)
            or isinstance(timestep, torch.LongTensor)
        ):
            raise ValueError(
                (
                    "Passing integer indices (e.g. from `enumerate(timesteps)` as timesteps to"
                    " `EulerDiscreteScheduler.step()` is not supported. Make sure to pass"
                    " one of the `scheduler.timesteps` as a timestep."
                ),
            )
        
        if not self.is_scale_input_called:
            logger.warning(
                "The `scale_model_input` function should be called before `step` to ensure correct denoising. "
                "See `StableDiffusionPipeline` for a usage example."
            )

        self._init_step_index(timestep.view((1)))

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
        sigma_to = self.sigmas[self.step_index+1]
        # sigma_up = (sigma_to**2 * (sigma_from**2 - sigma_to**2) / sigma_from**2) ** 0.5 
        sigma_up = (sigma_to**2 * (sigma_from**2 - sigma_to**2).abs() / sigma_from**2) ** 0.5 
        # sigma_down = sigma_to**2 / sigma_from 

        # 2. Convert to an ODE derivative 
        # derivative = (sample - pred_original_sample) / sigma 
        derivative = model_output 

        dt = sigma_down - sigma 
        # dt = sigma_down - sigma_from 

        prev_sample = sample - derivative * dt 

        device = model_output.device 
        # noise = randn_tensor(model_output.shape, dtype=model_output.dtype, device=device, generator=generator)
        # prev_sample = prev_sample + noise * sigma_up 

        prev_sample = prev_sample - self.noise_list[self.step_index] * sigma_up 

        # Cast sample back to model compatible dtype 
        prev_sample = prev_sample.to(model_output.dtype) 

        # upon completion increase step index by one 
        self._step_index += 1

        if not return_dict:
            return (prev_sample,)
        
        return SchedulerOutput(
            prev_sample=prev_sample, pred_original_sample=pred_original_sample 
        )
    
    def get_all_sigmas(self) -> torch.FloatTensor:
        sigmas = np.array(((1 - self.alphas_cumprod) / self.alphas_cumprod) ** 0.5)
        sigmas = np.concatenate([sigmas[::-1], [0.0]]).astype(np.float32)
        return torch.from_numpy(sigmas)
    
    def add_noise_off_schedule(
            self,
            original_samples: torch.FloatTensor,
            noise: torch.FloatTensor,
            timesteps: torch.FloatTensor,
    ) -> torch.FloatTensor:
        # Make sure sigmas and  timesteps have the same device and dtype as original_samples 
        sigmas = self.get_all_sigmas()
        sigmas = sigmas.to(device=original_samples.device, dtype=original_samples.dtype)
        if original_samples.device.type == "mps" and torch.is_floating_point(timesteps):
            # mps does not support float64 
            timesteps = timesteps.to(original_samples.device, dtype=torch.float32)
        else:
            timesteps = timesteps.to(original_samples.device)
            
        step_indices = 1000 - int(timesteps.item())

        sigma = sigmas[step_indices].flatten()
        while len(sigma.shape) < len(original_samples.shape):
            sigma = sigma.unsqueeze(-1)
        
        noisy_samples = original_samples + noise * sigma 
        return noisy_samples 
    

class MyLCMScheduler(LCMScheduler):

    def set_noise_list(self, noise_list):
        self.noise_list = noise_list 

    def step(
            self,
            model_output: torch.FloatTensor,
            timestep: int,
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
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`SchedulerOutput`] or `tuple`.
        Returns:
            [`SchedulerOutput`] or `tuple`:
                If return_dict is `True`, [`SchedulerOutput`] is returned, otherwise a
                tuple is returned where the first element is the sample tensor.
        """
        if self.num_inference_steps is None:
            raise ValueError(
                "Number of inference steps is 'None', you need to run 'set_timesteps' after creating the scheduler"
            )
        
        self._init_step_index(timestep)

        # 1. get previous step value 
        prev_step_index = self.step_index + 1
        if prev_step_index < len(self.timesteps):
            prev_timestep = self.timesteps[prev_step_index]
        else:
            prev_timestep = timestep 
        
        # 2. compute alphas, betas 
        alpha_prod_t = self.alphas_cumprod[timestep]
        alpha_prod_t_prev = self.alphas_cumprod[prev_timestep] if prev_timestep >= 0 else self.final_alpha_cumprod

        beta_prod_t = 1 - alpha_prod_t 
        beta_prod_t_prev = 1 - alpha_prod_t_prev 

        # 3. Get scalings for boundary conditions 
        c_skip, c_out = self.get_scalings_for_boundary_condition_discrete(timestep)

        # 4. Compute the predicted original sample x_0 based on the model parameterization 
        if self.config.prediction_type == "epsilon": # noise-prediction 
            predicted_original_sample = (sample - beta_prod_t.sqrt() * model_output) / alpha_prod_t.sqrt()
        elif self.config.prediction_type == "sample": # x-prediction 
            predicted_original_sample = model_output 
        elif self.config.prediction_type == "v_prediction": # v-prediction 
            predicted_original_sample = alpha_prod_t.sqrt() * sample - beta_prod_t.sqrt() * model_output 
        else:
            raise ValueError(
                f"prediction_type given as {self.config.prediction_type} must be one of `epsilon`, `sample` or"
                " `v_prediction` for `LCMScheduler`."
            )
        
        # 5. Clip or threshold "predicted x_0"
        if self.config.thresholding:
            predicted_original_sample = self._threshold_sample(predicted_original_sample)
        elif self.config.clip_sample:
            predicted_original_sample = predicted_original_sample.clamp(
                self.config.clip_sample_range, self.config.clip_sample_range 
            )

        # 6. Denoise model output using boundary conditions 
        denoised = c_out * predicted_original_sample + c_skip * sample 

        # 7. Sample and inject noise z ~ N(0, I) for MultiStep Inference 
        # Noise is not used on the final timestep of the timestep schedule. 
        # This also means that noise is not used for one-step sampling. 
        if self.step_index != self.num_inference_steps - 1:
            noise = self.noise_list[self.step_index]
            prev_sample = alpha_prod_t_prev.sqrt() * denoised + beta_prod_t_prev.sqrt() * noise 
        else:
            prev_sample = denoised 

        # upon completion increase step index by one 
        self._step_index += 1

        if not return_dict:
            return (prev_sample, denoised)
        
        return SchedulerOutput(prev_sample=prev_sample, predicted_original_sample=denoised)
    
    def inv_step(
            self,
            model_output: torch.FloatTensor,
            timestep: int,
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
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`SchedulerOutput`] or `tuple`.
        Returns:
            [`SchedulerOutput`] or `tuple`:
                If return_dict is `True`, [`SchedulerOutput`] is returned, otherwise a
                tuple is returned where the first element is the sample tensor.
        """
        if self.num_inference_steps is None:
            raise ValueError(
                "Number of inference steps is 'None', you need to run 'set_timesteps' after creating the scheduler"
            )
        
        self._init_step_index(timestep)

        # 1. get previous step value 
        prev_step_index = self.step_index + 1
        if prev_step_index < len(self.timesteps):
            prev_timestep = self.timesteps[prev_step_index]
        else:
            prev_timestep = timestep 

        # 2. compute alphas, betas 
        alpha_prod_t = self.alphas_cumprod[timestep]
        alpha_prod_t_prev = self.alphas_cumprod[prev_timestep] if prev_timestep >= 0 else self.final_alpha_cumprod

        beta_prod_t = 1 - alpha_prod_t 
        beta_prod_t_prev = 1 - alpha_prod_t_prev 

        # 3. Get scaling for boundary conditions 
        c_skip, c_out = self.get_scalings_for_boundary_condition_discrete(timestep)

        if self.step_index != self.num_inference_steps - 1:
            c_skip_actual = c_skip * alpha_prod_t_prev.sqrt()
            c_out_actual = c_out * alpha_prod_t_prev.sqrt()
            noise = self.noise_list[self.step_index] * beta_prod_t_prev.sqrt() 
        else:
            c_skip_actual = c_skip 
            c_out_actual = c_out 
            noise = 0

        
        dem = c_out_actual / (alpha_prod_t.sqrt()) + c_skip 
        eps_mul = beta_prod_t.sqrt() * c_out_actual / (alpha_prod_t.sqrt())

        prev_sample = (sample + eps_mul * model_output - noise) / dem 

        # upon completion increase step index by one 
        self._step_index += 1 

        if not return_dict:
            return (prev_sample, prev_sample)
        
        return SchedulerOutput(prev_sample=prev_sample, pred_original_sample=prev_sample)


# Based on code from https://github.com/pix2pixzero/pix2pix-zero
def noise_regularization(
        e_t, noise_pred_optimal, lambda_kl, lambda_ac, num_reg_steps, num_ac_rolls, generator=None
):
    for _outer in range(num_reg_steps):
        if lambda_kl > 0:
            _var = torch.autograd.Variable(e_t.detach().clone(), requires_grad=True)
            l_kld = patchify_latents_kl_divergence(_var, noise_pred_optimal)
            l_kld.backward()
            _grad = _var.grad.detach()
            _grad = torch.clip(_grad, -100, 100)
            e_t = e_t - lambda_kl * _grad 
        if lambda_ac > 0:
            for _inner in range(num_ac_rolls):
                _var = torch.autograd.Variable(e_t.detach().clone(), required_grad=True)
                l_ac = auto_corr_loss(_var, generator=generator)
                l_ac.backward()
                _grad = _var.grad.detach() / num_ac_rolls 
                e_t = e_t - lambda_ac * _grad 
        e_t = e_t.detach()

    return e_t 

# Based on code from https://github.com/pix2pixzero/pix2pix-zero
def auto_corr_lose(
        x, random_shift=True, generator=None
):
    B, C, H, W = x.shape 
    assert B == 1
    x = x.squeeze(0)
    # x must be shape [C,H,W] now  
    reg_loss = 0.0 
    for ch_idx in range(x.shape[0]):
        noise = x[ch_idx][None, None, :, :]
        while True:
            if random_shift:
                roll_amount = torch.randint(0, noise.shape[2] // 2, (1,), generator=generator).item()
            else:
                roll_amount = 1
            reg_loss += (
                noise * torch.roll(noise, shifts=roll_amount, dims=2)
            ).mean() ** 2
            reg_loss += (
                noise * torch.roll(noise, shifts=roll_amount, dims=3)
            ).mean() ** 2 
            if noise.shape[2] <= 8:
                break 
            noise = F.avg_pool2d(noise, kernel_size=2)
    return reg_loss 


def patchify_latents_kl_divergence(x0, x1, patch_size=4, num_channels=4):

    def patchify_tensor(input_tensor):
        patches = (
            input_tensor.unfold(1, patch_size, patch_size)
            .unfold(2, patch_size, patch_size)
            .unfold(3, patch_size, patch_size)
        )
        patches = patches.contiguous().view(-1, num_channels, patch_size, patch_size)
        return patches 
    
    x0 = patchify_tensor(x0)
    x1 = patchify_tensor(x1)

    kl = latents_kl_divergence(x0, x1).sum()
    return kl 

def latents_kl_divergence(x0, x1):
    EPSILON = 1e-6 
    x0 = x0.view(x0.shape[0], x0.shape[1], -1)
    x1 = x1.view(x1.shape[0], x1.shape[1], -1)
    mu0 = x0.mean(dim=-1)
    mu1 = x1.mean(dim=-1)
    var0 = x0.var(dim=-1)
    var1 = x1.var(dim=-1)
    kl = (
        torch.log((var1 + EPSILON) / (var0 + EPSILON))
        + (var0 + (mu0 - mu1) ** 2) / (var1 + EPSILON)
        - 1
    )
    kl = torch.abs(kl).sum(dim=-1)
    return kl 

def inversion_step(
        pipe,
        z_t: torch.tensor,
        t: torch.tensor,
        prompt_embeds,
        added_cond_kwargs,
        num_renoise_steps: int = 100,
        first_step_max_timestep: int = 250,
        generator=None,
) -> torch.tensor:
    extra_step_kwargs = {}
    avg_range = pipe.cfg.average_first_step_range if t.item() < first_step_max_timestep else pipe.cfg.average_step_range 
    num_renoise_steps = min(pipe.cfg.max_num_renoise_steps_first_step, num_renoise_steps) if t.item() < first_step_max_timestep else num_renoise_steps 

    noise_pred_avg = None 
    noise_pred_optimal = None 
    z_tp1_forward = pipe.scheduler.add_noise(pipe.z_0, pipe.noise, t.view((1))).detach()

    approximated_z_tp1 = z_t.clone()
    for i in range(num_renoise_steps + 1):

        with torch.no_grad():
            # if noise regularization is enabled, we need to double the batch size for the first step 
            if pipe.cfg.noise_regularization_num_reg_steps > 0 and i == 0:
                approximated_z_tp1 = torch.cat([z_tp1_forward, approximated_z_tp1])
                prompt_embeds_in = torch.cat([prompt_embeds, prompt_embeds])
                if added_cond_kwargs is not None:
                    added_cond_kwargs_in = {}
                    added_cond_kwargs_in['text_embeds'] = torch.cat([added_cond_kwargs['text_embeds'], added_cond_kwargs['text_embeds']])
                    added_cond_kwargs_in['time_ids'] = torch.cat([added_cond_kwargs['time_ids'], added_cond_kwargs['time_ids']])
                else:
                    added_cond_kwargs_in = None 
            else:
                prompt_embeds_in = prompt_embeds 
                added_cond_kwargs_in = added_cond_kwargs 

            noise_pred = unet_pass(pipe, approximated_z_tp1, t, prompt_embeds_in, added_cond_kwargs_in)

            # if noise regularization is enabled, we need to split the batch size for the first step 
            if pipe.cfg.noise_regularization_num_reg_steps > 0 and i == 0:
                noise_pred_optimal, noise_pred = noise_pred.chunk(2)
                if pipe.do_classifier_free_guidance:
                    noise_pred_optimal_uncond, noise_pred_optimal_text = noise_pred_optimal.chunk(2)
                    noise_pred_optimal = noise_pred_optimal_uncond + pipe.guidance_scale * (noise_pred_optimal_text - noise_pred_optimal_uncond)
                noise_pred_optimal = noise_pred_optimal.detach() 

            # perform guidance 
            if pipe.do_classifier_free_guidance:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + pipe.guidance_scale * (noise_pred_text - noise_pred_uncond)

            # Calculate average noise 
            if i >= avg_range[0] and i < avg_range[1]:
                j = i - avg_range[0]
                if noise_pred_avg is None:
                    noise_pred_avg = noise_pred.clone()
                else:
                    noise_pred_avg = j * noise_pred_avg / (j + 1) + noise_pred / (j + 1)
        
        if i >= avg_range[0] or (not pipe.cfg.average_latent_estimations and i > 0):
            noise_pred = noise_regularization(noise_pred, noise_pred_optimal, lambda_kl=pipe.cfg.noise_regularization_lambda_kl, lambda_ac=pipe.cfg.noise_regularization_lambda_ac, num_reg_steps=pipe.cfg.noise_regularization_num_reg_steps, num_ac_rolls=pipe.cfg.noise_regularization_num_ac_rolls, generator=generator)

        approximated_z_tp1 = pipe.scheduler.inv_step(noise_pred, t, z_t, **extra_step_kwargs, return_dict=False)[0].detach()

    # if average latents is enabled, we need to perform an additional step with the average noise 
    if pipe.cfg.average_latent_estimations and noise_pred_avg is not None:
        noise_pred_avg = noise_regularization(noise_pred_avg, noise_pred_optimal, lambda_kl=pipe.cfg.noise_regularization_lambda_kl, lambda_ac=pipe.cfg.noise_regularization_lambda_ac, num_reg_steps=pipe.cfg.noise_regularization_num_reg_steps, num_ac_rolls=pipe.cfg.noise_regularization_num_ac_rolls, generator=generator)
        approximated_z_tp1 = pipe.scheduler.inv_step(noise_pred_avg, t, z_t, **extra_step_kwargs, return_dict=False)[0].detach()

    # perform noise correction 
    if pipe.cfg.perform_noise_correction:
        noise_pred = unet_pass(pipe, approximated_z_tp1, t, prompt_embeds, added_cond_kwargs)

        # perform guidance 
        if pipe.do_classifier_free_guidance:
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + pipe.guidance_scale * (noise_pred_text - noise_pred_uncond)

        pipe.scheduler.step_and_update_noise(noise_pred, t, approximated_z_tp1, z_t, return_dict=False, optimize_epsilon_type=pipe.cfg.perform_noise_correction)

    return approximated_z_tp1 

@torch.no_grad()
def unet_pass(pipe, z_t, t, prompt_embeds, added_cond_kwargs):
    latent_model_input = torch.cat([z_t] * 2) if pipe.do_classifier_free_guidance else z_t 
    latent_model_input = pipe.scheduler.scale_model_input(latent_model_input, t)
    return pipe.unet(
        latent_model_input,
        t,
        encoder_hidden_states=prompt_embeds,
        timestep_cond=None,
        cross_attention_kwargs=pipe.cross_attention_kwargs,
        added_cond_kwargs=added_cond_kwargs,
        return_dict=False,
    )[0]

class SDDDIMPipeline(StableDiffusioImg2ImgPipeline):
    # @torch.no_grad() 
    def __call__(
            self,
            prompt: Union[str, List[str]] = None,
            image: PipelineImageInput = None,
            strength: float = 1.0,
            num_inversion_steps: Optional[int] = 50,
            timesteps: List[int] = None,
            guidance_scale: Optional[float] = 7.5,
            negative_scale: Optional[Union[str, List[str]]] = None,
            num_images_per_prompt: Optional[int] = 1,
            eta: Optional[float] = 0.0,
            generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
            prompt_embeds: Optional[torch.FloatTensor] = None,
            ip_adapter_image: Optional[PipelineImageInput] = None,
            output_type: Optional[str] = "pil",
            return_dict: bool = True,
            cross_attention_kwargs: Optional[Dict[str, Any]] = None,
            clip_skip: int = None,
            callback_on_step_end: Optional[Callable[[int, int, Dict], None]] = None,
            callback_on_step_end_tensor_inputs: List[str] = ["latents"],
            num_renoise_steps: int = 100,
            **kwargs,
    ):
        callback = kwargs.pop("callback", None)
        callback_steps = kwargs.pop("callback_steps", None)

        if callback is not None:
            deprecate(
                "callback",
                "1.0.0",
                "Passing `callback` as an input argument to `__call__` is deprecated, consider use `callback_on_step_end`",
            )
        if callback_steps is not None:
            deprecate(
                "callback_steps",
                "1.0.0",
                "Passing `callback_steps` as an input argument to `__call__` is deprecated, consider use `callback_on_step_end`",
            )

        # 1. Check inputs. Raise error if not correct 
        self.check_inputs(
            prompt,
            strength,
            callback_steps,
            negative_prompt,
            prompt_embeds,
            negative_prompt_embeds,
            callback_on_step_end_tensor_inputs,
        )

        self._guidance_scale = guidance_scale 
        self._clip_skip = clip_skip 
        self._cross_attention_kwargs = cross_attention_kwargs 

        # 2. Define call parameters 
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1 
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt) 
        else:
            batch_size = prompt_embeds.shape[0]

        device = self._execution_device 

        # 3. Encode input prompt 
        text_encoder_lora_scale = (
            self.cross_attention_kwargs.get("scale", None) if self.cross_attention_kwargs is not None else None 
        )
        prompt_embeds, negative_prompt_embeds = self.encode_prompt(
            prompt,
            device,
            num_images_per_prompt,
            self.do_classifier_free_guidance,
            negative_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            lora_scale=text_encoder_lora_scale,
            clip_skip=self.clip_skip,
        )
        # For classifier free guidance, we need to do two forward passes.
        # Here we concatenate the unconditional and text embeddings into a single batch 
        # to avoid doing two forward passes 
        if self.do_classifier_free_guidance:
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])

        if ip_adapter_image is not None:
            image_embeds, negative_image_embeds = self.encode_image(ip_adapter_image, device, num_images_per_prompt)
            if self.do_classifier_free_guidance:
                image_embeds = torch.cat([negative_image_embeds, image_embeds])

        # 4. Preprocess image 
        image = self.image_processor.preprocess(image)

        # 5. set timesteps 
        timesteps, num_inversion_steps = retrieve_timesteps(self.scheduler, num_inversion_steps, device, timesteps)
        timesteps, num_inversion_steps = self.get_timesteps(num_inversion_steps, strength, device)
        latent_timestep = timesteps[:1].repeat(batch_size * num_images_per_prompt) 

        # 6. Prepare latent variable 
        with torch.no_grad():
            latents = self.prepare_latents(
                image,
                latent_timestep,
                batch_size,
                num_images_per_prompt,
                prompt_embeds.dtype,
                device,
                generator,
            )

        # 7. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline 
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta) 

        # 7.1 Add image embeds for IP-Adapter 
        added_cond_kwargs = {"image_embeds": image_embeds} if ip_adapter_image is not None else None 

        # 7.2 Optionally get Guidance Scale Embedding 
        timestep_cond = None 
        if self.unet.config.time_cond_proj_dim is not None:
            guidance_scale_tensor = torch.tensor(self.guidance_scale - 1).repeat(batch_size * num_images_per_prompt)
            timestep_cond = self.get_guidance_scale_embedding(
                guidance_scale_tensor, embedding_dim=self.unet.config.time_cond_proj_dim 
            ).to(device=device, dtype=latents.dtype)

        # 8. Denoising loop 
        num_warmup_steps = len(timesteps) - num_inversion_steps * self.scheduler.order 

        self._num_timesteps = len(timesteps) 
        self.z_0 = torch.clone(latents) 
        self.noise = randn_tensor(self.z_0.shape, generator=generator, device=self.z_0.device, dtype=self.z_0.dtype)

        all_latents = [latents.clone()]
        with self.progress_bar(total=num_inversion_steps) as progress_bar:
            for i, t in enumerate(reversed(timesteps)):

                latents = inversion_step(self,
                                         latents,
                                         t,
                                         prompt_embeds,
                                         added_cond_kwargs,
                                         num_renoise_steps=num_renoise_steps,
                                         generator=generator)
                
                all_latents.append(latents.clone())

                if callback_on_step_end is not None:
                    callback_kwargs = {}
                    for k in callback_on_step_end_tensor_inputs:
                        callback_kwargs[k] = locals()[k]
                    callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)

                    latents = callback_outputs.pop("latents", latents)
                    prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)
                    negative_prompt_embeds = callback_outputs.pop("negative_prompt_embeds", negative_prompt_embeds)

                # call the callback, if provided 
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()
                    if callback is not None and i % callback_steps == 0:
                        step_idx = i // getattr(self.scheduler, "order", 1)
                        callback(step_idx, t, latents)

        image = latents 

        # Offload all models 
        self.maybe_free_model_hooks()

        return StableDiffusionPipelineOutput(images=image, nsfw_content_detected=None), all_latents 
    
class SDXLDDIMPipeline(StableDiffusionXLImg2ImgPipeline):
    # @torch.no_grad() 
    def __call__(
            self,
            prompt: Union[str, List[str]] = None,
            prompt_2: Optional[Union[str, List[str]]] = None,
            image: PipelineImageInput = None,
            strength: float = 0.3,
            num_inversion_steps: int = 50,
            timesteps: List[int] = None,
            denoising_start: Optional[float] = None,
            denoising_end: Optional[float] = None,
            guidance_scale: float = 1.0,
            negative_prompt: Optional[Union[str, List[str]]] = None,
            negative_prompt_2: Optional[Union[str, List[str]]] = None,
            num_images_per_prompt: Optional[int] = 1,
            eta: float = 0.0,
            generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
            latents: Optional[torch.FloatTensor] = None,
            prompt_embeds: Optional[torch.FloatTensor] = None,
            negative_prompt_embeds: Optional[torch.FloatTensor] = None,
            pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
            negative_pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
            ip_adapter_image: Optional[PipelineImageInput] = None,
            output_type: Optional[str] = "pil",
            return_dict: bool = True,
            cross_attention_kwargs: Optional[Dict[str, Any]] = None,
            guidance_rescale: float = 0.0,
            original_size: Tuple[int, int] = None,
            crops_coords_top_left: Tuple[int, int] = (0, 0),
            target_size: Tuple[int, int] = None,
            negative_original_size: Optional[Tuple[int, int]] = None,
            negative_crops_coords_top_left: Tuple[int, int] = (0, 0),
            negative_target_size: Optional[Tuple[int, int]] = None,
            aesthetic_score: float = 6.0,
            negative_aesthetic_score: float = 2.5,
            clip_skip: Optional[int] = None,
            callback_on_step_end: Optional[Callable[[int, int, Dict], None]] = None,
            callback_on_step_end_tensor_inputs: List[str] = ["latents"],
            num_renoise_steps: int = 100,
            **kwargs,
    ):
        callback = kwargs.pop("callback", None)
        callback_steps = kwargs.pop("callback_steps", None)

        if callback is not None:
            deprecate(
                "callback",
                "1.0.0",
                "Passing `callback` as an input argument to `__call__` is deprecated, consider use `callback_on_step_end`",
            )
        if callback_steps is not None:
            deprecate(
                "callback_steps",
                "1.0.0",
                "Passing `callback_steps` as an input argument to `__call__` is deprecated, consider use `callback_on_step_end`",
            )
        
        # 1.Check inputs. Raise error if not correct 
        self.check_inputs(
            prompt,
            prompt_2,
            strength,
            num_inversion_steps,
            callback_steps,
            negative_prompt,
            negative_prompt_2,
            prompt_embeds,
            negative_prompt_embeds,
            callback_on_step_end_tensor_inputs,
        )

        self._guidance_scale = guidance_scale 
        self._guidance_rescale = guidance_rescale 
        self._clip_skip = clip_skip 
        self._cross_attention_kwargs = cross_attention_kwargs 
        self._denoising_end = denoising_end 
        self._denoising_start = denoising_start 

        # 2. Define call parameters 
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1 
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        device = self._execution_device 

        # 3. Encoder input prompt 
        text_encoder_lora_scale = (
            self.cross_attention_kwargs.get("scale", None) if self.cross_attention_kwargs is not None else None 
        )
        (
            prompt_embeds,
            negative_prompt_embeds,
            pooled_prompt_embeds,
            negative_pooled_prompt_embeds,
        ) = self.encode_prompt(
            prompt=prompt,
            prompt_2=prompt_2,
            device=device,
            num_images_per_prompt=num_images_per_prompt,
            do_classifier_free_guidance=self.do_classifier_free_guidance,
            negative_prompt=negative_prompt,
            negative_prompt_2=negative_prompt_2,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
            lora_scale=text_encoder_lora_scale,
            clip_skip=self.clip_skip,
        )

        # 4. Preprocess image 
        image = self.image_processor.preprocess(image)

        # 5. Prepare timesteps 
        def denoising_value_valid(dnv):
            return isinstance(self.denoising_end, float) and 0 < dnv < 1 
        
        timesteps, num_inversion_steps = retrieve_timesteps(self.scheduler, num_inversion_steps, device, timesteps)

        timesteps, num_inversion_steps = self.get_timesteps(
            num_inversion_steps,
            strength,
            device,
            denoising_start=self.denoising_start if denoising_value_valid else None,
        )

        # 6. Prepare latent variables 
        with torch.no_grad():
            latents = self.prepare_latents(
                image,
                None,
                batch_size,
                num_images_per_prompt,
                prompt_embeds.dtype,
                device,
                generator,
                False,
            )

        # 7. Prepare extra step kwargs. 
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        height, width = latents.shape[-2:]
        height = height * self.vae_scale_factor 
        width = width * self.vae_scale_factor 

        original_size = original_size or (height, width)
        target_size = target_size or (height, width)

        # 8. Prepare added time ids & embeddings 
        if negative_original_size is None:
            negative_original_size = original_size 
        if negative_target_size is None:
            negative_target_size = target_size 
        
        add_text_embeds = pooled_prompt_embeds 
        if self.text_encoder_2 is None:
            text_encoder_projection_dim = int(pooled_prompt_embeds.shape[-1])
        else:
            text_encoder_projection_dim = self.text_encoder_2.config.projection_dim 

        add_time_ids, add_neg_time_ids = self._get_add_time_ids(
            original_size,
            crops_coords_top_left,
            target_size,
            aesthetic_score,
            negative_aesthetic_score,
            negative_original_size,
            negative_crops_coords_top_left,
            negative_target_size,
            dtype=prompt_embeds.dtype,
            text_encoder_projection_dim=text_encoder_projection_dim,
        )
        add_time_ids = add_time_ids.repeat(batch_size * num_images_per_prompt, 1)

        if self.do_classifier_free_guidance:
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
            add_text_embeds = torch.cat([negative_pooled_prompt_embeds, add_text_embeds], dim=0)
            add_neg_time_ids = add_neg_time_ids.repeat(batch_size * num_images_per_prompt, 1)
            add_time_ids = torch.cat([add_neg_time_ids, add_time_ids], dim=0)

        prompt_embeds = prompt_embeds.to(device)
        add_text_embeds = add_text_embeds.to(device)
        add_time_ids = add_time_ids.to(device)

        if ip_adapter_image is not None:
            image_embeds, negative_image_embeds = self.encode_image(ip_adapter_image, device, num_images_per_prompt)
            if self.do_classifier_free_guidance:
                image_embeds = torch.cat([negative_image_embeds, image_embeds])
                image_embeds = image_embeds.to(device)

        # 9. Denoising loop 
        num_warmup_steps = max(len(timesteps) - num_inversion_steps * self.scheduler.order, 0)

        self._num_timesteps = len(timesteps)
        self.z_0 = torch.clone(latents)
        self.noise = randn_tensor(self.z_0.shape, generator=generator, device=self.z_0.device, dtype=self.z_0.dtype)

        all_latents = [latents.clone()]
        with self.progress_bar(total=num_inversion_steps) as progress_bar:
            for i, t in enumerate(reversed(timesteps)):

                added_cond_kwargs = {"text_embeds": add_text_embeds, "time_ids": add_time_ids}
                if ip_adapter_image is not None:
                    added_cond_kwargs["image_embeds"] = image_embeds 

                latents = inversion_step(self,
                                         latents,
                                         t,
                                         prompt_embeds,
                                         added_cond_kwargs,
                                         num_renoise_steps=num_renoise_steps,
                                         generator=generator)
                
                all_latents.append(latents.clone())

                if callback_on_step_end is not None:
                    callback_kwargs = {}
                    for k in callback_on_step_end_tensor_inputs:
                        callback_kwargs[k] = locals()[k]
                    callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)

                    latents = callback_outputs.pop("latents", latents)
                    prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)
                    negative_prompt_embeds = callback_outputs.pop("negative_prompt_embeds", negative_prompt_embeds)
                    add_text_embeds = callback_outputs.pop("add_text_embeds", add_text_embeds)
                    negative_pooled_prompt_embeds = callback_outputs.pop(
                        "negative_pooled_prompt_embeds", negative_pooled_prompt_embeds 
                    )
                    add_time_ids = callback_outputs.pop("add_time_ids", add_time_ids)
                    add_neg_time_ids = callback_output.pop("add_neg_time_ids", add_neg_time_ids)

                # call the callback, if provided 
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()
                    if callback is not None and i % callback_steps == 0:
                        step_idx = i // getattr(self.scheduler, "order", 1)
                        callback(step_idx, t, latents)

        image = latents 

        # Offload all models 
        self.maybe_free_model_hooks() 

        return StableDiffusionXLPipelineOutput(image=image), all_latents
    

def _get_pipes(model_name, device):
    if 'xl' in model_name.lower():
        pipeline_inf, pipeline_inv = StableDiffusionXLImg2ImgPipeline, SDXLDDIMPipeline
    else:
        pipeline_inf, pipeline_inv = StableDiffusionImg2ImgPipeline, SDDDIMPipeline 
    
    if 'xl' in model_name.lower():
        pipe_inference = pipeline_inf.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            use_safetensors=True,
            variant="fp16",
            safety_checker=None,
        ).to(device)
    else:
        pipe_inference = pipeline_inf.from_pretrained(
            model_name,
            use_safetensors=True,
            safety_checker=None,
        ).to(device)

    pipe_inversion = pipeline_inv(**pipe_inference.components)

    return pipe_inversion, pipe_inference 

def get_pipes(model_name, scheduler_name, device="cuda"):
    if scheduler_name.lower() == "ddim":
        scheduler_class = MyDDIMScheduler
    elif scheduler_name.lower() == "euler":
        scheduler_class = MyEulerAncestralDiscreteScheduler
    elif scheduler_name.lower() == "lcm":
        scheduler_class = MyLCMScheduler 
    else:
        raise ValueError("Unknown scheduler type")
    
    pipe_inversion, pipe_inference = _get_pipes(model_name, device)

    pipe_inference.scheduler = scheduler_class.from_config(pipe_inference.scheduler.config)
    pipe_inversion.scheduler = scheduler_class.from_config(pipe_inversion.scheduler.config)

    if not "xl" in model_name.lower():
        pipe_inference.scheduler.add_noise = lambda init_latents, noise, timestep: init_latents 
        pipe_inversion.scheduler.add_noise = lambda init_latents, noise, timestep: init_latents 

    if "lcm" in scheduler_name.lower() and "xl" in model_name.lower():
        adapter_id = "latent-consistency/lcm-lora-sdxl"
        pipe_inversion.load_lora_weights(adapter_id)
        pipe_inference.load_lora_weights(adapter_id)
    
    return pipe_inversion, pipe_inference 

def create_noise_list(img_size, length, generator=None):
    VQAE_SCALE = 8
    latents_size = (1, 4, img_size[0] // VQAE_SCALE, img_size[1] // VQAE_SCALE)
    return [randn_tensor(latents_size, dtype=torch.float16, device=torch.device("cuda:0"), generator=generator) for i in range(length)]

def invert(init_image: Image,
           prompt: str,
           seed: 42,
           scheduler_name="euler",
           num_inversion_steps=4,
           num_inference_steps=4,
           guidance_scale=0.0,
           inversion_max_step=1.0,
           num_renoise_steps=9,
           pipe_inversion,
           pipe_inference,
           latents = None,
           edit_prompt = None,
           edit_cfg = 1.0,
           noise = None,
           do_reconstruction = True):
    
    generator = torch.Generator().manual_seed(seed)

    if "ddim" != scheduler_name.lower():
        if latents is None:
            noise = create_noise_list((512, 512), num_inversion_steps, generator=generator)
        pipe_inversion.scheduler.set_noise_list(noise)
        pipe_inference.shceduler.set_noise_list(noise)

    all_latents = None 

    if latents is None:
        print("Inverting...")
        res = pipe_inversion(prompt = prompt,
                             num_inversion_steps = num_inversion_steps,
                             num_inference_steps = num_inference_steps,
                             generator = generator,
                             image = init_image,
                             guidance_scale = guidance_scale,
                             strength = inversion_max_step,
                             denoising_start = 1.0-inversion_max_step,
                             num_renoise_steps = num_renoise_steps)
        latents = res[0][0]
        all_latents = res[1] 

        inv_latent = latents.clone()

    if do_reconstruction:
        print("Generating...")
        edit_prompt = prompt if edit_prompt is None else edit_prompt 
        guidance_scale = edit_cfg 
        img = pipe_inference(prompt = edit_prompt,
                                num_inference_steps = num_inference_steps,
                                negative_prompt = prompt,
                                image = latents,
                                strength = inversion_max_step,
                                denoising_start = 1.0-inversion_max_step,
                                guidance_scale = guidance_scale).images[0]
    else:
        img = None 

    return img, inv_latent, noise, all_latents 

