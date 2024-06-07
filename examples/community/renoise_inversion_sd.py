import torch 
import torch.nn.functional as F 
from PIL import Image 
from typing import List, Optional, Tuple, Union
from diffusers import StableDiffusionImg2ImgPipeline, StableDiffusionXLImg2ImgPipeline, DDIMScheduler, EulerAncestralDiscreteScheduler, LCMScheduler
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