import torch
from diffusers.utils.torch_utils import randn_tensor
from typing import List, Optional, Tuple, Union

class ScoreSdeVeInference():
        r"""
        The call function to the pipeline for generation.

        Args:
            batch_size (`int`, *optional*, defaults to 1):
                The number of images to generate.
            generator (`torch.Generator`, `optional`):
                A [`torch.Generator`](https://pytorch.org/docs/stable/generated/torch.Generator.html) to make
                generation deterministic.
            output_type (`str`, `optional`, defaults to `"pil"`):
                The output format of the generated image. Choose between `PIL.Image` or `np.array`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`ImagePipelineOutput`] instead of a plain tuple.

        Returns:
            [`~pipelines.ImagePipelineOutput`] or `tuple`:
                If `return_dict` is `True`, [`~pipelines.ImagePipelineOutput`] is returned, otherwise a `tuple` is
                returned where the first element is a list with the generated images.
        """
        def __init__(self, model, scheduler, num_inference_steps):
              self.model = model
              self.scheduler = scheduler
              self.num_inference_steps = num_inference_steps
        
        def inference(self, tho_spec, device=torch.device("cpu")):
            # tho_spec = tho_spec.unsqueeze(dim=1)
            generator = None
            img_size = self.model.sample_size
            shape = (tho_spec.shape[0], 1, img_size, img_size)
            image = randn_tensor(shape) * self.scheduler.init_noise_sigma
            image = image.to(device)

            self.scheduler.set_timesteps(self.num_inference_steps)

            # noise_scheduler.set_sigmas(num_inference_steps)

            for i, t in enumerate(self.scheduler.timesteps):
                sigma_t = self.scheduler.sigmas[i] * torch.ones(shape[0], device=device)

                # correction step
                for _ in range(self.scheduler.config.correct_steps):
                    sample_with_spec = torch.cat((image, tho_spec), dim=1)
                    model_output = self.model(sample_with_spec, sigma_t).sample
                    image = self.scheduler.step_correct(model_output, image, generator=generator).prev_sample

                # prediction step
                image_with_spec = torch.cat((image, tho_spec), dim=1)
                #MT : with lucydrains
                # t_mod = t.repeat(8).to(torch.device("cuda:0"))
                # model_output = self.model(image_with_spec, t_mod)
                #MT: with diffusers
                model_output = self.model(image_with_spec, sigma_t).sample
                image = self.scheduler.step_pred(model_output, t, image, generator=generator).prev_sample

                print('HHHHHHHHHHHHH')
                print(i)
                print(t)
                print(sigma_t)
                print(torch.min(image))
                print(torch.max(image))


            print('KKKKK')
            print(torch.min(image))
            print(torch.max(image))

            image = (image / 2 + 0.5).clamp(0, 1)
            # sample = sample.cpu().permute(0, 2, 3, 1).numpy()

            return image

class OldScoreSdeVeInference():
        r"""
        The call function to the pipeline for generation.

        Args:
            batch_size (`int`, *optional*, defaults to 1):
                The number of images to generate.
            generator (`torch.Generator`, `optional`):
                A [`torch.Generator`](https://pytorch.org/docs/stable/generated/torch.Generator.html) to make
                generation deterministic.
            output_type (`str`, `optional`, defaults to `"pil"`):
                The output format of the generated image. Choose between `PIL.Image` or `np.array`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`ImagePipelineOutput`] instead of a plain tuple.

        Returns:
            [`~pipelines.ImagePipelineOutput`] or `tuple`:
                If `return_dict` is `True`, [`~pipelines.ImagePipelineOutput`] is returned, otherwise a `tuple` is
                returned where the first element is a list with the generated images.
        """
        def __init__(self, model, scheduler, num_inference_steps):
              self.model = model
              self.scheduler = scheduler
              self.num_inference_steps = num_inference_steps

        def inference(self, tho_spec, device=torch.device("cpu")):
            # tho_spec = tho_spec.unsqueeze(dim=1)
            generator = None
            img_size = self.model.sample_size
            shape = (tho_spec.shape[0], 1, img_size, img_size)
            sample = randn_tensor(shape) * self.scheduler.init_noise_sigma
            sample = sample.to(device)

            self.scheduler.set_timesteps(self.num_inference_steps)
            self.scheduler.set_sigmas(self.num_inference_steps)

            for i, t in enumerate(self.scheduler.timesteps):
                sigma_t = self.scheduler.sigmas[i] * torch.ones(shape[0], device=device)

                # # correction step
                for _ in range(self.scheduler.config.correct_steps):
                    sample_with_spec = torch.cat((sample, tho_spec), dim=1)
                    model_output = self.model(sample_with_spec, sigma_t).sample
                    sample = self.scheduler.step_correct(model_output, sample, generator=generator).prev_sample

                # prediction step
                sample_with_spec = torch.cat((sample, tho_spec), dim=1)
                model_output = self.model(sample_with_spec, sigma_t).sample
                output = self.scheduler.step_pred(model_output, t, sample, generator=generator)

                sample, sample_mean = output.prev_sample, output.prev_sample_mean

            sample = (sample_mean / 2 + 0.5).clamp(0, 1)
            return sample




class DDPMInference():
        r"""
        The call function to the pipeline for generation.

        Args:
            batch_size (`int`, *optional*, defaults to 1):
                The number of images to generate.
            generator (`torch.Generator`, `optional`):
                A [`torch.Generator`](https://pytorch.org/docs/stable/generated/torch.Generator.html) to make
                generation deterministic.
            output_type (`str`, `optional`, defaults to `"pil"`):
                The output format of the generated image. Choose between `PIL.Image` or `np.array`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`ImagePipelineOutput`] instead of a plain tuple.

        Returns:
            [`~pipelines.ImagePipelineOutput`] or `tuple`:
                If `return_dict` is `True`, [`~pipelines.ImagePipelineOutput`] is returned, otherwise a `tuple` is
                returned where the first element is a list with the generated images.
        """
        def __init__(self, model, scheduler, num_inference_steps):
              self.model = model
              self.scheduler = scheduler
              self.num_inference_steps = num_inference_steps
        
        def inference(self, tho_spec, device=torch.device("cpu")):
            # tho_spec = tho_spec.unsqueeze(dim=1)
            generator = None
            img_size = self.model.sample_size
            shape = (tho_spec.shape[0], 1, img_size, img_size)
            image = randn_tensor(shape) * self.scheduler.init_noise_sigma
            image = image.to(device)

            self.scheduler.set_timesteps(self.num_inference_steps)

            # noise_scheduler.set_sigmas(num_inference_steps)

            for i, t in enumerate(self.scheduler.timesteps):
                # sigma_t = noise_scheduler.sigmas[i] * torch.ones(shape[0], device=device)

                # # correction step
                # for _ in range(noise_scheduler.config.correct_steps):
                #     sample_with_spec = torch.cat((sample, tho_spec), dim=1)
                #     model_output = model(sample_with_spec, sigma_t).sample
                #     sample = noise_scheduler.step_correct(model_output, sample, generator=generator).prev_sample

                # prediction step
                image_with_spec = torch.cat((image, tho_spec), dim=1)
                #MT : with lucydrains
                # t_mod = t.repeat(8).to(torch.device("cuda:0"))
                # model_output = self.model(image_with_spec, t_mod)
                #MT: with diffusers
                model_output = self.model(image_with_spec, t).sample
                image = self.scheduler.step(model_output, t, image, generator=generator).prev_sample

            image = (image / 2 + 0.5).clamp(0, 1)
            # sample = sample.cpu().permute(0, 2, 3, 1).numpy()

            return image