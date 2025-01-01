# Copied from https://github.com/lilygoli/SpotLessSplats/blob/main/examples/datasets/sd_feature_extraction.ipynb
# pip install diffusers==0.27.2 transformers==4.40.1

import os
import random
from glob import glob
from tqdm import tqdm
import argparse
import PIL.Image
import distibuted_tasks

import gc
import numpy as np
import torch
import torch.nn.functional as F
from typing import Any, Callable, Dict, List, Optional, Union
from diffusers.models.unets.unet_2d_condition import UNet2DConditionModel
from diffusers import DDIMScheduler
from diffusers import StableDiffusionPipeline


class MyUNet2DConditionModel(UNet2DConditionModel):
    def forward(
            self,
            sample: torch.FloatTensor,
            timestep: Union[torch.Tensor, float, int],
            up_ft_indices,
            encoder_hidden_states: torch.Tensor,
            deform=None,
            class_labels: Optional[torch.Tensor] = None,
            timestep_cond: Optional[torch.Tensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            cross_attention_kwargs: Optional[Dict[str, Any]] = None):
        r"""
        Args:
            sample (`torch.FloatTensor`): (batch, channel, height, width) noisy inputs tensor
            timestep (`torch.FloatTensor` or `float` or `int`): (batch) timesteps
            encoder_hidden_states (`torch.FloatTensor`): (batch, sequence_length, feature_dim) encoder hidden states
            cross_attention_kwargs (`dict`, *optional*):
                A kwargs dictionary that if specified is passed along to the `AttnProcessor` as defined under
                `self.processor` in
                [diffusers.cross_attention](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/cross_attention.py).
        """
        # By default samples have to be AT least a multiple of the overall upsampling factor.
        # The overall upsampling factor is equal to 2 ** (# num of upsampling layears).
        # However, the upsampling interpolation output size can be forced to fit any upsampling size
        # on the fly if necessary.
        default_overall_up_factor = 2 ** self.num_upsamplers

        # upsample size should be forwarded when sample is not a multiple of `default_overall_up_factor`
        forward_upsample_size = False
        upsample_size = None

        if any(s % default_overall_up_factor != 0 for s in sample.shape[-2:]):
            # logger.info("Forward upsample size to force interpolation output size.")
            forward_upsample_size = True

        # prepare attention_mask
        if attention_mask is not None:
            attention_mask = (1 - attention_mask.to(sample.dtype)) * -10000.0
            attention_mask = attention_mask.unsqueeze(1)

        # 0. center input if necessary
        if self.config.center_input_sample:
            sample = 2 * sample - 1.0

        # 1. time
        timesteps = timestep
        if not torch.is_tensor(timesteps):
            # TODO: this requires sync between CPU and GPU. So try to pass timesteps as tensors if you can
            # This would be a good case for the `match` statement (Python 3.10+)
            is_mps = sample.device.type == "mps"
            if isinstance(timestep, float):
                dtype = torch.float32 if is_mps else torch.float64
            else:
                dtype = torch.int32 if is_mps else torch.int64
            timesteps = torch.tensor([timesteps], dtype=dtype, device=sample.device)
        elif len(timesteps.shape) == 0:
            timesteps = timesteps[None].to(sample.device)

        # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
        timesteps = timesteps.expand(sample.shape[0])

        t_emb = self.time_proj(timesteps)

        # timesteps does not contain any weights and will always return f32 tensors
        # but time_embedding might actually be running in fp16. so we need to cast here.
        # there might be better ways to encapsulate this.
        t_emb = t_emb.to(dtype=self.dtype)

        emb = self.time_embedding(t_emb, timestep_cond)

        if self.class_embedding is not None:
            if class_labels is None:
                raise ValueError("class_labels should be provided when num_class_embeds > 0")

            if self.config.class_embed_type == "timestep":
                class_labels = self.time_proj(class_labels)

            class_emb = self.class_embedding(class_labels).to(dtype=self.dtype)
            emb = emb + class_emb

        # 2. pre-process
        sample = self.conv_in(sample)

        # 3. down
        down_block_res_samples = (sample,)
        for downsample_block in self.down_blocks:
            if hasattr(downsample_block, "has_cross_attention") and downsample_block.has_cross_attention:
                sample, res_samples = downsample_block(
                    hidden_states=sample,
                    temb=emb,
                    encoder_hidden_states=encoder_hidden_states,
                    attention_mask=attention_mask,
                    cross_attention_kwargs=cross_attention_kwargs,
                )
            else:
                sample, res_samples = downsample_block(hidden_states=sample, temb=emb)

            down_block_res_samples += res_samples

        # 4. mid
        if self.mid_block is not None:
            sample = self.mid_block(
                sample,
                emb,
                encoder_hidden_states=encoder_hidden_states,
                attention_mask=attention_mask,
                cross_attention_kwargs=cross_attention_kwargs,
            )

        # 5. up
        up_ft = {}
        for i, upsample_block in enumerate(self.up_blocks):

            is_final_block = i == len(self.up_blocks) - 1

            res_samples = down_block_res_samples[-len(upsample_block.resnets):]
            down_block_res_samples = down_block_res_samples[: -len(upsample_block.resnets)]

            # if we have not reached the final block and need to forward the
            # upsample size, we do it here
            if not is_final_block and forward_upsample_size:
                upsample_size = down_block_res_samples[-1].shape[2:]

            if hasattr(upsample_block, "has_cross_attention") and upsample_block.has_cross_attention:
                sample = upsample_block(
                    hidden_states=sample,
                    temb=emb,
                    res_hidden_states_tuple=res_samples,
                    encoder_hidden_states=encoder_hidden_states,
                    cross_attention_kwargs=cross_attention_kwargs,
                    upsample_size=upsample_size,
                    attention_mask=attention_mask,
                )
            else:
                sample = upsample_block(
                    hidden_states=sample, temb=emb, res_hidden_states_tuple=res_samples, upsample_size=upsample_size
                )

            if i in up_ft_indices:
                up_ft[i] = sample.detach()
                if deform is not None:
                    dot = sample.permute(0, 2, 3, 1).reshape((-1, sample.shape[1])) @ deform.unsqueeze(1)
                    dot = dot.reshape((sample.shape[0], 1, sample.shape[-2], sample.shape[-1]))
                    dot = dot / torch.norm(sample, dim=1, keepdim=True)
                    sample = (1 + dot) * sample

        # 6. post-process
        if self.conv_norm_out:
            sample = self.conv_norm_out(sample)
            sample = self.conv_act(sample)
        sample = self.conv_out(sample)

        output = {}
        output['up_ft'] = up_ft
        return output, sample


class OneStepSDPipeline(StableDiffusionPipeline):
    @torch.no_grad()
    def __call__(
            self,
            img_tensor,
            t,
            up_ft_indices,
            negative_prompt: Optional[Union[str, List[str]]] = None,
            generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
            prompt_embeds: Optional[torch.FloatTensor] = None,
            callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
            callback_steps: int = 1,
            cross_attention_kwargs: Optional[Dict[str, Any]] = None,
            deform=None,
            noise=None
    ):
        device = self._execution_device
        latents = self.vae.encode(img_tensor).latent_dist.sample() * self.vae.config.scaling_factor
        t = torch.tensor(t, dtype=torch.long, device=device)
        if noise is None:
            noise = torch.randn_like(latents).to(device)
        latents_noisy = self.scheduler.add_noise(latents, noise, t)
        unet_output, noise_pred = self.unet(latents_noisy,
                                            t,
                                            up_ft_indices,
                                            encoder_hidden_states=prompt_embeds,
                                            cross_attention_kwargs=cross_attention_kwargs,
                                            deform=deform)
        # compute the previous noisy sample x_t -> x_t-1
        latents_clean = self.scheduler.step(noise_pred, t, latents_noisy).pred_original_sample
        # print(latents_clean.shape, noise_pred.shape)

        # scale and decode the image latents with vae
        latents_clean = 1 / self.vae.config.scaling_factor * latents_clean
        image = self.vae.decode(latents_clean).sample

        return unet_output, image


class SDFeaturizer:
    def __init__(self, sd_id='stabilityai/stable-diffusion-2-1', index=1):
        unet = MyUNet2DConditionModel.from_pretrained(sd_id, subfolder="unet", use_safetensors=False)
        onestep_pipe = OneStepSDPipeline.from_pretrained(sd_id, unet=unet, safety_checker=None, use_safetensors=False)
        onestep_pipe.scheduler = DDIMScheduler.from_pretrained(sd_id, subfolder="scheduler", use_safetensors=False)
        onestep_pipe.scheduler.set_timesteps(50)
        gc.collect()
        onestep_pipe = onestep_pipe.to("cuda")
        onestep_pipe.enable_attention_slicing()
        self.pipe = onestep_pipe

    @torch.no_grad()
    def forward(self,
                img_tensor,  # single image, [1,c,h,w]
                prompt,
                deform=None,
                t=261,
                up_ft_index=[1],
                ensemble_size=8, noise=None):

        img_tensor = img_tensor.repeat(ensemble_size, 1, 1, 1).cuda()  # ensem, c, h, w

        prompt_embeds = self.pipe.encode_prompt(
            prompt=prompt,
            device='cuda',
            num_images_per_prompt=1,
            do_classifier_free_guidance=False)[0]  # [1, 77, dim]
        prompt_embeds = prompt_embeds.repeat(ensemble_size, 1, 1)
        unet_ft_all, image = self.pipe(
            img_tensor=img_tensor,
            t=t,
            up_ft_indices=up_ft_index,
            prompt_embeds=prompt_embeds,
            deform=deform, noise=noise)
        fts = []
        mx_shape = 0, 0
        for i in up_ft_index:
            unet_ft = unet_ft_all['up_ft'][i]  # ensem, c, h, w
            unet_ft = unet_ft.mean(0, keepdim=True)  # 1,c,h,w
            mx_shape = max(mx_shape[0], unet_ft.shape[-2]), max(mx_shape[0], unet_ft.shape[-1])
            fts += [unet_ft]
        fts_resized = []
        for i in range(len(up_ft_index)):
            fts_resized += F.interpolate(fts[i], size=(mx_shape[0], mx_shape[1]), mode='bilinear')

        unet_ft_all = torch.cat(fts_resized, dim=0)  # n,c,h,w
        return unet_ft_all, image


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("image_dir", type=str)
    parser.add_argument("--output", "-o", type=str, default=None)
    parser.add_argument("--extensions", "-e", nargs="+", type=str, default=["jpg", "JPG", "jpeg", "JPEG"])
    parser.add_argument("--image-size", "-s", type=int, default=800)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--image_list", "--image-list", type=str, default=None)
    parser.add_argument("--nerfw_tsv", type=str, default=None)
    distibuted_tasks.configure_arg_parser(parser)

    return parser.parse_args()


def main():
    # parse args
    args = parse_args()
    if args.output is None:
        args.output = os.path.join(os.path.dirname(args.image_dir.rstrip("/")), "SD")
    img_size = args.image_size

    # seed
    torch.random.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    # find images
    image_list = []
    if args.image_list is not None:
        with open(args.image_list, "r") as f:
            image_list = [os.path.join(args.image_dir, i) for i in f]
    elif args.nerfw_tsv is not None:
        import csv
        with open(args.nerfw_tsv, "r") as f:
            rd = csv.DictReader(f, delimiter="\t", quotechar='"')
            for i in rd:
                image_list.append(os.path.join(args.image_dir, i["filename"]))
    else:
        for ext in args.extensions:
            image_list += list(glob(os.path.join(args.image_dir, "**/*.{}".format(ext)), recursive=True))
        image_list.sort()
    n_found_images = len(image_list)

    # get an image list slice
    image_list = distibuted_tasks.get_task_list_with_args(args, image_list)
    print("extract features from {} of {} images".format(len(image_list), n_found_images))

    dift = SDFeaturizer()

    with torch.no_grad(), tqdm(image_list) as t:
        for image_path in t:
            # build image path
            image_relative_path = image_path[len(args.image_dir):].lstrip("/")
            image_relative_path_without_ext = image_relative_path[:image_relative_path.rfind(".")]
            t.set_description(image_relative_path)

            img = PIL.Image.open(image_path).convert('RGB')
            img = img.resize((img_size, img_size))
            img_tensor = (torch.tensor(np.array(img)) / 255.0 - 0.5) * 2
            img_tensor = img_tensor.permute(2, 0, 1)

            fts, image = dift.forward(img_tensor,
                                      prompt='',
                                      ensemble_size=4,
                                      t=261,
                                      up_ft_index=[1, ])

            # save
            output_path = os.path.join(args.output, image_relative_path_without_ext + ".npy")
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            np.save(output_path, fts.cpu().numpy())

    print("Saved to `{}`".format(args.output))


if __name__ == "__main__":
    main()
