"""Clean-room DMD2 trainer for Qwen-Image-Edit.

This implementation follows the algorithmic ingredients described by the DMD2
paper, without copying its CC BY-NC-SA reference implementation:

* distribution matching without a teacher-generated regression-pair dataset;
* a faster-updated fake score (two-time-scale training);
* an adversarial loss between generated and real *image latents*; and
* student rollouts on the configured inference schedule before score matching.

The discriminator is conditional on Qwen's multimodal prompt embeddings.  For
image editing those embeddings already include the reference-image signal, so
it can judge whether a target latent is realistic under the edit condition.
"""

from __future__ import annotations

import os
from contextlib import contextmanager

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger

from lightx2v_train.runtime.distributed import (
    barrier,
    get_world_size,
    is_distributed,
    is_main_process,
    reduce_mean,
)
from lightx2v_train.runtime.sequence_parallel import broadcast_sequence_parallel_value
from lightx2v_train.utils.registry import TRAINER_REGISTER

from .image_edit_trainer import ImageEditDmdTrainer


class ConditionalLatentPatchDiscriminator(nn.Module):
    """Small conditional PatchGAN over one-frame Qwen VAE latents.

    DMD2's adversarial term works in the image-generation representation.  A
    Qwen VAE latent is that representation during denoising, avoids an
    expensive decoder backward pass, and remains condition-aware through the
    Qwen-VL prompt embedding.
    """

    def __init__(self, latent_channels: int, condition_dim: int, base_channels: int = 64):
        super().__init__()
        base_channels = int(base_channels)
        if base_channels < 8:
            raise ValueError("DMD2 GAN base_channels must be at least 8.")
        channels = (base_channels, base_channels * 2, base_channels * 4)
        self.features = nn.Sequential(
            nn.Conv2d(latent_channels, channels[0], kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(channels[0], channels[1], kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(channels[1], channels[2], kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.condition = nn.Sequential(
            nn.Linear(condition_dim, channels[2]),
            nn.SiLU(),
            nn.Linear(channels[2], channels[2]),
        )
        self.head = nn.Conv2d(channels[2], 1, kernel_size=3, stride=1, padding=1)

    @staticmethod
    def _pool_condition(prompt_embeds: torch.Tensor, prompt_embed_mask: torch.Tensor | None) -> torch.Tensor:
        if prompt_embeds.ndim != 3:
            raise ValueError(
                "DMD2 discriminator expects prompt embeddings with shape [batch, tokens, channels], "
                f"got {tuple(prompt_embeds.shape)}."
            )
        if prompt_embed_mask is None:
            return prompt_embeds.mean(dim=1)
        mask = prompt_embed_mask.to(device=prompt_embeds.device, dtype=prompt_embeds.dtype)
        if mask.ndim != 2:
            raise ValueError(f"DMD2 prompt_embed_mask must be [batch, tokens], got {tuple(mask.shape)}.")
        mask = mask.unsqueeze(-1)
        return (prompt_embeds * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)

    def forward(self, latents: torch.Tensor, prompt_embeds: torch.Tensor, prompt_embed_mask: torch.Tensor | None) -> torch.Tensor:
        if latents.ndim != 5 or latents.shape[2] != 1:
            raise ValueError(
                "DMD2 Qwen-Image-Edit GAN expects one-frame latents [batch, channels, 1, height, width], "
                f"got {tuple(latents.shape)}."
            )
        features = self.features(latents[:, :, 0].float())
        condition = self.condition(self._pool_condition(prompt_embeds.float(), prompt_embed_mask))
        return self.head(features + condition[:, :, None, None])


@contextmanager
def _frozen_parameters(module: nn.Module):
    requires_grad = [parameter.requires_grad for parameter in module.parameters()]
    try:
        for parameter in module.parameters():
            parameter.requires_grad_(False)
        yield
    finally:
        for parameter, value in zip(module.parameters(), requires_grad):
            parameter.requires_grad_(value)


@TRAINER_REGISTER("image_edit_dmd2")
class ImageEditDmd2Trainer(ImageEditDmdTrainer):
    """Qwen-Image-Edit DMD2 with a two-time-scale fake critic and latent GAN."""

    trainer_name = "image_edit_dmd2"

    def __init__(self, config):
        super().__init__(config)
        dmd2 = self.training_config.get("dmd2")
        if not isinstance(dmd2, dict):
            raise ValueError("image_edit_dmd2 requires a training.dmd2 mapping.")
        self.dmd2_config = dmd2
        self.fake_updates_per_student = max(1, int(dmd2.get("fake_updates_per_student", self.fake_update_ratio)))
        if self.fake_updates_per_student != self.fake_update_ratio:
            raise ValueError(
                "For image_edit_dmd2, training.dmd.fake_update_ratio and "
                "training.dmd2.fake_updates_per_student must be identical."
            )
        if self.cdm_enabled:
            raise ValueError("image_edit_dmd2 implements DMD2 directly and does not combine it with training.dmd.cdm.")
        if self.ida_trick.enabled:
            raise ValueError("image_edit_dmd2 implements DMD2 directly and does not combine it with training.dmd.ida.")

        gan = dmd2.get("gan")
        if not isinstance(gan, dict) or not bool(gan.get("enabled", False)):
            raise ValueError("image_edit_dmd2 requires training.dmd2.gan.enabled=true for the DMD2 adversarial term.")
        self.gan_config = gan
        self.gan_weight = float(gan.get("weight", 0.005))
        if self.gan_weight <= 0:
            raise ValueError("training.dmd2.gan.weight must be positive.")
        self.gan_base_channels = int(gan.get("base_channels", 64))
        self.gan_condition_dim = gan.get("condition_dim")
        self.gan_max_grad_norm = float(gan.get("max_grad_norm", self.max_grad_norm))
        self.gan_optimizer_config = gan.get("optimizer", {})
        if not isinstance(self.gan_optimizer_config, dict):
            raise ValueError("training.dmd2.gan.optimizer must be a mapping.")

    def setup(self, resume_ckpt_path=None):
        # The parent creates student/fake/teacher first.  The discriminator is
        # intentionally replicated and gradient-synchronised explicitly: it is
        # tiny compared with the FSDP-sharded Qwen transformers.
        super().setup(resume_ckpt_path=None)
        self._setup_gan()
        if resume_ckpt_path is not None:
            self._load_resume_state(resume_ckpt_path)
        logger.info(
            "[train] image_edit_dmd2: fake_updates_per_student={} gan_weight={} "
            "gan_base_channels={} gan_condition_dim={}",
            self.fake_updates_per_student,
            self.gan_weight,
            self.gan_base_channels,
            self.gan_condition_dim,
        )

    def _setup_gan(self):
        condition_dim = self.gan_condition_dim
        if condition_dim is None:
            condition_dim = getattr(self.model.transformer.config, "joint_attention_dim", None)
        if condition_dim is None:
            raise ValueError(
                "Cannot infer Qwen prompt embedding width for DMD2 GAN. Set "
                "training.dmd2.gan.condition_dim explicitly."
            )
        self.gan_condition_dim = int(condition_dim)
        self.gan_discriminator = ConditionalLatentPatchDiscriminator(
            latent_channels=int(self.model.latent_channels),
            condition_dim=self.gan_condition_dim,
            base_channels=self.gan_base_channels,
        ).to(device=self.model.device, dtype=torch.float32)
        if is_distributed():
            for tensor in list(self.gan_discriminator.parameters()) + list(self.gan_discriminator.buffers()):
                dist.broadcast(tensor, src=0)
        self.gan_optimizer = self._build_optimizer(self.gan_discriminator.parameters(), self.gan_optimizer_config)
        self.gan_lr_scheduler = self._build_lr_scheduler(
            self.gan_optimizer,
            num_training_steps=self.max_train_iters,
        )

    def _sync_gan_grads(self):
        if not is_distributed():
            return
        world_size = get_world_size()
        for parameter in self.gan_discriminator.parameters():
            if parameter.grad is not None:
                dist.all_reduce(parameter.grad, op=dist.ReduceOp.SUM)
                parameter.grad.div_(world_size)

    def _gan_logits(self, latents, condition):
        return self.gan_discriminator(
            latents,
            condition["prompt_embed"],
            condition.get("prompt_embed_mask"),
        )

    def _gan_discriminator_loss(self, real_latents, generated_latents, condition):
        real_logits = self._gan_logits(real_latents, condition)
        fake_logits = self._gan_logits(generated_latents.detach(), condition)
        # Logistic GAN loss: D(real)->positive, D(fake)->negative.
        return F.softplus(-real_logits).mean() + F.softplus(fake_logits).mean()

    def _gan_generator_loss(self, generated_latents, condition):
        with _frozen_parameters(self.gan_discriminator):
            fake_logits = self._gan_logits(generated_latents, condition)
        return F.softplus(-fake_logits).mean()

    def _forward_student_loss(self, latent_shape, conditions):
        """DMD2's no-regression distribution-matching generator update."""
        condition, negative_condition = conditions
        self._prepare_sampling_schedule(latent_shape)
        end_step_idx = self.sample_end_step()
        x0, _, _ = self.run_back_simulation(
            condition,
            latent_shape,
            end_step_idx,
            grad_enabled=True,
            xt=self.sample_initial_latents(latent_shape),
        )
        sigma = self.scheduler.sample_renoise_sigma(
            latent_shape[0],
            device=self.model.device,
            dtype=self.running_dtype,
        )
        sigma = broadcast_sequence_parallel_value(sigma)
        noise = broadcast_sequence_parallel_value(
            torch.randn(latent_shape, device=self.model.device, dtype=torch.float32)
        )
        renoised_xt = self.scheduler.add_noise(x0.detach(), noise, sigma)
        with torch.no_grad():
            self.fake_model.transformer.eval()
            self.teacher_model.transformer.eval()
            velocity_fake = self._predict_velocity(self.fake_model, renoised_xt, sigma, condition)
            velocity_teacher = self._predict_teacher_velocity(
                renoised_xt,
                sigma,
                condition,
                negative_condition,
            )
        expanded_sigma = self.scheduler._expand_to_ndim(sigma, renoised_xt.ndim)
        x_pred_fake = renoised_xt - expanded_sigma * velocity_fake
        x_pred_teacher = renoised_xt - expanded_sigma * velocity_teacher
        loss_dmd = self._dmd_loss(x0, x_pred_fake, x_pred_teacher)
        return loss_dmd, x0

    def _real_latents(self, sample):
        with torch.no_grad():
            return self.model.encode_to_latent(sample).detach()

    def _gan_state_path(self, checkpoint_dir):
        return os.path.join(checkpoint_dir, "dmd2_gan.pt")

    def _load_resume_state(self, resume_ckpt_path):
        super()._load_resume_state(resume_ckpt_path)
        state_path = self._gan_state_path(resume_ckpt_path)
        if not os.path.isfile(state_path):
            raise RuntimeError(
                f"DMD2 GAN state is missing from checkpoint: {state_path}. "
                "A full image_edit_dmd2 resume requires discriminator and optimizer state."
            )
        state = torch.load(state_path, map_location="cpu", weights_only=False)
        if int(state.get("condition_dim", -1)) != self.gan_condition_dim:
            raise RuntimeError(
                "DMD2 checkpoint GAN condition_dim does not match the active configuration: "
                f"checkpoint={state.get('condition_dim')} config={self.gan_condition_dim}."
            )
        self.gan_discriminator.load_state_dict(state["discriminator"])
        self.gan_optimizer.load_state_dict(state["optimizer"])
        self.gan_lr_scheduler.load_state_dict(state["lr_scheduler"])
        logger.info("[checkpoint][resume] restored DMD2 GAN from {}", state_path)

    def save_checkpoint(self, iteration, save_total_limit):
        super().save_checkpoint(iteration, save_total_limit)
        checkpoint_dir = os.path.join(self.output_train_dir, f"checkpoint-{iteration:09d}")
        if is_main_process():
            torch.save(
                {
                    "condition_dim": self.gan_condition_dim,
                    "discriminator": self.gan_discriminator.state_dict(),
                    "optimizer": self.gan_optimizer.state_dict(),
                    "lr_scheduler": self.gan_lr_scheduler.state_dict(),
                },
                self._gan_state_path(checkpoint_dir),
            )
        barrier()
        logger.info("[checkpoint][save] DMD2 GAN state saved iter={} path={}", iteration, checkpoint_dir)

    def train(self):
        resume_ckpt_path, current_iter = self._resolve_resume()
        self.setup(resume_ckpt_path=resume_ckpt_path)
        if is_main_process():
            os.makedirs(self.output_train_dir, exist_ok=True)
        barrier()

        if self.infer_every_iters:
            self.inferencer.set_data(self.dataloader_eval)
            if current_iter == 0:
                self.run_inference(current_iter)

        logger.info(
            "[train] start method={} iter={}/{} world_size={} grad_accum={} fake_updates_per_student={}",
            self.trainer_name,
            current_iter,
            self.max_train_iters,
            get_world_size(),
            self.gradient_accumulation_iters,
            self.fake_updates_per_student,
        )
        samples = self._iter_train_samples()
        grad_accum_iters = max(1, int(self.gradient_accumulation_iters))

        while current_iter < self.max_train_iters:
            self.optimizer.zero_grad(set_to_none=True)
            self.gan_optimizer.zero_grad(set_to_none=True)
            running_dmd = 0.0
            running_gan_generator = 0.0
            running_gan_discriminator = 0.0

            for micro_idx in range(grad_accum_iters):
                sample = next(samples)
                conditions = self._encode_conditions(sample)
                latent_shape = self._latent_shape(sample)
                self._set_student_gradient_sync(micro_idx == grad_accum_iters - 1)
                dmd_loss, generated_latents = self._forward_student_loss(latent_shape, conditions)
                real_latents = self._real_latents(sample)

                self.gan_discriminator.train()
                discriminator_loss = self._gan_discriminator_loss(real_latents, generated_latents, conditions[0])
                (discriminator_loss / grad_accum_iters).backward()

                generator_gan_loss = self._gan_generator_loss(generated_latents, conditions[0])
                student_loss = dmd_loss + self.gan_weight * generator_gan_loss
                (student_loss / grad_accum_iters).backward()

                running_dmd += dmd_loss.detach().item() / grad_accum_iters
                running_gan_generator += generator_gan_loss.detach().item() / grad_accum_iters
                running_gan_discriminator += discriminator_loss.detach().item() / grad_accum_iters

            self._sync_sequence_parallel_grads(self.trainable_params)
            self._sync_gan_grads()
            torch.nn.utils.clip_grad_norm_(self.trainable_params, self.max_grad_norm)
            torch.nn.utils.clip_grad_norm_(self.gan_discriminator.parameters(), self.gan_max_grad_norm)
            self.optimizer.step()
            self.lr_scheduler.step()
            self.gan_optimizer.step()
            self.gan_lr_scheduler.step()
            self.optimizer.zero_grad(set_to_none=True)
            self.gan_optimizer.zero_grad(set_to_none=True)
            self._after_student_optimizer_step("main")

            running_fake = 0.0
            for fake_update_idx in range(self.fake_updates_per_student):
                self.fake_optimizer.zero_grad(set_to_none=True)
                fake_loss = 0.0
                for micro_idx in range(grad_accum_iters):
                    sample = next(samples)
                    conditions = self._encode_conditions(sample)
                    latent_shape = self._latent_shape(sample)
                    self._set_fake_gradient_sync(micro_idx == grad_accum_iters - 1)
                    result = super().forward_loss(latent_shape, conditions, stage="fake")
                    loss = result["fake"]
                    (loss / grad_accum_iters).backward()
                    fake_loss += loss.detach().item() / grad_accum_iters
                self._sync_sequence_parallel_grads(self.fake_trainable_params)
                torch.nn.utils.clip_grad_norm_(self.fake_trainable_params, self.max_grad_norm)
                self.fake_optimizer.step()
                self.fake_lr_scheduler.step()
                self.fake_optimizer.zero_grad(set_to_none=True)
                running_fake += fake_loss / self.fake_updates_per_student

            current_iter += 1
            if current_iter % self.train_log_every_iters == 0:
                display_dmd = reduce_mean(running_dmd)
                display_fake = reduce_mean(running_fake)
                display_gan_generator = reduce_mean(running_gan_generator)
                display_gan_discriminator = reduce_mean(running_gan_discriminator)
                if is_main_process():
                    logger.info(
                        "[train] iter={}/{} dmd={:.6f} fake={:.6f} gan_g={:.6f} gan_d={:.6f} lr={:.8f}",
                        current_iter,
                        self.max_train_iters,
                        display_dmd,
                        display_fake,
                        display_gan_generator,
                        display_gan_discriminator,
                        self.optimizer.param_groups[0]["lr"],
                    )
                self.log_metrics(
                    {
                        "train/dmd": display_dmd,
                        "train/fake": display_fake,
                        "train/dmd2_gan_generator": display_gan_generator,
                        "train/dmd2_gan_discriminator": display_gan_discriminator,
                        "train/lr": self.optimizer.param_groups[0]["lr"],
                    },
                    step=current_iter,
                )

            if self.save_every_iters and current_iter % self.save_every_iters == 0:
                self.save_checkpoint(current_iter, self.save_total_limit)
            if self.infer_every_iters and current_iter % self.infer_every_iters == 0:
                self.run_inference(current_iter)

        self.finish_monitor()
        logger.info("[train] finished iter={}/{}", current_iter, self.max_train_iters)
