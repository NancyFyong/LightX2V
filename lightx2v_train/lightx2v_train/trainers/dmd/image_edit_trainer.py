"""``image_edit_dmd`` trainer: DMD distillation for Qwen-Image-**Edit** models.

Why this file exists
--------------------
``lightx2v_train.trainers.dmd.DmdTrainer`` builds its conditioning with

    condition = self.model.encode_prompt_condition(prompt)

which only exists on text-to-image models (``qwen_image``, ``flux2_dev``, ...).
``QwenImageEditModel`` instead implements ``encode_condition(sample)``, which
additionally VAE-encodes the reference images and returns
``source_latents`` / ``source_img_shapes``. Running the stock ``dmd`` trainer with
``model.name: qwen_image_edit`` therefore raises ``AttributeError``.

This trainer is a thin subclass that
  * builds the positive condition with ``model.encode_condition(sample)``,
  * builds the CFG negative condition by re-encoding only the *text* with the
    same reference images and reusing the already computed source latents,
so the reference-image conditioning is available to student / fake / teacher.

Install: copy next to the upstream trainers and register it, see
``tools/lightx2v_patches/install.sh``. Then use ``training.method: image_edit_dmd``.
"""

from __future__ import annotations

import torch
from loguru import logger

from lightx2v_train.runtime.sequence_parallel import broadcast_sequence_parallel_value
from lightx2v_train.utils.registry import TRAINER_REGISTER

from .trainer import DmdTrainer


@TRAINER_REGISTER("image_edit_dmd")
class ImageEditDmdTrainer(DmdTrainer):
    trainer_name = "image_edit_dmd"
    # the edit models carry the instruction in the prompt, an empty negative
    # prompt is the convention used by Qwen-Image-Edit inference
    default_negative_prompt = " "

    # ------------------------------------------------------------------
    # DmdTrainer calls ``model.prepare_denoiser_input(latents)`` without the
    # condition, but the edit model needs it (the reference latents are
    # concatenated to the hidden states and define img_shapes).
    # ------------------------------------------------------------------
    def _predict_velocity(self, model, latents, sigma, condition):
        denoiser_input = model.prepare_denoiser_input(latents, condition)
        prediction = model.denoise(denoiser_input, sigma, condition)
        return model.postprocess_denoiser_output(prediction, denoiser_input)

    def _predict_teacher_velocity(self, latents, sigma, condition, negative_condition):
        if negative_condition is None:
            return self._predict_velocity(self.teacher_model, latents, sigma, condition)

        teacher = self.teacher_model
        if teacher.cfg_on_denoiser_output():
            denoiser_input = teacher.prepare_denoiser_input(latents, condition)
            cond_prediction = teacher.denoise(denoiser_input, sigma, condition)
            uncond_prediction = teacher.denoise(denoiser_input, sigma, negative_condition)
            prediction = self._do_cfg(cond_prediction, uncond_prediction, self.guidance_scale, self.cfg_norm)
            return teacher.postprocess_denoiser_output(prediction, denoiser_input)

        velocity_cond = self._predict_velocity(teacher, latents, sigma, condition)
        velocity_uncond = self._predict_velocity(teacher, latents, sigma, negative_condition)
        return self._do_cfg(velocity_cond, velocity_uncond, self.guidance_scale, self.cfg_norm)

    def _encode_conditions(self, sample):
        with torch.no_grad():
            condition = self.model.encode_condition(sample)
            negative_condition = None
            if self.guidance_scale > 1:
                prompt = sample["conditioning"].get("prompt", "")
                is_scalar_prompt = isinstance(prompt, str)
                batch_size = 1 if is_scalar_prompt else len(prompt)
                negative_prompt = self._negative_prompt_for_conditioning(
                    sample["conditioning"],
                    batch_size,
                    return_scalar=is_scalar_prompt,
                )
                negative_condition = self._encode_negative_condition(sample, negative_prompt, condition)

        condition = broadcast_sequence_parallel_value(condition)
        if negative_condition is not None:
            negative_condition = broadcast_sequence_parallel_value(negative_condition)
        return condition, negative_condition

    def _encode_negative_condition(self, sample, negative_prompt, positive_condition):
        """Text-only re-encode: reuse the reference latents of the positive pass."""
        model = self.model
        source_images = model._source_images_from_sample(sample)
        condition_images = model._condition_images_from_source_tensors(source_images)
        prompt_embed, prompt_embed_mask = model.text_pipeline.encode_prompt(
            prompt=negative_prompt,
            image=condition_images,
            device=model.device,
            num_images_per_prompt=1,
            max_sequence_length=model.config["model"].get("max_sequence_length", 1024),
        )
        negative_condition = {
            "prompt_embed": prompt_embed,
            "prompt_embed_mask": prompt_embed_mask,
        }
        for key in ("source_latents", "source_img_shapes"):
            if key in positive_condition:
                negative_condition[key] = positive_condition[key]
        return negative_condition

    def _log_extra_setup(self):
        logger.info(
            "[train] image_edit_dmd: reference-image conditioning enabled "
            "(source_latents + img_shapes shared with the CFG negative pass)"
        )
