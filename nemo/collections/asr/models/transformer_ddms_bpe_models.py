import einops
from typing import Any, Dict, List, Optional, Union

import torch
from lightning.pytorch import Trainer
from omegaconf import DictConfig, OmegaConf, open_dict
from torch.utils.data import DataLoader

from nemo.collections.asr.data.audio_to_text_dali import DALIOutputs
from nemo.collections.asr.models import EncDecTransfModelBPE
from nemo.collections.asr.parts.submodules.discrete_diffusion_scheduler import get_noise_scheduler

class EncDecTransfDDMSModelBPE(EncDecTransfModelBPE):
    def __init__(self, cfg: DictConfig, trainer: Trainer = None):
        super().__init__(cfg=cfg, trainer=trainer)
        self.noise_schedule = get_noise_scheduler(cfg.transf_decoder)
        self.time_min = 0.0
        self.time_max = 1.0
        self.num_steps = 10

    def _prepare_decoder_input_and_target(self, input_ids, target_ids, mask_prob=None):

        if mask_prob is None:
            mask_prob = self.noise_schedule.sample_time(batch_size=input_ids.size(0), device=input_ids.device)
        mask_prob = einops.repeat(mask_prob, 'b -> b t', t = input_ids.size(1))
        will_mask = torch.bernoulli(mask_prob).to(dtype=torch.bool).to(input_ids.device)
        masked_input_ids = torch.where(will_mask, 0, input_ids)
        masked_target_ids = torch.where(~will_mask, self.tokenizer.pad_id, target_ids)

        return masked_input_ids, masked_target_ids
    
    def compute_audio_loss(self, batch):

        if batch is None:
            return 0

        signal, signal_len, transcript, transcript_len = batch
        input_ids, labels = transcript[:, 1:], transcript[:, 1:] # Remove the bos token for input_ids as well

        # Modify the input_ids by masking out some of the tokens
        input_ids, labels = self._prepare_decoder_input_and_target(input_ids, labels)

        transf_log_probs, encoded_len, enc_states, enc_mask = self.forward(
            input_signal=signal,
            input_signal_length=signal_len,
            transcript=input_ids,
            transcript_length=transcript_len,
        )

        transf_loss = self.transf_loss(log_probs=transf_log_probs, labels=labels)

        return transf_loss

    def _single_reverse_step(self, x_t, t, enc_states, enc_mask):
        

    def validation_step(self, batch, batch_idx, dataloader_idx=0, eval_mode="val"):
        signal, signal_len, transcript, transcript_len = batch
        input_ids, labels = transcript[:, 1:], transcript[:, 1:]
        
        # Modify the input_ids by masking out some of the tokens
        # For validation efficiency, we only test on all-mask prediction
        mask_prob = torch.tensor([1.0])
        input_ids, labels = self._prepare_decoder_input_and_target(input_ids, labels, mask_prob=mask_prob)

        if isinstance(batch, DALIOutputs) and batch.has_processed_signal:
            transf_log_probs, encoded_len, enc_states, enc_mask = self.forward(
                processed_signal=signal,
                processed_signal_length=signal_len,
                transcript=input_ids,
                transcript_length=transcript_len,
            )
        else:
            transf_log_probs, encoded_len, enc_states, enc_mask = self.forward(
                input_signal=signal,
                input_signal_length=signal_len,
                transcript=input_ids,
                transcript_length=transcript_len,
            )
        prediction_logprobs, prediction_labels = transf_log_probs.max(dim=-1)
        transf_loss = self.transf_loss(log_probs=transf_log_probs, labels=labels)

        ground_truths = [self.tokenizer.ids_to_text(sent) for sent in transcript.detach().cpu().tolist()]
        translations = [self.tokenizer.ids_to_text(sent) for sent in prediction_labels.detach().cpu().tolist()]

        self.val_loss(loss=transf_loss, num_measurements=transf_log_probs.shape[0] * transf_log_probs.shape[1])

        output_dict = {f'{eval_mode}_loss': transf_loss, 'translations': translations, 'ground_truths': ground_truths}

        self.validation_step_outputs.append(output_dict)

        return output_dict

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        signal, signal_len, transcript, transcript_len = batch
        input_ids, labels = transcript[:, 1:], transcript[:, 1:]
        
        time_steps = torch.linspace(self.time_min, self.time_max, self.num_steps + 1)[1:]
        # Modify the input_ids by masking out some of the tokens
        for t in reversed(time_steps): 
            mask_prob = torch.tensor([t])
            input_ids, labels = self._prepare_decoder_input_and_target(input_ids, labels, mask_prob=mask_prob)

            if isinstance(batch, DALIOutputs) and batch.has_processed_signal:
                transf_log_probs, encoded_len, enc_states, enc_mask = self.forward(
                    processed_signal=signal,
                    processed_signal_length=signal_len,
                    transcript=input_ids,
                    transcript_length=transcript_len,
                )
            else:
                transf_log_probs, encoded_len, enc_states, enc_mask = self.forward(
                    input_signal=signal,
                    input_signal_length=signal_len,
                    transcript=input_ids,
                    transcript_length=transcript_len,
                )
            prediction_logprobs, prediction_labels = transf_log_probs.max(dim=-1)
            input_ids = prediction_labels
        

        return prediction_labels