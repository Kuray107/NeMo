import os
import einops
from tqdm import tqdm
from typing import Any, Dict, List, Optional, Union

import torch
import torch.distributed as dist
from lightning.pytorch import Trainer
from omegaconf import DictConfig, OmegaConf, open_dict
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F

from nemo.collections.asr.data.audio_to_text_dali import DALIOutputs
from nemo.collections.asr.models import EncDecTransfModelBPE
from nemo.collections.asr.parts.submodules.discrete_diffusion_scheduler import get_noise_scheduler

class EncDecTransfDDMSModelBPE(EncDecTransfModelBPE):
    def __init__(self, cfg: DictConfig, trainer: Trainer = None):
        super().__init__(cfg=cfg, trainer=trainer)
        
        self.noise_schedule = get_noise_scheduler(cfg.transf_decoder)
        self.unfolded_training = True

    def _prepare_decoder_input_and_target(self, input_ids, target_ids, mask_prob=None):
        if mask_prob is None:
            mask_prob = self.noise_schedule.sample_time(batch_size=input_ids.size(0), device=input_ids.device)
        
        mask_prob = einops.repeat(mask_prob, 'b -> b t', t = input_ids.size(1))
        will_mask = torch.bernoulli(mask_prob).to(dtype=torch.bool).to(input_ids.device)
        masked_input_ids = torch.where(will_mask, 0, input_ids)
        masked_target_ids = torch.where(~will_mask, self.tokenizer.pad_id, target_ids)

        return masked_input_ids, masked_target_ids, will_mask

    def _sampler(self, input_ids, log_probs, mask_prob):
        if self.sampler == 'topk':
            return self._topk_sampler(input_ids, log_probs, mask_prob)
        else:
            return self._random_sampler(input_ids, mask_prob)

    def _random_sampler(self, input_ids, mask_prob):
        mask_prob = torch.tensor([mask_prob])
        mask_prob = einops.repeat(mask_prob, 'b -> b t', t = input_ids.size(1))
        will_mask = torch.bernoulli(mask_prob).to(dtype=torch.bool).to(input_ids.device)
        masked_input_ids = torch.where(will_mask, 0, input_ids)
        
        return masked_input_ids, will_mask

    def _topk_sampler(self, input_ids, log_probs, mask_prob):
        if mask_prob == 1.0:
            return torch.zeros_like(input_ids), torch.ones_like(input_ids, dtype=torch.bool)
        
        topk = round(input_ids.size(1) * (1 - mask_prob.item())) # number of tokens to keep
        topk_log_probs, topk_indices = torch.topk(log_probs, k=topk, dim=-1) # find topk indices to keep
        will_mask = torch.ones_like(input_ids, dtype=torch.bool)
        will_mask = will_mask.scatter(1, topk_indices, False)
        masked_input_ids = torch.where(will_mask, 0, input_ids)
        
        return masked_input_ids, will_mask

    def compute_audio_loss(self, batch):

        if batch is None:
            return 0

        signal, signal_len, transcript, transcript_len = batch
        input_ids, labels = transcript[:, 1:], transcript[:, 1:] # Remove the bos token for input_ids as well

        # Modify the input_ids by masking out some of the tokens
        input_ids, masked_labels, will_mask = self._prepare_decoder_input_and_target(input_ids, labels)

        transf_log_probs, encoded_len, enc_states, enc_mask = self.forward(
            input_signal=signal,
            input_signal_length=signal_len,
            transcript=input_ids,
            transcript_length=transcript_len,
        )

        transf_loss = self.transf_loss(log_probs=transf_log_probs, labels=masked_labels)

        if self.unfolded_training:
            # perform unfolded training by resampling
            prediction_logprobs, prediction_labels = transf_log_probs.max(dim=-1)
            del transf_log_probs, encoded_len, enc_states, enc_mask
            del masked_labels
            # carry-over unmasking
            prediction_labels = torch.where(~will_mask, input_ids, prediction_labels)
            del input_ids
            # resample again with the new prediction_labels and calculate the loss again
            input_ids, masked_labels, will_mask = self._prepare_decoder_input_and_target(prediction_labels, labels)
            transf_log_probs, encoded_len, enc_states, enc_mask = self.forward(
                input_signal=signal,
                input_signal_length=signal_len,
                transcript=input_ids,
                transcript_length=transcript_len,
            )
            transf_loss += self.transf_loss(log_probs=transf_log_probs, labels=masked_labels)
        
        del transf_log_probs, encoded_len, enc_states, enc_mask

        return transf_loss


    def validation_step(self, batch, batch_idx, dataloader_idx=0, eval_mode="val"):
        signal, signal_len, transcript, transcript_len = batch
        input_ids, labels = transcript[:, 1:].clone(), transcript[:, 1:]
        
        # For validation efficiency, we only test 2-step denoising
        for mask_prob in [1.0, 0.5]:
            input_ids, labels, will_mask = self._prepare_decoder_input_and_target(
                input_ids, labels, mask_prob=torch.tensor([mask_prob]).repeat(input_ids.size(0))
            )

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
            
            # carry-over unmasking
            if mask_prob < 1.0:
                prediction_labels = torch.where(~will_mask, prev_prediction_labels, prediction_labels)
            
            prev_prediction_labels = prediction_labels
            input_ids = prediction_labels

        ground_truths = [
            self.tokenizer.ids_to_text(sent[:tlen-1]) for sent, tlen in zip(transcript[:, 1:].detach().cpu().tolist(), transcript_len.detach().cpu().tolist())
        ]
    
        translations = [
            self.tokenizer.ids_to_text(sent[:tlen-1]) for sent, tlen in zip(prediction_labels.detach().cpu().tolist(), transcript_len.detach().cpu().tolist())
        ]

        loss = self.transf_loss(log_probs=transf_log_probs, labels=labels)
        self.val_loss(loss=loss, num_measurements=transf_log_probs.shape[0] * transf_log_probs.shape[1])

        output_dict = {f'{eval_mode}_loss': loss, 'translations': translations, 'ground_truths': ground_truths}
        self.validation_step_outputs.append(output_dict)

        return output_dict

    def transcribe(self, test_manifest, batch_size=1, num_steps=1, sampler='topk'):
        self.num_steps = num_steps
        self.sampler = sampler

        dl_config = {
            'manifest_filepath': test_manifest,
            'sample_rate': self.preprocessor._sample_rate,
            'batch_size': batch_size,
            'trim_silence': False,
            'shuffle': False,
            'num_workers': min(batch_size, os.cpu_count() - 1),
            'pin_memory': True,
        }

        temporary_datalayer = self._setup_dataloader_from_config(config=DictConfig(dl_config))

        translations = []
        for i, batch in enumerate(tqdm(temporary_datalayer, desc="Transcribing")):
            predictions = self.test_step(batch, i, num_steps)
            translations.append(predictions)
        return translations

    def test_step(self, batch, batch_idx, dataloader_idx=0, carry_over_unmask=True):
        signal, signal_len, transcript, transcript_len = batch
        signal = signal.to(self.device)
        signal_len = signal_len.to(self.device)
        transcript = transcript.to(self.device)
        transcript_len = transcript_len.to(self.device)
        input_ids, labels = torch.zeros((1, 256), dtype=torch.long, device=self.device), transcript[:, 1:].to(self.device)
        
        time_steps = torch.linspace(0.0, 1.0, self.num_steps + 1)[1:]
        prev_prediction_labels = torch.ones_like(input_ids)
        prev_prediction_logprobs = torch.zeros_like(input_ids)
        prediction_logprobs = None
        for mask_prob in reversed(time_steps):
            input_ids, will_mask = self._sampler(
                input_ids, prediction_logprobs, mask_prob=mask_prob
            )

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

            # carry-over unmasking
            if mask_prob < 1.0 and carry_over_unmask:
                prediction_labels = torch.where(~will_mask, prev_prediction_labels, prediction_labels)
            
            # find all <eos> token and remove them
            eos_token_id = 3
            eos_idx  = (prediction_labels == eos_token_id).nonzero(as_tuple=True)[1]
            if len(eos_idx) == 0:
                eos_idx = prediction_labels.size(1)
            else:
                eos_idx = eos_idx[0].item()

            prediction_labels = prediction_labels[:, :eos_idx]
            prediction_logprobs = prediction_logprobs[:, :eos_idx]
            input_ids = prediction_labels
            prev_prediction_labels = prediction_labels
            prev_prediction_logprobs = prediction_logprobs
        
        prediction_labels = self.tokenizer.ids_to_text(prediction_labels.detach().cpu().tolist()[0])
        return prediction_labels