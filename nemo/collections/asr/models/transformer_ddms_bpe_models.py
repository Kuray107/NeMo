import os
import einops
from tqdm import tqdm
from typing import Any, Dict, List, Optional, Union

import torch
from lightning.pytorch import Trainer
from omegaconf import DictConfig, OmegaConf, open_dict
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F

from nemo.collections.asr.data.audio_to_text_dali import DALIOutputs
from nemo.collections.asr.losses.ctc import CTCLoss
from nemo.collections.asr.models import EncDecTransfModelBPE
from nemo.collections.asr.parts.submodules.discrete_diffusion_scheduler import get_noise_scheduler

class EncDecTransfDDMSModelBPE(EncDecTransfModelBPE):
    def __init__(self, cfg: DictConfig, trainer: Trainer = None):
        super().__init__(cfg=cfg, trainer=trainer)
        
        self.noise_schedule = get_noise_scheduler(cfg.transf_decoder)
        self.ins_mask_ratio = None
        if 'ctc' in cfg.transf_decoder.dec_type:
            self.ctc_loss = CTCLoss(
                num_classes=1, # setting <nospeech> as blank for now
                zero_infinity=True,
                reduction=cfg.get("ctc_reduction", "mean_batch"),
            )
            self.ins_mask_ratio = 0.5

    def _prepare_decoder_input_and_target(self, input_ids, target_ids, mask_prob=None, ins_mask_ratio=None):

        if mask_prob is None:
            mask_prob = self.noise_schedule.sample_time(batch_size=input_ids.size(0), device=input_ids.device)
        if ins_mask_ratio:
            ins_mask_prob = mask_prob * ins_mask_ratio
            return self._prepare_decoder_input_and_target_ctc(input_ids, target_ids, mask_prob, ins_mask_prob)

        mask_prob = einops.repeat(mask_prob, 'b -> b t', t = input_ids.size(1))
        will_mask = torch.bernoulli(mask_prob).to(dtype=torch.bool).to(input_ids.device)
        masked_input_ids = torch.where(will_mask, 0, input_ids)
        masked_target_ids = torch.where(~will_mask, self.tokenizer.pad_id, target_ids)

        return masked_input_ids, masked_target_ids, will_mask

    def _prepare_decoder_input_and_target_ctc(self, input_ids, target_ids, mask_prob, ins_mask_prob):
        input_lengths = input_ids.ne(2).sum(-1)
        masked_input_ids = []
        masked_target_ids = []
        for input_id, target_id, input_length, mask_p, ins_mask_p in zip(input_ids, target_ids, input_lengths, mask_prob, ins_mask_prob):
            num_mask = round(input_length.item() * mask_p.item())
            num_ins_mask = round(input_length.item() * ins_mask_p.item())
            # randomly select the indices between 0 and input_length - 1
            mask_indices = torch.randperm(input_length.item(), device=input_ids.device)[:num_mask]
            input_id[mask_indices] = 0

            # create a new input_id and target_id with length num_ins_mask + input_ids.size(1) with all zeros
            masked_input_id = torch.zeros(input_length + num_ins_mask, dtype=input_ids.dtype, device=input_ids.device)
            masked_target_id = torch.ones(input_length + num_ins_mask, dtype=input_ids.dtype, device=input_ids.device)
            # randomly sample the indices between 0 and new_input_length - 1 for insertion
            ins_mask_indices = torch.randperm(masked_input_id.size(0), device=input_ids.device)[:num_ins_mask]
            # exclude the indices in ins_mask_indices and add the input_ids to the rest of new_input_ids
            will_ins = torch.isin(torch.arange(masked_input_id.size(0), device=input_ids.device), ins_mask_indices)
            masked_input_id[~will_ins] = input_id[:input_length]
            masked_target_id[~will_ins] = target_id[:input_length]
            
            masked_input_ids.append(masked_input_id)
            masked_target_ids.append(masked_target_id)
        
        masked_input_ids = pad_sequence(masked_input_ids, batch_first=True, padding_value=2)
        masked_target_ids = pad_sequence(masked_target_ids, batch_first=True, padding_value=2)
        
        return masked_input_ids, masked_target_ids, masked_input_ids == 0

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
        input_ids, labels = transcript[:, 1:].clone(), transcript[:, 1:] # Remove the bos token for input_ids as well

        # Modify the input_ids by masking out some of the tokens
        input_ids, labels, will_mask = self._prepare_decoder_input_and_target(
            input_ids, labels,ins_mask_ratio=self.ins_mask_ratio
        )

        transf_log_probs, encoded_len, enc_states, enc_mask = self.forward(
            input_signal=signal,
            input_signal_length=signal_len,
            transcript=input_ids,
            transcript_length=transcript_len,
        )

        if 'ctc' in self.cfg.transf_decoder.dec_type:
            # Create 1-hot vectors for input_ids
            input_ids_1hot = F.one_hot(input_ids, num_classes=self.tokenizer.vocab_size).to(torch.float)
            input_ids_1hot = torch.log(input_ids_1hot + 1e-8)
            # Replace the masked positions with the predicted log_probs
            transf_log_probs = transf_log_probs * will_mask.unsqueeze(-1) + input_ids_1hot + (~will_mask.unsqueeze(-1))
            
            # Labels are the original transcript
            labels = transcript[:, 1:]
            transf_loss = self.ctc_loss(
                log_probs=transf_log_probs, 
                targets=labels, 
                input_lengths=input_ids.ne(2).sum(-1), 
                target_lengths=labels.ne(2).sum(-1)
            )
        else:
            transf_loss = self.transf_loss(log_probs=transf_log_probs, labels=labels)
        return transf_loss


    def validation_step(self, batch, batch_idx, dataloader_idx=0, eval_mode="val"):
        signal, signal_len, transcript, transcript_len = batch
        input_ids, labels = transcript[:, 1:], transcript[:, 1:]
        
        # Modify the input_ids by masking out some of the tokens
        # For validation efficiency, we only test on all-mask prediction
        mask_prob = torch.tensor([1.0])
        input_ids, labels, _ = self._prepare_decoder_input_and_target(
            input_ids, labels, mask_prob=mask_prob, ins_mask_ratio=self.ins_mask_ratio
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
        transf_loss = self.transf_loss(log_probs=transf_log_probs, labels=labels)

        ground_truths = [self.tokenizer.ids_to_text(sent) for sent in transcript.detach().cpu().tolist()]
        translations = [
            self.tokenizer.ids_to_text(sent).replace('<|nospeech|>', '') for sent in prediction_labels.detach().cpu().tolist()
        ]

        self.val_loss(loss=transf_loss, num_measurements=transf_log_probs.shape[0] * transf_log_probs.shape[1])

        output_dict = {f'{eval_mode}_loss': transf_loss, 'translations': translations, 'ground_truths': ground_truths}

        self.validation_step_outputs.append(output_dict)

        return output_dict

    def transcribe(self, test_manifest, batch_size=1, num_steps=1, sampler='topk'):
        self.num_steps = num_steps,
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
            predictions = self.test_step(batch, i)
            translations.append(predictions)
        return translations

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        signal, signal_len, transcript, transcript_len = batch
        signal = signal.to(self.device)
        signal_len = signal_len.to(self.device)
        transcript = transcript.to(self.device)
        transcript_len = transcript_len.to(self.device)
        input_ids, labels = transcript[:, 1:].to(self.device), transcript[:, 1:].to(self.device)
        
        time_steps = torch.linspace(0.0, 1.0, self.num_steps + 1)[1:]
        prev_prediction_labels = torch.ones_like(input_ids)
        prediction_logprobs = None
        for mask_prob in reversed(time_steps): 
            # input_ids, labels, will_mask = self._prepare_decoder_input_and_target(input_ids, labels, mask_prob=mask_prob)
            input_ids, will_mask = self._sampler(input_ids, prediction_logprobs, mask_prob=mask_prob)

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
            input_ids = prediction_labels
            prev_prediction_labels = prediction_labels
            
        
        prediction_labels = self.tokenizer.ids_to_text(prediction_labels.detach().cpu().tolist()[0])
        return prediction_labels