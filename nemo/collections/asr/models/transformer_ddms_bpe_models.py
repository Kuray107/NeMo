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
from nemo.collections.common.losses import WeightedSmoothedCrossEntropyLoss, SmoothedCrossEntropyLoss
from nemo.collections.asr.parts.submodules.discrete_diffusion_scheduler import get_noise_scheduler


def lens_to_mask(lens, max_length):
    batch_size = lens.shape[0]
    mask = torch.arange(max_length).repeat(batch_size, 1).to(lens.device) < lens[:, None]
    return mask

def do_nucleus_sampling(p_x0, p_nucleus=0.9):
    p_x0 = p_x0.exp()
    print (p_x0)
    sorted_probs, sorted_indices = torch.sort(p_x0, descending=True, dim=-1)
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
    top_p_mask = cumulative_probs <= p_nucleus
    top_p_mask[..., 0] = True #always authorize at least the maximum-prob token 
    nucleus_probs = sorted_probs * top_p_mask
    nucleus_probs /= nucleus_probs.sum(dim=-1, keepdim=True)
    p_x0 = torch.zeros_like(p_x0).scatter_(-1, sorted_indices, nucleus_probs)
    breakpoint()
    return p_x0


class EncDecTransfDDMSModelBPE(EncDecTransfModelBPE):
    def __init__(self, cfg: DictConfig, trainer: Trainer = None):
        super().__init__(cfg=cfg, trainer=trainer)
        
        self.noise_schedule = get_noise_scheduler(cfg.transf_decoder)
        self.unfolded_training = True
        self.inference_length = 256
        
        # Define weighted CE loss
        self.transf_loss = WeightedSmoothedCrossEntropyLoss(
            pad_id=self.tokenizer.pad_id, label_smoothing=self.cfg.label_smoothing
        )

    def _prepare_decoder_input_and_target(self, input_ids, target_ids, mask_prob=None):
        if mask_prob is None:
            mask_prob = self.noise_schedule.sample_time(batch_size=input_ids.size(0), device=input_ids.device)
        weights = 1 / (mask_prob)
        
        mask_prob = einops.repeat(mask_prob, 'b -> b t', t = input_ids.size(1))
        will_mask = torch.bernoulli(mask_prob).to(dtype=torch.bool).to(input_ids.device)
        masked_input_ids = torch.where(will_mask, 0, input_ids)
        masked_target_ids = torch.where(~will_mask, self.tokenizer.pad_id, target_ids)

        return masked_input_ids, masked_target_ids, will_mask, weights

    def _sampler(self, input_ids, log_probs, mask_prob):
        if self.sampler == 'topk':
            return self._topk_sampler(input_ids, log_probs, mask_prob)
        elif self.sampler == 'topk-r':
            return self._topk_r_sampler(input_ids, log_probs, mask_prob)
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

    def _topk_r_sampler(self, input_ids, log_probs, mask_prob):
        pass
    
    def compute_audio_loss(self, batch):

        if batch is None:
            return 0
        
        # Decide if it is ConcatDataset
        if hasattr(self._train_dl.dataset, 'collate_fn'):
            signal, signal_len, transcript, transcript_len = batch
            num_cfg_samples = round(signal.size(0) * self.cfg.cfg_ratio)
        else:
            signal, signal_len, transcript, transcript_len, sample_id = batch         
            perm = torch.cat([
                torch.nonzero(sample_id == 0, as_tuple=False).squeeze(-1),
                torch.nonzero(sample_id == 1, as_tuple=False).squeeze(-1),
            ], dim=0)
            signal        = signal.index_select(0, perm)
            signal_len    = signal_len.index_select(0, perm)
            transcript    = transcript.index_select(0, perm)
            transcript_len= transcript_len.index_select(0, perm)
            num_cfg_samples = torch.sum(sample_id == 1).item()
            signal[signal.size(0)-num_cfg_samples:] = 0.0


        input_ids, labels = transcript[:, 1:], transcript[:, 1:] # Remove the bos token for input_ids as well

        # Modify the input_ids by masking out some of the tokens
        input_ids, masked_labels, will_mask, weights = self._prepare_decoder_input_and_target(input_ids, labels)

        transf_log_probs, encoded_len, enc_states, enc_mask = self.forward(
            input_signal=signal,
            input_signal_length=signal_len,
            transcript=input_ids,
            transcript_length=transcript_len,
            num_cfg_samples=num_cfg_samples
        )

        transf_loss = self.transf_loss(log_probs=transf_log_probs, labels=masked_labels, sample_weights=weights)

        if self.unfolded_training:
            # perform unfolded training by resampling
            prediction_logprobs, prediction_labels = transf_log_probs.max(dim=-1)
            del transf_log_probs, encoded_len, enc_states, enc_mask
            del masked_labels
            # carry-over unmasking
            prediction_labels = torch.where(~will_mask, input_ids, prediction_labels)
            del input_ids
            # resample again with the new prediction_labels and calculate the loss again
            input_ids, masked_labels, will_mask, weights = self._prepare_decoder_input_and_target(prediction_labels, labels)
            transf_log_probs, encoded_len, enc_states, enc_mask = self.forward(
                input_signal=signal,
                input_signal_length=signal_len,
                transcript=input_ids,
                transcript_length=transcript_len,
                num_cfg_samples=num_cfg_samples
            )
            transf_loss += self.transf_loss(log_probs=transf_log_probs, labels=masked_labels, sample_weights=weights)
        
        del transf_log_probs, encoded_len, enc_states, enc_mask

        return transf_loss
    
    def forward(
        self,
        input_signal=None,
        input_signal_length=None,
        processed_signal=None,
        processed_signal_length=None,
        transcript=None,
        transcript_length=None,
        num_cfg_samples=0
    ):
        """
        Forward pass of the model.
        Args:
            input_signal: Tensor that represents a batch of raw audio signals,
                of shape [B, T]. T here represents timesteps, with 1 second of audio represented as
                `self.sample_rate` number of floating point values.
            input_signal_length: Vector of length B, that contains the individual lengths of the audio
                sequences.
            processed_signal: Tensor that represents a batch of processed audio signals,
                of shape (B, D, T) that has undergone processing via some DALI preprocessor.
            processed_signal_length: Vector of length B, that contains the individual lengths of the
                processed audio sequences.
            num_cfg_samples: number of samples in batch to perform classifier-free guidance
        Returns:
            A tuple of 3 elements -
            1) The log probabilities tensor of shape [B, T, D].
            2) The lengths of the acoustic sequence after propagation through the encoder, of shape [B].
            3) The greedy token predictions of the model of shape [B, T] (via argmax)
        """
        has_input_signal = input_signal is not None and input_signal_length is not None
        has_processed_signal = processed_signal is not None and processed_signal_length is not None
        if (has_input_signal ^ has_processed_signal) == False:
            raise ValueError(
                f"{self} Arguments ``input_signal`` and ``input_signal_length`` are mutually exclusive "
                " with ``processed_signal`` and ``processed_signal_len`` arguments."
            )

        if not has_processed_signal:
            processed_signal, processed_signal_length = self.preprocessor(
                input_signal=input_signal, length=input_signal_length
            )

        if self.spec_augmentation is not None and self.training:
            processed_signal = self.spec_augmentation(input_spec=processed_signal, length=processed_signal_length)

        encoded, encoded_len = self.encoder(audio_signal=processed_signal, length=processed_signal_length)

        enc_states = encoded.permute(0, 2, 1)
        enc_states = self.adapter(enc_states)
        enc_mask = lens_to_mask(encoded_len, enc_states.shape[1]).to(enc_states.dtype)
        if self.use_transf_encoder:
            enc_states = self.transf_encoder(encoder_states=enc_states, encoder_mask=enc_mask)

        transf_log_probs = None
        if transcript is not None:
            dec_mask = lens_to_mask(transcript_length, transcript.shape[1]).to(transcript.dtype)
            dec_states = self.transf_decoder(
                input_ids=transcript, decoder_mask=dec_mask, encoder_embeddings=enc_states, encoder_mask=enc_mask, num_cfg_samples=num_cfg_samples
            )
            transf_log_probs = self.log_softmax(hidden_states=dec_states)

        return transf_log_probs, encoded_len, enc_states, enc_mask


    def validation_step(self, batch, batch_idx, dataloader_idx=0, eval_mode="val"):
        signal, signal_len, transcript, transcript_len = batch
        input_ids, labels = transcript[:, 1:].clone(), transcript[:, 1:]
        
        # For validation efficiency, we only test 2-step denoising
        for mask_prob in [1.0, 0.5]:
            input_ids, labels, will_mask, weights = self._prepare_decoder_input_and_target(
                input_ids, labels, mask_prob=torch.tensor([mask_prob]).repeat(input_ids.size(0))
            )

            transf_log_probs, encoded_len, enc_states, enc_mask = self.forward(
                input_signal=signal,
                input_signal_length=signal_len,
                transcript=input_ids,
                transcript_length=transcript_len,
            )
            # if mask_prob < 1.0:
            #     cfg_transf_log_probs, encoded_len, enc_states, enc_mask = self.forward(
            #         input_signal=signal,
            #         input_signal_length=signal_len,
            #         transcript=input_ids,
            #         transcript_length=transcript_len,
            #         num_cfg_samples=signal.size(0)
            #     )
            #     transf_log_probs = 0.9 * transf_log_probs + 0.1 * cfg_transf_log_probs # simple CFG averaging
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

    def transcribe(self, test_manifest, batch_size=1, num_steps=1, sampler='topk', cfg_weight=None):
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
        total_len = []
        self.short = 0
        self.long = 0
        for i, batch in enumerate(tqdm(temporary_datalayer, desc="Transcribing")):
            total_len.append(batch[3][0])
            predictions = self.test_step(batch, i, num_steps, cfg_weight=cfg_weight)
            translations.extend(predictions)
        print ("Average transcription length: ", sum(total_len) / (i+1))
        print ("Max transcription length: ", max(total_len))
        print ("Short results: ", self.short)
        print ("Long results: ", self.long)

        return translations

    def test_step(self, batch, batch_idx, dataloader_idx=0, carry_over_unmask=True, cfg_weight=None):
        signal, signal_len, transcript, transcript_len = batch
        signal = signal.to(self.device)
        signal_length = signal_len.to(self.device)

        batch_size = signal.size(0)
        input_ids = torch.zeros((batch_size, self.inference_length), dtype=torch.long, device=self.device)
        input_ids_length = torch.ones((batch_size,), dtype=torch.long, device=self.device) * self.inference_length
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
                    processed_signal_length=signal_length,
                    transcript=input_ids,
                    transcript_length=input_ids_length,
                )
            else:
                transf_log_probs, encoded_len, enc_states, enc_mask = self.forward(
                    input_signal=signal,
                    input_signal_length=signal_length,
                    transcript=input_ids,
                    transcript_length=input_ids_length,
                )

            # Classifier-free guidance re-weighting
            if mask_prob < 0.5 and cfg_weight:
                cfg_transf_log_probs, encoded_len, enc_states, enc_mask = self.forward(
                    input_signal=signal,
                    input_signal_length=signal_length,
                    transcript=input_ids,
                    transcript_length=input_ids_length,
                    num_cfg_samples=signal.size(0)
                )
                transf_log_probs = (1 - cfg_weight) * transf_log_probs + 0 * cfg_transf_log_probs

            # Zero Masking Probability
            transf_log_probs[:, :, self.tokenizer.pad_id] = float('-inf')
            prediction_logprobs, prediction_labels = transf_log_probs.max(dim=-1)
            # prediction_logprobs, prediction_labels = sample_categorical(transf_log_probs, temperature=1.0)

            # carry-over unmasking
            if mask_prob < 1.0 and carry_over_unmask:
                prediction_labels = torch.where(~will_mask, prev_prediction_labels, prediction_labels)
                prediction_logprobs = torch.where(~will_mask, prev_prediction_logprobs, prediction_logprobs)
            
            # find all <eos> token and update lengths for each sample in the batch
            eos_token_id = self.tokenizer.eos_id
            batch_size = prediction_labels.size(0)
            seq_len = prediction_labels.size(1)
            
            # Find EOS position for each sample in the batch
            # For each sample, find the first EOS token, or use seq_len if no EOS found
            eos_positions = torch.full((batch_size,), seq_len, dtype=torch.long, device=prediction_labels.device)
            for batch_idx in range(batch_size):
                eos_indices = (prediction_labels[batch_idx] == eos_token_id).nonzero(as_tuple=True)[0]
                if len(eos_indices) > 0:
                    eos_positions[batch_idx] = eos_indices[0].item() + 1  # +1 to include EOS token
            
            # Update input_ids_length to reflect actual sequence length (up to EOS) for each sample
            input_ids_length = eos_positions
            
            # Find the maximum length needed to keep all samples aligned
            max_len = eos_positions.max().item()
            
            # Truncate all samples to max_len (but keep track of individual lengths via input_ids_length)
            prediction_labels = prediction_labels[:, :max_len]
            prediction_logprobs = prediction_logprobs[:, :max_len]
            input_ids = prediction_labels
            prev_prediction_labels = prediction_labels
            prev_prediction_logprobs = prediction_logprobs
        
        # Process all samples in the batch
        batch_size = prediction_labels.size(0)
        translations = []
        
        for batch_idx in range(batch_size):
            sample_length = input_ids_length[batch_idx].item()
            sample_labels = prediction_labels[batch_idx, :sample_length]
            if transcript is not None:
                actual_transcript_len = (transcript_len[batch_idx].item() - 1) # Remove the bos token
                if sample_length < actual_transcript_len:
                    self.short += 1
                elif sample_length > actual_transcript_len:
                    self.long += 1
            
            # Convert to text
            sample_text = self.tokenizer.ids_to_text(sample_labels.detach().cpu().tolist())
            translations.append(sample_text)
        
        return translations

def sample_categorical(categorical_probs, temperature=1.0, dp=False):
    if temperature == 0.0:
        return categorical_probs.max(dim=-1)  # Skip noise when temperature is 0 (sampling Gumbel is costly)
    noise = torch.rand_like(categorical_probs, dtype=(torch.float64 if dp else torch.float32))
    gumbel_noise = (-torch.log(noise)) ** temperature
    return (categorical_probs / gumbel_noise).max(dim=-1)