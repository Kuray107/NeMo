import itertools
import os
import einops
from tqdm import tqdm
from typing import Any, Dict, List, Optional, Union

import editdistance
import torch
import torch.distributed as dist
from lightning.pytorch import Trainer
from omegaconf import DictConfig, OmegaConf, open_dict
from torch.utils.data import DataLoader
from torchmetrics.text import SacreBLEUScore
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
                reduction="mean",
            )
            self.ins_mask_ratio = 0.1
        self.ins_mask = True

    def _prepare_decoder_input_and_target(self, input_ids, target_ids, mask_prob=None):
        if mask_prob is None:
            mask_prob = self.noise_schedule.sample_time(batch_size=input_ids.size(0), device=input_ids.device)
        if self.ins_mask:
            ins_mask_prob = 1 - mask_prob
            if self.cfg.transf_decoder.dec_type == 'ddms_ctc':
                return self._prepare_decoder_input_and_target_ctc2(input_ids, target_ids, mask_prob)
            else:
                return self._prepare_decoder_input_and_target_ins(input_ids, target_ids, mask_prob)

        mask_prob = einops.repeat(mask_prob, 'b -> b t', t = input_ids.size(1))
        will_mask = torch.bernoulli(mask_prob).to(dtype=torch.bool).to(input_ids.device)
        masked_input_ids = torch.where(will_mask, 0, input_ids)
        masked_target_ids = torch.where(~will_mask, self.tokenizer.pad_id, target_ids)

        return masked_input_ids, masked_target_ids, will_mask

    def _prepare_decoder_input_and_target_ctc2(self, input_ids, target_ids, mask_prob):
        input_lengths = input_ids.ne(2).sum(-1)
        masked_input_ids = []
        masked_target_ids = []
        for input_id, target_id, input_length, mask_p in zip(input_ids, target_ids, input_lengths, mask_prob):
            num_mask = round(input_length.item() * mask_p.item()) + 1 # ensure at least one token is masked
            # randomly select the indices between 0 and input_length - 1
            mask_indices = torch.randperm(input_length.item(), device=input_ids.device)[:num_mask]
            input_id[mask_indices] = 0

            ins_mask_indices = torch.rand(input_length, device=input_ids.device) < 0.5
            num_ins_mask = ins_mask_indices.sum().item()
            ins_mask_indices = torch.arange(num_ins_mask, device=input_ids.device) + ins_mask_indices.nonzero(as_tuple=True)[0] + 1
            # create a new input_id and target_id with length input_length * 2 - 1 with all zeros and ones
            masked_input_id = torch.zeros(input_length + num_ins_mask, dtype=input_ids.dtype, device=input_ids.device)
            masked_target_id = torch.ones(input_length + num_ins_mask, dtype=input_ids.dtype, device=input_ids.device)
            # put the original input_id and target_id at the odd indices
            will_ins = torch.isin(torch.arange(masked_input_id.size(0), device=input_ids.device), ins_mask_indices)
            masked_input_id[~will_ins] = input_id[:input_length]
            masked_target_id[~will_ins] = target_id[:input_length]
            masked_target_id = torch.where(masked_input_id == 0, self.tokenizer.pad_id, masked_target_id)
            
            masked_input_ids.append(masked_input_id)
            masked_target_ids.append(masked_target_id)
        
        masked_input_ids = pad_sequence(masked_input_ids, batch_first=True, padding_value=2)
        masked_target_ids = pad_sequence(masked_target_ids, batch_first=True, padding_value=2)
        return masked_input_ids, masked_target_ids, masked_input_ids == 0

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
            assert sum(~will_ins) == input_length
            masked_input_id[~will_ins] = input_id[:input_length]
            masked_target_id[~will_ins] = target_id[:input_length]
            
            masked_input_ids.append(masked_input_id)
            masked_target_ids.append(masked_target_id)
        
        masked_input_ids = pad_sequence(masked_input_ids, batch_first=True, padding_value=2)
        masked_target_ids = pad_sequence(masked_target_ids, batch_first=True, padding_value=2)
        
        return masked_input_ids, masked_target_ids, masked_input_ids == 0

    def _sampler(self, input_ids, log_probs, mask_prob):
        if self.sampler == 'topk':
            if self.cfg.transf_decoder.dec_type == 'ddms_ctc':
                return self._topk_sampler_ctc(input_ids, log_probs, mask_prob)
            else:
                return self._topk_sampler(input_ids, log_probs, mask_prob)
        else:
            return self._random_sampler(input_ids, mask_prob)

    def _random_sampler(self, input_ids, mask_prob):
        mask_prob = torch.tensor([mask_prob])
        mask_prob = einops.repeat(mask_prob, 'b -> b t', t = input_ids.size(1))
        will_mask = torch.bernoulli(mask_prob).to(dtype=torch.bool).to(input_ids.device)
        masked_input_ids = torch.where(will_mask, 0, input_ids)
        
        return masked_input_ids, will_mask, None

    def _topk_sampler(self, input_ids, log_probs, mask_prob):
        if mask_prob == 1.0:
            return torch.zeros_like(input_ids), torch.ones_like(input_ids, dtype=torch.bool)
        
        topk = round(input_ids.size(1) * (1 - mask_prob.item())) # number of tokens to keep
        topk_log_probs, topk_indices = torch.topk(log_probs, k=topk, dim=-1) # find topk indices to keep
        will_mask = torch.ones_like(input_ids, dtype=torch.bool)
        will_mask = will_mask.scatter(1, topk_indices, False)
        masked_input_ids = torch.where(will_mask, 0, input_ids)
        
        return masked_input_ids, will_mask, None
    
    def _topk_sampler_ctc(self, input_ids, log_probs, mask_prob):
        input_length = input_ids.size(1)
        # topk_masked_input = self._topk_sampler(input_ids, log_probs, mask_prob)[0]
        topk_masked_input = self._random_sampler(input_ids, mask_prob)[0]
        if mask_prob == 1:
            return topk_masked_input, topk_masked_input == 0, None
        # topk_masked_input = self._random_sampler(input_ids, mask_prob)[0]
        assert topk_masked_input.size(1) == input_length

        # create a new input_id and target_id with length num_ins_mask + input_ids.size(1) with all zeros
        # masked_input_ids = torch.zeros(input_length * 2 - 1, dtype=input_ids.dtype, device=input_ids.device)
        # masked_input_ids[::2] = topk_masked_input[0]
        # masked_input_ids = masked_input_ids.unsqueeze(0)
        ins_mask_indices = torch.rand(input_length, device=input_ids.device) < 0.5
        num_ins_mask = ins_mask_indices.sum().item()
        ins_mask_indices = torch.arange(num_ins_mask, device=input_ids.device) + ins_mask_indices.nonzero(as_tuple=True)[0] + 1
        # create a new input_id and target_id with length input_length * 2 - 1 with all zeros and ones
        masked_input_ids = torch.zeros(input_length + num_ins_mask, dtype=input_ids.dtype, device=input_ids.device)
        # put the original input_id and target_id at the odd indices
        will_ins = torch.isin(torch.arange(masked_input_ids.size(0), device=input_ids.device), ins_mask_indices)
        masked_input_ids[~will_ins] = topk_masked_input[0][:input_length]
        masked_input_ids = masked_input_ids.unsqueeze(0)

        # will_ins = torch.zeros_like(masked_input_ids, dtype=torch.bool)
        # will_ins[0][1::2] = True

        return masked_input_ids, (masked_input_ids == 0), will_ins.squeeze(0)

    def compute_audio_loss(self, batch):

        if batch is None:
            return 0

        signal, signal_len, transcript, transcript_len = batch
        input_ids, labels = transcript[:, 1:].clone(), transcript[:, 1:] # Remove the bos token for input_ids as well

        # Modify the input_ids by masking out some of the tokens
        input_ids, labels, will_mask = self._prepare_decoder_input_and_target(input_ids, labels)

        transf_log_probs, encoded_len, enc_states, enc_mask = self.forward(
            input_signal=signal,
            input_signal_length=signal_len,
            transcript=input_ids,
            transcript_length=input_ids.ne(2).sum(-1),
        )
        ce_loss = self.transf_loss(log_probs=transf_log_probs, labels=labels)

        if 'ctc' in self.cfg.transf_decoder.dec_type:
            # Create 1-hot vectors for input_ids
            input_ids_1hot = F.one_hot(input_ids, num_classes=self.tokenizer.vocab_size).to(torch.float)
            input_ids_1hot = torch.log(input_ids_1hot + 1e-12)

            # Replace the masked positions with the predicted log_probs
            transf_log_probs = transf_log_probs * will_mask.unsqueeze(-1) + input_ids_1hot * (~will_mask.unsqueeze(-1))
    
            # transf_loss = 0.0
            # for transf_logprob, label, input_id, will_m in zip(transf_log_probs, labels, input_ids, will_mask):
            #     # split the transf_logprob and input_id based on will_m where will_m is True, you have to split it into segments
            #     segments = []
            #     start_idx = 0
            #     for i in range(len(will_m)):
            #         if will_m[i]:
            #             if i == 0 or not will_m[i - 1]:
            #                 start_idx = i
            #         else:
            #             if i > 0 and will_m[i - 1]:
            #                 end_idx = i
            #                 segments.append((start_idx, end_idx))
            #     if will_m[-1]:
            #         segments.append((start_idx, len(will_m)))

            #     new_log_probs = []
            #     input_lengths = []
            #     new_labels = []
            #     target_lengths = []
            #     assert len(segments) > 0, f"No masked segments found. will_m={will_m}"
            #     for (start, end) in segments:
            #         new_log_probs.append(transf_logprob[start:end])
            #         input_lengths.append(end - start)
            #         # here instead of append label[start:end], we need to remove all blank tokens (assumed to be 1)
            #         if (label[start:end] == 1).all():
            #             new_labels.append(label[start:end])
            #         else:
            #             new_labels.append(label[start:end][label[start:end] != 1])
            #         target_lengths.append(new_labels[-1].size(0))
                
            #     # pad the new_log_probs and new_labels to the max length
            #     transf_logprob = pad_sequence(new_log_probs, batch_first=True, padding_value=float('-inf'))
            #     label = pad_sequence(new_labels, batch_first=True, padding_value=2)

            #     transf_loss += self.ctc_loss(
            #         log_probs=transf_logprob, 
            #         targets=label, 
            #         input_lengths=torch.tensor(input_lengths, device=transf_log_probs.device),
            #         target_lengths=torch.tensor(target_lengths, device=transf_log_probs.device)
            #     )
            
            # # average the loss over the batch size
            # transf_loss = transf_loss / transf_log_probs.size(0)
            ctc_loss = self.ctc_loss(
                log_probs=transf_log_probs,
                targets=transcript[:, 1:],
                input_lengths=input_ids.ne(2).sum(-1),
                target_lengths=transcript[:, 1:].ne(2).sum(-1)
            )
            transf_loss = ctc_loss
            print (f"CE Loss: {ce_loss.item():.4f}, CTC Loss: {ctc_loss.item():.4f}")

        else:
            transf_loss = ce_loss
        return transf_loss


    def validation_step(self, batch, batch_idx, dataloader_idx=0, eval_mode="val"):
        signal, signal_len, transcript, transcript_len = batch
        input_ids, labels = transcript[:, 1:].clone(), transcript[:, 1:]
        
        # Modify the input_ids by masking out some of the tokens
        # For validation efficiency, we only test on all-mask prediction
        mask_prob = torch.tensor([0.8]).repeat(input_ids.size(0))
        input_ids, labels, _ = self._prepare_decoder_input_and_target(
            input_ids, labels, mask_prob=mask_prob
        )

        if isinstance(batch, DALIOutputs) and batch.has_processed_signal:
            transf_log_probs, encoded_len, enc_states, enc_mask = self.forward(
                processed_signal=signal,
                processed_signal_length=signal_len,
                transcript=input_ids,
                transcript_length=input_ids.ne(2).sum(-1),
            )
        else:
            transf_log_probs, encoded_len, enc_states, enc_mask = self.forward(
                input_signal=signal,
                input_signal_length=signal_len,
                transcript=input_ids,
                transcript_length=input_ids.ne(2).sum(-1),
            )
        prediction_logprobs, prediction_labels = transf_log_probs.max(dim=-1)

        ground_truths = [self.tokenizer.ids_to_text(sent) for sent in transcript.detach().cpu().tolist()]
        
        if self.cfg.transf_decoder.dec_type == 'ddms_ctc':
            translations = []
            for pred_label in prediction_labels:
                pred_label = torch.unique_consecutive(pred_label)
                pred_label = pred_label[pred_label != 1]
                translations.append(self.tokenizer.ids_to_text(pred_label))
            
            labels = transcript[:, 1:]
            loss = self.ctc_loss(
                log_probs=transf_log_probs, 
                targets=labels, 
                input_lengths=input_ids.ne(2).sum(-1), 
                target_lengths=labels.ne(2).sum(-1) # here we remove the eos token
            )

        else:
            translations = [
                self.tokenizer.ids_to_text(sent) for sent in prediction_labels.detach().cpu().tolist()
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
            if i > 11:
                break
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
        # input_ids = transcript[:, 1:].to(self.device)
        time_steps = torch.linspace(0.0, 1.0, self.num_steps + 1)[1:]
        prev_prediction_labels = torch.ones_like(input_ids)
        prev_prediction_logprobs = torch.zeros_like(input_ids)
        prediction_logprobs = None
        for mask_prob in torch.tensor([1.0, 0.8]):
            input_ids, will_mask, will_ins = self._sampler(
                input_ids, prediction_logprobs, mask_prob=mask_prob
            )

            if isinstance(batch, DALIOutputs) and batch.has_processed_signal:
                transf_log_probs, encoded_len, enc_states, enc_mask = self.forward(
                    processed_signal=signal,
                    processed_signal_length=signal_len,
                    transcript=input_ids,
                    transcript_length=input_ids.ne(2).sum(-1),
                )
            else:
                transf_log_probs, encoded_len, enc_states, enc_mask = self.forward(
                    input_signal=signal,
                    input_signal_length=signal_len,
                    transcript=input_ids,
                    transcript_length=input_ids.ne(2).sum(-1),
                )
            prediction_logprobs, prediction_labels = transf_log_probs.max(dim=-1)

            # carry-over unmasking
            if mask_prob < 1.0 and carry_over_unmask:
                # for ctc with insertion mask, we need to handle the carry-over differently
                if will_ins is not None:
                    assert sum(~will_ins) == prev_prediction_labels.size(1)
                    ins_prev_prediction_labels = torch.zeros_like(prediction_labels)
                    ins_prev_prediction_labels[~will_ins.unsqueeze(0)] = prev_prediction_labels
                    prev_prediction_labels = ins_prev_prediction_labels
                    ins_prev_prediction_logprobs = torch.zeros_like(prediction_logprobs)
                    ins_prev_prediction_logprobs[~will_ins.unsqueeze(0)] = prev_prediction_logprobs
                    prev_prediction_logprobs = ins_prev_prediction_logprobs
                prediction_labels = torch.where(~will_mask, prev_prediction_labels, prediction_labels)
                prediction_logprobs = torch.where(~will_mask, prev_prediction_logprobs, prediction_logprobs)
            
            # for ctc model, remove the blank tokens
            if self.cfg.transf_decoder.dec_type == 'ddms_ctc':
                prediction_labels, prediction_logprobs = greedy_ctc_decoder(
                    prediction_labels[0], prediction_logprobs[0]
                )
                prediction_labels = prediction_labels.unsqueeze(0)
                prediction_logprobs = prediction_logprobs.unsqueeze(0)

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

def greedy_ctc_decoder(prediction_labels, prediction_logprobs):
    indices, counts = torch.unique_consecutive(prediction_labels, return_counts=True)
    
    start_idx = 0
    new_log_probs = []
    new_pred_labels = []
    for indice, count in zip(indices, counts):
        if indice == 1:
            start_idx += count
            continue
        avg_log_prob = prediction_logprobs[start_idx:start_idx + count].mean()
        new_log_probs.append(avg_log_prob)
        start_idx += count

        new_pred_labels.append(indice)

    return torch.tensor(new_pred_labels, device=prediction_labels.device), torch.tensor(new_log_probs, device=prediction_logprobs.device)