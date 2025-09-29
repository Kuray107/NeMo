import einops
import torch
from lightning.pytorch import Trainer
from omegaconf import DictConfig, OmegaConf, open_dict

from nemo.collections.asr.models import EncDecTransfModelBPE
from nemo.collections.asr.parts.submodules.discrete_diffusion_scheduler import get_noise_scheduler

class EncDecTransfDDMSModelBPE(EncDecTransfModelBPE):
    def __init__(self, cfg: DictConfig, trainer: Trainer = None):
        super().__init__(cfg=cfg, trainer=trainer)
        self.noise_schedule = get_noise_scheduler(cfg.transf_decoder)

    def _prepare_decoder_input_and_target(self, input_ids, target_ids):

        mask_prob = self.noise_schedule.sample_time(batch_size=input_ids.size(0), device=input_ids.device)
        mask_prob = einops.repeat(mask_prob, 'b -> b t', t = input_ids.size(1))
        will_mask = torch.bernoulli(mask_prob).to(dtype=torch.bool).to(input_ids.device)
        masked_input_ids = torch.where(will_mask, 0, input_ids)
        masked_target_ids = torch.where(~will_mask, self.tokenizer.pad_id, target_ids)
        breakpoint()

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