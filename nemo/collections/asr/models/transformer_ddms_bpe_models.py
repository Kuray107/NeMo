import einops
import torch

from nemo.collections.asr.models import EncDecTransfModelBPE
from nemo.collections.asr.parts.submodules.discrete_diffusion_scheduler import get_noise_scheduler

class EncDecTransfDDMSModelBPE(EncDecTransfModelBPE):
    def _prepare_decoder_input_and_target(self, token_ids, token_lengths):

        time = self.noise_schedule.sample_time(batch_size=token_ids.size(0), device=token_ids.device)
        mask_prob = 1 - time
        mask_prob = einops.repeat(mask_prob, 'b -> b t', t = token_ids.size(1))
        will_mask = torch.bernoulli(mask_prob).to(dtype=torch.bool).to(token_ids.device)
        masked_token_ids = torch.where(will_mask, 0, token_ids)

        return masked_token_ids
    
    def compute_audio_loss(self, batch):

        if batch is None:
            return 0

        signal, signal_len, transcript, transcript_len = batch
        input_ids, labels = transcript[:, 1:], transcript[:, 1:] # Remove the bos token for input_ids as well

        # Modify the input_ids by masking out some of the tokens
        input_ids = self._prepare_decoder_input_and_target(self, input_ids, transcript_len)

        transf_log_probs, encoded_len, enc_states, enc_mask = self.forward(
            input_signal=signal,
            input_signal_length=signal_len,
            transcript=input_ids,
            transcript_length=transcript_len,
        )

        transf_loss = self.transf_loss(log_probs=transf_log_probs, labels=labels)

        return transf_loss