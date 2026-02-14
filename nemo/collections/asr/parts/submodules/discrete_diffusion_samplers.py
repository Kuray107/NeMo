import torch
import einops
import torch.nn.functional as F
import numpy as np

from nemo.utils import logging

def sample_categorical(categorical_probs, temperature=1.0, dp=False):
    if temperature == 0.0:
        return categorical_probs.argmax(dim=-1)  # Skip noise when temperature is 0 (sampling Gumbel is costly)
    noise = torch.rand_like(categorical_probs, dtype=(torch.float64 if dp else torch.float32))
    gumbel_noise = (-torch.log(noise)) ** temperature
    return (categorical_probs / gumbel_noise).argmax(dim=-1)


def get_sampler(config, model):
    type = config.get("type", "ancestral-cache")    
    if type == 'ancestral-cache':
        return AncestralCacheSampler(config, model)
    elif type == 'conf-top-k':
        return ConfTopKSampler(config, model)
    elif type == 'conf-top-k-margin':
        return ConfTopKMarginSampler(config, model)
    elif type == 'EB-conf-top-k':
        return EntropyBoundedConfTopKSampler(config, model)
    elif type == 'EB-conf-top-k-pos-biased':
        return EntropyBoundedPositionalBiasedConfTopKSampler(config, model)
    elif type == 'conf-top-p':
        return ConfTopPSampler(config, model)
    elif type == 'DFM':
        return DiscreteFlowMatchingSampler(config, model)
    else:
        raise ValueError(f"Invalid sampler: {type}")



class Sampler:
    def __init__(self, config, model):
        super().__init__()
        self.config = config
        self.model = model
        self.use_greedy = config.get("use_greedy", False)
        self.p_nucleus = config.get("p_nucleus", 1.0)
        self.use_float64 = config.get("use_float64", True)
        self.num_steps = config.get("num_steps", 16)
        self.mask_id = model.mask_id
        self.pad_id = model.pad_id
        self.eos_id = model.eos_id

    def update(self, **kwargs):
        raise NotImplementedError
    
    def get_pred_ids_and_probs(self, log_probs):
        # Zero Masking Probability
        log_probs[:, :, self.pad_id] = -torch.inf
        log_probs[:, :, self.mask_id] = -torch.inf

        p_x0 = log_probs.exp()
        p_x0 = self.do_nucleus_sampling(p_x0)

        if self.use_greedy:
            pred_ids = p_x0.argmax(dim=-1)
        else:
            pred_ids = sample_categorical(p_x0, dp=self.use_float64)

        return p_x0, pred_ids

    def do_nucleus_sampling(self, p_x0):
        if self.p_nucleus < 1:
            sorted_probs, sorted_indices = torch.sort(p_x0, descending=True, dim=-1)
            cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
            top_p_mask = cumulative_probs <= self.p_nucleus
            top_p_mask[..., 0] = True #always authorize at least the maximum-prob token 
            nucleus_probs = sorted_probs * top_p_mask
            nucleus_probs /= nucleus_probs.sum(dim=-1, keepdim=True)
            p_x0 = torch.zeros_like(p_x0).scatter_(-1, sorted_indices, nucleus_probs)
        
        return p_x0

    def update_length(self, token_ids, copy_flag):

        # # find all <eos> token and update lengths for each sample in the batch
        seq_len = token_ids.size(1)
        
        # Find EOS position for each sample in the batch.
        # If no EOS, keep length as seq_len and do not pad/truncate.
        eos_mask = token_ids == self.eos_id
        if not eos_mask.any():
            new_token_lengths = torch.full(
                (token_ids.size(0),),
                seq_len,
                dtype=torch.long,
                device=token_ids.device,
            )
            return token_ids, copy_flag, new_token_lengths
        idx = torch.arange(seq_len, device=token_ids.device).unsqueeze(0)
        first_eos = torch.where(eos_mask, idx, seq_len).min(dim=1).values
        eos_positions = torch.where(
            eos_mask.any(dim=1),
            first_eos + 1,  # +1 to include EOS token
            torch.full_like(first_eos, seq_len),
        )
        
        # Update input_ids_length to reflect actual sequence length (up to EOS) for each sample
        new_token_lengths = eos_positions
        
        # Find the maximum length needed to keep all samples aligned
        max_len = eos_positions.max().item()
        
        # Truncate all samples to max_len (but keep track of individual lengths via input_ids_length)
        new_ids = token_ids[:, :max_len]
        new_copy_flag = copy_flag[:, :max_len]
        # Pad everything after the first EOS (per sample).
        pos = torch.arange(max_len, device=token_ids.device).unsqueeze(0)
        pad_after_eos = pos >= eos_positions.unsqueeze(1)
        new_ids = new_ids.masked_fill(pad_after_eos, self.pad_id)
        new_copy_flag = new_copy_flag.masked_fill(pad_after_eos, True)

        return new_ids, new_copy_flag, new_token_lengths



class AncestralCacheSampler(Sampler):

    def __init__(self, config, model):
        super().__init__(config, model)
        logging.info(f"AncestralCacheSampler initialized with num_steps: {self.num_steps}")

    def update(self, current_ids, copy_flag, alpha_t, alpha_s, log_probs, is_last_step=False, **kwargs):

        p_x0, pred_ids = self.get_pred_ids_and_probs(log_probs)
        
        if not is_last_step:
            mask_prob = torch.tensor([(1 - alpha_s) / (1 - alpha_t)], device=current_ids.device)
            assert torch.all(mask_prob <= 1.0) and torch.all(mask_prob >= 0.0), f"mask_prob: {mask_prob} for alpha_t: {alpha_t} and alpha_s: {alpha_s}"
            mask_prob = einops.repeat(mask_prob, 'b -> b t', t = current_ids.shape[1])
            will_be_updated = torch.bernoulli(1 - mask_prob).to(dtype=torch.bool).to(pred_ids.device)
        else: 
            # Last step: update all tokens
            will_be_updated = torch.ones_like(current_ids, dtype=torch.bool)
        # carry-over unmasking
        new_ids = torch.where(will_be_updated, pred_ids, current_ids)
        new_ids = torch.where(copy_flag, current_ids, new_ids)

        # update copy_flag according to will_be_updated
        new_copy_flag = copy_flag | will_be_updated
        new_ids, new_copy_flag, new_ids_length = self.update_length(new_ids, new_copy_flag)

        return new_ids, new_ids_length, new_copy_flag



class ConfTopKSampler(Sampler):

    def __init__(self, config, model):
        super().__init__(config, model)
        logging.info(f"AncestralConfTopKSampler initialized with num_steps: {self.num_steps}")

    def update(self, current_ids, copy_flag, alpha_t, alpha_s, log_probs, is_last_step=False, **kwargs):
        p_x0, pred_ids = self.get_pred_ids_and_probs(log_probs)

        masked_flag = ~copy_flag
        new_ids = current_ids.clone()
        new_copy_flag = copy_flag.clone()
        pred_conf = torch.gather(p_x0, dim=-1, index=pred_ids.unsqueeze(-1)).squeeze(-1)
        pred_conf = torch.where(masked_flag, pred_conf, torch.zeros_like(pred_conf))

        if is_last_step:
            new_ids = torch.where(copy_flag, current_ids, pred_ids)
            new_copy_flag = torch.ones_like(copy_flag, dtype=torch.bool, device=copy_flag.device)
        else:
            # Calculate base number of tokens to update per step
            base_tokens_per_step = current_ids.shape[1] // self.num_steps
            
            # Handle remainder tokens with probabilistic distribution
            remainder_tokens = current_ids.shape[1] % self.num_steps
            if remainder_tokens > 0:
                # Distribute remainder tokens probabilistically across steps (per sample)
                prob_additional_token = remainder_tokens / self.num_steps
                add_token = (torch.rand(current_ids.size(0), device=current_ids.device) < prob_additional_token).long()
            else:
                add_token = torch.zeros(current_ids.size(0), dtype=torch.long, device=current_ids.device)

            base_tokens_per_step = torch.full(
                (current_ids.size(0),),
                base_tokens_per_step,
                dtype=torch.long,
                device=current_ids.device,
            ) + add_token

            masked_counts = masked_flag.sum(dim=1)
            num_tokens_per_sample = torch.minimum(base_tokens_per_step, masked_counts)
            # Ensure at least 1 token is updated when there are masked tokens.
            num_tokens_per_sample = torch.where(
                masked_counts > 0,
                torch.clamp(num_tokens_per_sample, min=1),
                torch.zeros_like(num_tokens_per_sample),
            )

            # update new_ids and new_state according to the top-k tokens (per sample)
            k_max = int(num_tokens_per_sample.max().item())
            if k_max > 0:
                topk_indices = torch.topk(pred_conf, k=k_max, dim=-1).indices  # [B, k_max]
                rank_mask = torch.arange(k_max, device=current_ids.device).unsqueeze(0)
                select_mask = rank_mask < num_tokens_per_sample.unsqueeze(1)  # [B, k_max]

                batch_idx = torch.arange(current_ids.size(0), device=current_ids.device).unsqueeze(1).expand_as(topk_indices)
                selected_batches = batch_idx[select_mask]
                selected_positions = topk_indices[select_mask]

                new_ids[selected_batches, selected_positions] = pred_ids[selected_batches, selected_positions]
                new_copy_flag[selected_batches, selected_positions] = True

        new_ids, new_copy_flag, new_ids_length = self.update_length(new_ids, new_copy_flag)

        return new_ids, new_ids_length, new_copy_flag


class ConfTopKMarginSampler(Sampler):

    def __init__(self, config, model):
        super().__init__(config, model)
        logging.info(f"ConfTopKMarginSampler initialized with num_steps: {self.num_steps}")

    def update(self, current_ids, copy_flag, alpha_t, alpha_s, log_probs, is_last_step=False, **kwargs):
        p_x0, pred_ids = self.get_pred_ids_and_probs(log_probs)

        masked_flag = ~copy_flag
        new_ids = current_ids.clone()
        new_copy_flag = copy_flag.clone()
        top2 = torch.topk(p_x0, k=2, dim=-1).values
        margin_conf = top2[..., 0] - top2[..., 1]
        margin_conf = torch.where(masked_flag, margin_conf, torch.zeros_like(margin_conf))

        if is_last_step:
            new_ids = torch.where(copy_flag, current_ids, pred_ids)
            new_copy_flag = torch.ones_like(copy_flag, dtype=torch.bool, device=copy_flag.device)
        else:
            # Calculate base number of tokens to update per step
            base_tokens_per_step = current_ids.shape[1] // self.num_steps
            
            # Handle remainder tokens with probabilistic distribution
            remainder_tokens = current_ids.shape[1] % self.num_steps
            if remainder_tokens > 0:
                # Distribute remainder tokens probabilistically across steps (per sample)
                prob_additional_token = remainder_tokens / self.num_steps
                add_token = (torch.rand(current_ids.size(0), device=current_ids.device) < prob_additional_token).long()
            else:
                add_token = torch.zeros(current_ids.size(0), dtype=torch.long, device=current_ids.device)

            base_tokens_per_step = torch.full(
                (current_ids.size(0),),
                base_tokens_per_step,
                dtype=torch.long,
                device=current_ids.device,
            ) + add_token

            masked_counts = masked_flag.sum(dim=1)
            num_tokens_per_sample = torch.minimum(base_tokens_per_step, masked_counts)
            # Ensure at least 1 token is updated when there are masked tokens.
            num_tokens_per_sample = torch.where(
                masked_counts > 0,
                torch.clamp(num_tokens_per_sample, min=1),
                torch.zeros_like(num_tokens_per_sample),
            )

            # update new_ids and new_state according to the top-k tokens (per sample)
            k_max = int(num_tokens_per_sample.max().item())
            if k_max > 0:
                topk_indices = torch.topk(margin_conf, k=k_max, dim=-1).indices  # [B, k_max]
                rank_mask = torch.arange(k_max, device=current_ids.device).unsqueeze(0)
                select_mask = rank_mask < num_tokens_per_sample.unsqueeze(1)  # [B, k_max]

                batch_idx = torch.arange(current_ids.size(0), device=current_ids.device).unsqueeze(1).expand_as(topk_indices)
                selected_batches = batch_idx[select_mask]
                selected_positions = topk_indices[select_mask]

                new_ids[selected_batches, selected_positions] = pred_ids[selected_batches, selected_positions]
                new_copy_flag[selected_batches, selected_positions] = True

        new_ids, new_copy_flag, new_ids_length = self.update_length(new_ids, new_copy_flag)

        return new_ids, new_ids_length, new_copy_flag


class EntropyBoundedConfTopKSampler(Sampler):
    def __init__(self, config, model):
        super().__init__(config, model)
        self.gamma = config.get("gamma", 0.1)
        logging.info(f"ConfTopKEntropySampler initialized with num_steps: {self.num_steps}")
        logging.info(f"gamma: {self.gamma}")

    def update(self, current_ids, copy_flag, alpha_t, alpha_s, log_probs, is_last_step=False, **kwargs):
        p_x0, pred_ids = self.get_pred_ids_and_probs(log_probs)

        masked_flag = ~copy_flag
        new_ids = current_ids.clone()
        new_copy_flag = copy_flag.clone()

        pred_conf = torch.gather(p_x0, dim=-1, index=pred_ids.unsqueeze(-1)).squeeze(-1)
        err = 1.0 - pred_conf
        err = torch.where(masked_flag, err, torch.full_like(err, float("inf")))

        entropy = torch.distributions.Categorical(probs=p_x0).entropy()

        if is_last_step:
            new_ids = torch.where(copy_flag, current_ids, pred_ids)
            new_copy_flag = torch.ones_like(copy_flag, dtype=torch.bool, device=copy_flag.device)
        else:
            sorted_err, sorted_idx = torch.sort(err, dim=-1)
            entropy_sorted = entropy.gather(dim=-1, index=sorted_idx)
            acc_entropy = torch.cumsum(entropy_sorted, dim=-1)
            cummax_entropy = torch.cummax(entropy_sorted, dim=-1).values

            num_tokens_per_sample = (acc_entropy - cummax_entropy <= self.gamma).sum(dim=-1)
            masked_counts = masked_flag.sum(dim=1)
            num_tokens_per_sample = torch.minimum(num_tokens_per_sample, masked_counts)
            num_tokens_per_sample = torch.where(
                masked_counts > 0,
                torch.clamp(num_tokens_per_sample, min=1),
                torch.zeros_like(num_tokens_per_sample),
            )

            k_max = int(num_tokens_per_sample.max().item())
            topk_indices = sorted_idx[:, :k_max]
            rank_mask = torch.arange(k_max, device=current_ids.device).unsqueeze(0)
            select_mask = rank_mask < num_tokens_per_sample.unsqueeze(1)

            batch_idx = torch.arange(current_ids.size(0), device=current_ids.device).unsqueeze(1).expand_as(topk_indices)
            selected_batches = batch_idx[select_mask]
            selected_positions = topk_indices[select_mask]

            new_ids[selected_batches, selected_positions] = pred_ids[selected_batches, selected_positions]
            new_copy_flag[selected_batches, selected_positions] = True

        new_ids, new_copy_flag, new_ids_length = self.update_length(new_ids, new_copy_flag)
        return new_ids, new_ids_length, new_copy_flag

class EntropyBoundedPositionalBiasedConfTopKSampler(Sampler):
    def __init__(self, config, model):
        super().__init__(config, model)
        self.lambda_val = config.get("lambda_val", 0.1)
        self.gamma = config.get("gamma", 0.1)
        logging.info(f"EntropyBoundedPositionalBiasedConfTopKSampler initialized with num_steps: {self.num_steps}")
        logging.info(f"lambda_val: {self.lambda_val}")
        logging.info(f"gamma: {self.gamma}")

    def update(self, current_ids, copy_flag, alpha_t, alpha_s, log_probs, is_last_step=False, **kwargs):
        p_x0, pred_ids = self.get_pred_ids_and_probs(log_probs)

        masked_flag = ~copy_flag
        new_ids = current_ids.clone()
        new_copy_flag = copy_flag.clone()

        pred_conf = torch.gather(p_x0, dim=-1, index=pred_ids.unsqueeze(-1)).squeeze(-1)
        position_ids = torch.arange(current_ids.shape[1], device=current_ids.device, dtype=pred_conf.dtype)
        positional_bias = torch.exp(-self.lambda_val * position_ids).unsqueeze(0)
        biased_conf = pred_conf * positional_bias
        err = 1.0 - biased_conf
        err = torch.where(masked_flag, err, torch.full_like(err, float("inf")))

        entropy = torch.distributions.Categorical(probs=p_x0).entropy()

        if is_last_step:
            new_ids = torch.where(copy_flag, current_ids, pred_ids)
            new_copy_flag = torch.ones_like(copy_flag, dtype=torch.bool, device=copy_flag.device)
        else:
            sorted_err, sorted_idx = torch.sort(err, dim=-1)
            entropy_sorted = entropy.gather(dim=-1, index=sorted_idx)
            acc_entropy = torch.cumsum(entropy_sorted, dim=-1)
            cummax_entropy = torch.cummax(entropy_sorted, dim=-1).values

            num_tokens_per_sample = (acc_entropy - cummax_entropy <= self.gamma).sum(dim=-1)
            masked_counts = masked_flag.sum(dim=1)
            num_tokens_per_sample = torch.minimum(num_tokens_per_sample, masked_counts)
            num_tokens_per_sample = torch.where(
                masked_counts > 0,
                torch.clamp(num_tokens_per_sample, min=1),
                torch.zeros_like(num_tokens_per_sample),
            )

            k_max = int(num_tokens_per_sample.max().item())
            topk_indices = sorted_idx[:, :k_max]
            rank_mask = torch.arange(k_max, device=current_ids.device).unsqueeze(0)
            select_mask = rank_mask < num_tokens_per_sample.unsqueeze(1)

            batch_idx = torch.arange(current_ids.size(0), device=current_ids.device).unsqueeze(1).expand_as(topk_indices)
            selected_batches = batch_idx[select_mask]
            selected_positions = topk_indices[select_mask]

            new_ids[selected_batches, selected_positions] = pred_ids[selected_batches, selected_positions]
            new_copy_flag[selected_batches, selected_positions] = True

        new_ids, new_copy_flag, new_ids_length = self.update_length(new_ids, new_copy_flag)
        return new_ids, new_ids_length, new_copy_flag

class ConfTopPSampler(Sampler):
    def __init__(self, config, model):
        super().__init__(config, model)
        self.top_p_conf = config.get("top_p_conf", 0.8)
        self.warm_up_alpha_threshold = config.get("warm_up_alpha_threshold", 0.0)
        self.fallback_strategy = config.get("fallback_strategy", "top-k")

        logging.info(f"ConfTopPSampler initialized with num_steps: {self.num_steps}")
        logging.info(f"top_p_conf: {self.top_p_conf}")
        logging.info(f"warm_up_alpha_threshold: {self.warm_up_alpha_threshold}")
        logging.info(f"fallback_strategy: {self.fallback_strategy}")

    def update(self, current_ids, copy_flag, alpha_t, alpha_s, log_probs, is_last_step=False, **kwargs):
        
        p_x0, pred_ids = self.get_pred_ids_and_probs(log_probs)

        if is_last_step:
            new_ids = torch.where(copy_flag, current_ids, pred_ids)
            new_copy_flag = torch.ones_like(copy_flag, dtype=torch.bool, device=copy_flag.device)
        else:
            masked_flag = ~copy_flag
            new_ids = current_ids.clone()
            new_copy_flag = copy_flag.clone()
            pred_conf = torch.gather(p_x0, dim=-1, index=pred_ids.unsqueeze(-1)).squeeze(-1)
            pred_conf = torch.where(masked_flag, pred_conf, torch.zeros_like(pred_conf))

            total_tokens = current_ids.shape[1]
            select_mask = (pred_conf > self.top_p_conf) & masked_flag
            use_conf_mask = (alpha_t > self.warm_up_alpha_threshold) & select_mask.any(dim=1)
            fallback_mask = ~use_conf_mask

            # Fallback: either we are in the warm-up phase or we don't have any tokens with enough confidence
            masked_counts = masked_flag.sum(dim=1)
            base_tokens_per_step = total_tokens // self.num_steps
            remainder_tokens = total_tokens % self.num_steps
            if remainder_tokens > 0:
                # Distribute remainder tokens probabilistically across steps (per sample)
                prob_additional_token = remainder_tokens / self.num_steps
                add_token = (torch.rand(current_ids.size(0), device=current_ids.device) < prob_additional_token).long()
            else:
                add_token = torch.zeros(current_ids.size(0), dtype=torch.long, device=current_ids.device)

            base_tokens_per_step = torch.full(
                (current_ids.size(0),),
                base_tokens_per_step,
                dtype=torch.long,
                device=current_ids.device,
            ) + add_token

            num_tokens_per_sample = torch.minimum(base_tokens_per_step, masked_counts)
            num_tokens_per_sample = torch.where(
                masked_counts > 0,
                torch.clamp(num_tokens_per_sample, min=1),
                torch.zeros_like(num_tokens_per_sample),
            )
            if self.fallback_strategy == 'greedy':
                num_tokens_per_sample = torch.where(
                    masked_counts > 0,
                    torch.ones_like(num_tokens_per_sample),
                    torch.zeros_like(num_tokens_per_sample),
                )

            # Apply confidence-based selection where applicable
            if use_conf_mask.any():
                conf_selected = select_mask & use_conf_mask.unsqueeze(1)
                batch_idx = (
                    torch.arange(current_ids.size(0), device=current_ids.device)
                    .unsqueeze(1)
                    .expand_as(conf_selected)
                )
                conf_batches = batch_idx[conf_selected]
                conf_positions = torch.arange(total_tokens, device=current_ids.device).unsqueeze(0).expand_as(select_mask)[conf_selected]
                new_ids[conf_batches, conf_positions] = pred_ids[conf_batches, conf_positions]
                new_copy_flag[conf_batches, conf_positions] = True

            # Apply fallback selection for remaining samples
            if fallback_mask.any():
                scores = pred_conf if self.fallback_strategy in ("top-k", "greedy") else torch.rand_like(pred_conf)
                scores = scores.masked_fill(~masked_flag, float("-inf"))
                k_max = int(num_tokens_per_sample.max().item())
                if k_max > 0:
                    topk_indices = torch.topk(scores, k=k_max, dim=-1).indices  # [B, k_max]
                    rank_mask = torch.arange(k_max, device=current_ids.device).unsqueeze(0)
                    select_fallback = rank_mask < num_tokens_per_sample.unsqueeze(1)
                    select_fallback &= fallback_mask.unsqueeze(1)

                    batch_idx = torch.arange(current_ids.size(0), device=current_ids.device).unsqueeze(1).expand_as(topk_indices)
                    selected_batches = batch_idx[select_fallback]
                    selected_positions = topk_indices[select_fallback]

                    new_ids[selected_batches, selected_positions] = pred_ids[selected_batches, selected_positions]
                    new_copy_flag[selected_batches, selected_positions] = True

            logging.debug(f"decode_token_number: {int((new_copy_flag & ~copy_flag).sum().item())}")
        
        new_ids, new_copy_flag, new_ids_length = self.update_length(new_ids, new_copy_flag)

        return new_ids, new_ids_length, new_copy_flag



class DiscreteFlowMatchingSampler(Sampler):
    def __init__(self, config, model):
        super().__init__(config, model)
        logging.info(f"DiscreteFlowMatchingSampler initialized with num_steps: {self.num_steps}")

    def _get_step_size(self, alpha_t, alpha_s):    
        step_size = alpha_s - alpha_t
        return step_size.clamp(min=0)

    def update(self, current_ids, copy_flag, alpha_t, alpha_s, log_probs, is_last_step=False, **kwargs):
        p_x0, pred_ids = self.get_pred_ids_and_probs(log_probs)

        if is_last_step:
            new_ids = torch.where(copy_flag, current_ids, pred_ids)
            new_copy_flag = torch.ones_like(copy_flag, dtype=torch.bool, device=copy_flag.device)
            new_ids, new_copy_flag, new_ids_length = self.update_length(new_ids, new_copy_flag)
        else:
            one_hot_x_t = F.one_hot(current_ids, num_classes=log_probs.shape[-1]).float()
            u = (p_x0 - one_hot_x_t) / (1.0 - alpha_t)
            step_size = self._get_step_size(alpha_t, alpha_s)
            new_ids = sample_categorical(categorical_probs=one_hot_x_t + step_size * u, dp=self.use_float64)
            new_copy_flag = new_ids != self.mask_id
            new_ids, new_copy_flag, new_ids_length = self.update_length(new_ids, new_copy_flag)

        return new_ids, new_ids_length, new_copy_flag


class ReMDMCap(Sampler):
    def __init__(self, config, model):
        super().__init__(config, model)
        self.eta_cap = config.get("eta_cap", 0.05)

    def update(self, current_state, current_ids, copy_flag, alpha_t, alpha_s, log_probs, is_last_step=False, **kwargs):
        # Zero Masking Probability
        log_probs[:, :, self.model.speech_padding_id] = -torch.inf
        log_probs[:, :, self.model.speech_eos_id] = -torch.inf

        p_x0 = log_probs.exp()
        p_x0 = self.do_nucleus_sampling(p_x0)

        pred_speech_token_ids = sample_categorical(p_x0, dp=self.use_float64)
        pred_speech_token_embeds = self.model.speech_token_embeddings(pred_speech_token_ids) # (B, T, D)

        # Copy Flag will be remasked according to the eta
        if alpha_t > 0:
            sigma = torch.min(torch.ones_like(alpha_t) * self.eta_cap, (1 - alpha_s) / alpha_t).unsqueeze(0)
        else:
            sigma = torch.ones_like(alpha_t).unsqueeze(0) * self.eta_cap

        remask_prob = einops.repeat(sigma, 'b -> b t', t = current_state.shape[1])
        will_be_remasked = torch.bernoulli(remask_prob).to(dtype=torch.bool).to(copy_flag.device)
        copy_flag = torch.where(will_be_remasked.unsqueeze(-1), torch.zeros_like(copy_flag, dtype=torch.bool, device=copy_flag.device), copy_flag)


        if not is_last_step:
            mask_prob = torch.tensor([(1 - alpha_s - alpha_t * sigma) / (1 - alpha_t)], device=current_state.device)
            assert mask_prob <= 1.0 and mask_prob >= 0.0, f"mask_prob: {mask_prob} for alpha_t: {alpha_t} and alpha_s: {alpha_s}"
            mask_prob = einops.repeat(mask_prob, 'b -> b t', t = current_state.shape[1])
            will_be_updated = torch.bernoulli(1 - mask_prob).to(dtype=torch.bool).to(pred_speech_token_ids.device)
        else: 
            # Last step: update all tokens
            will_be_updated = torch.ones_like(current_ids, dtype=torch.bool)
        
        will_be_updated = einops.rearrange(will_be_updated, 'b t -> b t 1')
        new_state = torch.where(will_be_updated.expand_as(current_state), pred_speech_token_embeds, current_state)
        new_state = torch.where(copy_flag.expand_as(current_state), current_state, new_state)
        new_ids = torch.where(will_be_updated.squeeze(-1), pred_speech_token_ids, current_ids)
        new_ids = torch.where(copy_flag.squeeze(-1), current_ids, new_ids)

        # update copy_flag according to will_be_updated
        new_copy_flag = copy_flag | will_be_updated

        return new_state, new_ids, new_copy_flag



class ReMDMRescale(Sampler):
    def __init__(self, config, model):
        super().__init__(config, model)
        self.eta_rescale = config.get("eta_rescale", 0.05)

    def update(self, current_state, current_ids, copy_flag, alpha_t, alpha_s, log_probs, is_last_step=False, **kwargs):
        # Zero Masking Probability
        log_probs[:, :, self.model.speech_padding_id] = -torch.inf
        log_probs[:, :, self.model.speech_eos_id] = -torch.inf

        p_x0 = log_probs.exp()
        p_x0 = self.do_nucleus_sampling(p_x0)

        pred_speech_token_ids = sample_categorical(p_x0, dp=self.use_float64)
        pred_speech_token_embeds = self.model.speech_token_embeddings(pred_speech_token_ids) # (B, T, D)

        # Copy Flag will be remasked according to the eta
        if alpha_t > 0:
            sigma = torch.min(torch.ones_like(alpha_t), (1 - alpha_s) / alpha_t).unsqueeze(0) * self.eta_rescale
        else:
            sigma = torch.ones_like(alpha_t).unsqueeze(0) * self.eta_rescale

        remask_prob = einops.repeat(sigma, 'b -> b t', t = current_state.shape[1])
        will_be_remasked = torch.bernoulli(remask_prob).to(dtype=torch.bool).to(copy_flag.device)
        copy_flag = torch.where(will_be_remasked.unsqueeze(-1), torch.zeros_like(copy_flag, dtype=torch.bool, device=copy_flag.device), copy_flag)


        if not is_last_step:
            mask_prob = torch.tensor([(1 - alpha_s - alpha_t * sigma) / (1 - alpha_t)], device=current_state.device)
            assert mask_prob <= 1.0 and mask_prob >= 0.0, f"mask_prob: {mask_prob} for alpha_t: {alpha_t} and alpha_s: {alpha_s}"
            mask_prob = einops.repeat(mask_prob, 'b -> b t', t = current_state.shape[1])
            will_be_updated = torch.bernoulli(1 - mask_prob).to(dtype=torch.bool).to(pred_speech_token_ids.device)
        else: 
            # Last step: update all tokens
            will_be_updated = torch.ones_like(current_ids, dtype=torch.bool)
        will_be_updated = einops.rearrange(will_be_updated, 'b t -> b t 1')
        # carry-over unmasking
        new_state = torch.where(will_be_updated.expand_as(current_state), pred_speech_token_embeds, current_state)
        new_state = torch.where(copy_flag.expand_as(current_state), current_state, new_state)
        new_ids = torch.where(will_be_updated.squeeze(-1), pred_speech_token_ids, current_ids)
        new_ids = torch.where(copy_flag.squeeze(-1), current_ids, new_ids)

        # update copy_flag according to will_be_updated
        new_copy_flag = copy_flag | will_be_updated

        return new_state, new_ids, new_copy_flag



class ReMDMLoop(Sampler):

    def __init__(self, config, model):
        super().__init__(config, model)
        self.t_on = config.get("t_on", 0.55)
        self.t_off = config.get("t_off", 0.05)
        self.alpha_const = config.get("alpha_const", 0.9)
        self.eta_cap = config.get("eta_cap", 0.05)

    def update(self, current_state, current_ids, copy_flag, alpha_t, alpha_s, log_probs, is_last_step=False, **kwargs):

        # Zero Masking Probability
        log_probs[:, :, self.model.speech_padding_id] = -torch.inf
        log_probs[:, :, self.model.speech_eos_id] = -torch.inf

        p_x0 = log_probs.exp()
        p_x0 = self.do_nucleus_sampling(p_x0)

        pred_speech_token_ids = sample_categorical(p_x0, dp=self.use_float64)
        pred_speech_token_embeds = self.model.speech_token_embeddings(pred_speech_token_ids) # (B, T, D)

        time_t = 1 - alpha_t # Note that this works only for log-linear noise schedule
        time_s = 1 - alpha_s
        if time_t > self.t_on:
            rescale_alpha_t = self.alpha_const * (1 - time_t) / (1 - self.t_on)
            rescale_alpha_s = self.alpha_const * (1 - time_s) / (1 - self.t_on)
        elif time_t <= self.t_off:
            rescale_alpha_t = 1 - (1 - self.alpha_const) * time_t / self.t_off
            rescale_alpha_s = 1 - (1 - self.alpha_const) * time_s / self.t_off
        
        if is_last_step:
            will_be_updated = torch.ones_like(current_ids, dtype=torch.bool)        
        elif time_t > self.t_on or time_t <= self.t_off: # MDLM
            mask_prob = torch.tensor([(1 - rescale_alpha_s) / (1 - rescale_alpha_t)], device=current_state.device)
            assert mask_prob <= 1.0 and mask_prob >= 0.0, f"mask_prob: {mask_prob} for alpha_t: {rescale_alpha_t} and alpha_s: {rescale_alpha_s}"
            mask_prob = einops.repeat(mask_prob, 'b -> b t', t = current_state.shape[1])
            will_be_updated = torch.bernoulli(1 - mask_prob).to(dtype=torch.bool).to(pred_speech_token_ids.device)
        else: # use ReMDM
            sigma = torch.ones_like(alpha_t).unsqueeze(0) * self.eta_cap
            remask_prob = einops.repeat(sigma, 'b -> b t', t = current_state.shape[1])
            will_be_remasked = torch.bernoulli(remask_prob).to(dtype=torch.bool).to(copy_flag.device)
            copy_flag = torch.where(will_be_remasked.unsqueeze(-1), torch.zeros_like(copy_flag, dtype=torch.bool, device=copy_flag.device), copy_flag)

            mask_prob = torch.tensor([(1 - self.alpha_const - self.alpha_const * sigma) / (1 - self.alpha_const)], device=current_state.device)
            assert mask_prob <= 1.0 and mask_prob >= 0.0, f"mask_prob: {mask_prob} for sigma: {sigma} and alpha_const: {self.alpha_const}"
            mask_prob = einops.repeat(mask_prob, 'b -> b t', t = current_state.shape[1])
            will_be_updated = torch.bernoulli(1 - mask_prob).to(dtype=torch.bool).to(pred_speech_token_ids.device)
            
        will_be_updated = einops.rearrange(will_be_updated, 'b t -> b t 1')
        # carry-over unmasking
        new_state = torch.where(will_be_updated.expand_as(current_state), pred_speech_token_embeds, current_state)
        new_state = torch.where(copy_flag.expand_as(current_state), current_state, new_state)
        new_ids = torch.where(will_be_updated.squeeze(-1), pred_speech_token_ids, current_ids)
        new_ids = torch.where(copy_flag.squeeze(-1), current_ids, new_ids)

        # update copy_flag according to will_be_updated
        new_copy_flag = copy_flag | will_be_updated

        return new_state, new_ids, new_copy_flag


class ForwardBackward(Sampler):
    def _update(self, x, t, dt, logits=None, **kwargs):
        _, alpha_s, _ = self.model.noise(t - dt)
        _, alpha_t, sigma_t = self.model.noise(t)

        if logits is None:
            logits = self.model.forward(x, sigma_t)
        p_x0 = logits.exp()
        p_x0 = self.do_nucleus_sampling(p_x0)

        masked_flag = (x == self.mask_index).to(torch.bool)
        xs = x.clone()
        x0 = sample_categorical(p_x0, dp=self.use_float64) #Size [num_transfer_tokens] #Only sample from categorial for chosen token
        p = F.softmax(logits, dim=-1)
        x0_p = torch.gather(p, dim=-1, index=x0.unsqueeze(-1)).squeeze(-1)
        conf = torch.where(masked_flag, x0_p, torch.zeros_like(x0_p))

        if (alpha_t > 0).all():
          sigma = (alpha_s - alpha_t) / alpha_t
        else:
          sigma = 1
        q_xs = p_x0 * (1 - sigma)
        q_xs[..., self.mask_index] = sigma
        q_xs_2 = p_x0 * ((alpha_s - (1 - sigma) * alpha_t) / (1 - alpha_t))
        q_xs_2[..., self.mask_index] = (1 - alpha_s - sigma * alpha_t) / (1 - alpha_t)
        copy_flag = (x != self.mask_index).to(torch.bool)
        q_xs = torch.where(copy_flag.unsqueeze(-1), q_xs, q_xs_2)
        xs = sample_categorical(q_xs, dp=self.use_float64)

        logits_cache = logits if torch.allclose(xs, x) and not self.model.time_conditioning else None
        return logits_cache, xs, conf