import torch

def get_noise_scheduler(config):
    type = config.get("type", "log-linear")
    if type == 'log-linear':
        return LogLinear(config)
    else:
        raise ValueError(f"Invalid noise schedule type: {type}")


class LogLinear:
    def __init__(self, config):
        super().__init__()
        self.eps = config.get('eps', 1e-8)
        self.alpha_0 = config.get('alpha_0', 1.0)

    def sample_time(self, batch_size: int, device: torch.device, rng: torch.random.Generator = None, time_min: float = 1e-8, time_max: float = 1.0) -> torch.Tensor:
        """
        Randomly sample a batchsize of time_steps from U[self.time_min, self.time_max]
        Supports an external random number generator for better reproducibility
        This is a linear schedule from time_min to time_max
        """
        time = torch.rand((batch_size,), generator=rng, device=device)
        offset = torch.arange(batch_size, device=device)

        time = ((time + offset) / batch_size) % 1 
        time = time * (time_max - time_min) + time_min

        return time

    def compute_noise_parameters(self, t):
        t = (1 - self.eps) * t
        alpha_t = self.alpha_0 * (1 - t)
        dalpha_t = - self.alpha_0 * (1 - self.eps) * torch.ones_like(alpha_t)
        sigma_t = - torch.log(alpha_t) 
        return dalpha_t, alpha_t, sigma_t