import torch
from flow_matching.path.scheduler import Scheduler, SchedulerOutput
from torch import Tensor


class TrigScheduler(Scheduler):
    """Trigonometric scheduler."""

    def __call__(self, t: Tensor) -> SchedulerOutput:
        theta = 0.5 * torch.pi * t

        alpha_t = torch.sin(theta)
        sigma_t = torch.cos(theta)

        d_alpha_t = 0.5 * torch.pi * torch.cos(theta)
        d_sigma_t = -0.5 * torch.pi * torch.sin(theta)

        return SchedulerOutput(
            alpha_t=alpha_t,
            sigma_t=sigma_t,
            d_alpha_t=d_alpha_t,
            d_sigma_t=d_sigma_t,
        )

    def snr_inverse(self, snr: Tensor) -> Tensor:
        return (2.0 / torch.pi) * torch.atan(torch.sqrt(snr))
