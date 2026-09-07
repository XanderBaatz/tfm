from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch


class Sampleable(ABC):
    """Distribution which can be sampled from."""

    @property
    @abstractmethod
    def dim(self) -> int:
        """Dimensionality of the distribution."""

    @abstractmethod
    def sample(self, num_samples: int) -> torch.Tensor:
        """Sample from object (e.g., distribution).

        Args:
            num_samples: the desired number of samples

        Returns:
            samples: shape (batch_size, dim)

        """


class LabeledSampleable(ABC):
    """Distribution which can be sampled from."""

    @abstractmethod
    def sample(self, num_samples: int) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Return (samples, labels).

        Args:
            num_samples: the desired number of samples

        Returns:
            samples: b, d
            labels: b

        """

