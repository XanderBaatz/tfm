from flow_matching.utils.manifolds import Manifold
from torch import Tensor


class UnitFlatTorus(Manifold):
    r"""Represents a flat torus on the unit domain :math:`[0, 1)^D`.

    Compatible with fractional crystal coordinates in [0, 1)^3.
    """

    def __init__(self, scale: float = 1.0) -> None:
        super().__init__()
        self.scale = scale

    def expmap(self, x: Tensor, u: Tensor) -> Tensor:
        """Exponential map: x_new = (x + u) mod scale."""
        return (x + u) % self.scale

    def logmap(self, x: Tensor, y: Tensor) -> Tensor:
        """Logarithmic map: Minimal displacement d in [-0.5*scale, 0.5*scale)^D.

        Replaces manual `torus_logmap(x, y)` functions.
        """
        return (y - x + 0.5 * self.scale) % self.scale - 0.5 * self.scale

    def projx(self, x: Tensor) -> Tensor:
        """Project points into boundary [0, scale)."""
        return x % self.scale

    def proju(self, x: Tensor, u: Tensor) -> Tensor:
        """Tangent vectors pass through directly on flat space."""
        return u
