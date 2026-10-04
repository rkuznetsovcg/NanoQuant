"""Bounded input correlations and a stabilized Sylvester ADMM update."""
import torch


class BlockInputMetric:
    def __init__(self, blocks, eigenvalues, eigenvectors):
        self.blocks = blocks
        self.eigenvalues = eigenvalues
        self.eigenvectors = eigenvectors

    @classmethod
    @torch.no_grad()
    def from_covariance(cls, blocks, diagonal, exponent=0.5):
        matrices, eigenvalues, eigenvectors = [], [], []
        offset = 0
        for covariance in blocks:
            size = covariance.shape[0]
            scale = diagonal[offset:offset+size].float().clamp_min(1e-12).sqrt()
            transported = covariance.float() / scale[:, None] / scale[None, :]
            transported = (transported + transported.mT) * 0.5
            values, vectors = torch.linalg.eigh(transported)
            values = values.clamp_min(1e-6)
            tempered = values.pow(exponent)
            tempered = tempered * (values.sum() / tempered.sum().clamp_min(1e-12))
            matrices.append((vectors * tempered.unsqueeze(0)) @ vectors.mT)
            eigenvalues.append(tempered)
            eigenvectors.append(vectors)
            offset += size
        if offset != diagonal.numel():
            raise ValueError("Covariance blocks do not cover the input channels")
        return cls(matrices, eigenvalues, eigenvectors)

    def apply(self, value):
        """Left multiply by the block metric without a full dense matrix."""
        result = torch.empty_like(value)
        offset = 0
        for block in self.blocks:
            end = offset + block.shape[0]
            result[offset:end] = block @ value[offset:end]
            offset = end
        if offset != value.shape[0]:
            raise ValueError("Metric channel count does not match the operand")
        return result


@torch.no_grad()
def metric_solve_step(X, Y, Z, U, rho, reg, left_metric=None, right_metric=None, eps=1e-12):
    """Solve Gram F G + alpha F = X.T A Y G + rho(Z-U).

    Preserve NanoQuant's scale-dependent alpha. With identity metrics this
    reduces to its original stabilized normal equations. Cached channel-block
    eigensystems bound state; only the rank-sized Gram is diagonalized per step.
    """
    dtype = X.dtype
    X, Y, Z, U = (value.float() for value in (X, Y, Z, U))
    AX = X if left_metric is None else left_metric.apply(X)
    gram = X.mT @ AX
    gram = (gram + gram.mT) * 0.5
    alpha = (rho * gram.diagonal().mean().abs() + reg).clamp_min(eps)
    rhs = AX.mT @ Y
    if right_metric is not None:
        rhs = right_metric.apply(rhs.mT).mT
    rhs = rhs + rho * (Z-U)
    if right_metric is None:
        gram.diagonal().add_(alpha)
        chol, info = torch.linalg.cholesky_ex(gram)
        if info.is_cuda:
            torch._assert_async(info.eq(0), "Curvature ADMM system is not positive definite")
            result = torch.cholesky_solve(rhs, chol)
        elif info.item() == 0:
            result = torch.cholesky_solve(rhs, chol)
        else:
            result = torch.linalg.solve(gram, rhs)
    else:
        values, basis = torch.linalg.eigh(gram)
        values = values.clamp_min(0)
        rotated = basis.mT @ rhs
        result = torch.empty_like(rhs)
        offset = 0
        for spectrum, vectors in zip(right_metric.eigenvalues, right_metric.eigenvectors):
            end = offset + spectrum.numel()
            local = rotated[:, offset:end] @ vectors
            local = local / (values[:, None] * spectrum[None, :] + alpha)
            result[:, offset:end] = basis @ local @ vectors.mT
            offset = end
        if offset != rhs.shape[1]:
            raise ValueError("Right metric does not cover the solved channels")
    return result.to(dtype)
