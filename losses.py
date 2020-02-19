import torch
from torch import nn
from functools import partial


class L1Loss(nn.Module):
    name = 'l1'

    def __init__(self):
        super().__init__()

    def forward(self, x, y, normalizer=None):
        if not normalizer:
            return (x - y).abs().mean()
        else:
            return (x - y).abs().sum() / normalizer


class MSELoss(nn.Module):
    name = 'mse'

    def __init__(self):
        super().__init__()

    def forward(self, x, y, normalizer=None):
        if not normalizer:
            return ((x - y) ** 2).mean()
        else:
            return ((x - y) ** 2).sum() / normalizer


class SaturatingGANLoss(nn.Module):
    name = 'saturating-gan'
    need_lipschitz_d = False

    def __init__(self, real_label=1.0, fake_label=0.0, with_logits=False):
        super().__init__()
        self.register_buffer('real_label', torch.tensor(real_label))
        self.register_buffer('fake_label', torch.tensor(fake_label))
        if with_logits:
            self.loss = nn.BCEWithLogitsLoss()
        else:
            self.loss = nn.BCELoss()

    def forward(self, verdict, target_is_real):
        if target_is_real:
            target = self.real_label
        else:
            target = self.fake_label
        return self.loss(verdict, target.expand_as(verdict))


class NonSaturatingGANLoss(nn.Module):
    name = 'non-saturating-gan'
    need_lipschitz_d = False

    def __init__(self, real_label=1.0, fake_label=0.0, with_logits=False):
        super().__init__()
        self.register_buffer('real_label', torch.tensor(real_label))
        self.register_buffer('fake_label', torch.tensor(fake_label))
        self.with_logits = with_logits

    def forward(self, verdict, target_is_real):
        p = torch.sigmoid(verdict) if self.with_logits else verdict
        j = torch.log(p + 1e-4)
        return -j if target_is_real else j


class WGANLoss(nn.Module):
    name = 'wgan'
    need_lipschitz_d = True

    def __init__(self, with_logits=True):
        super().__init__()

    def forward(self, verdict, target_is_real):
        return -verdict.mean() if target_is_real else verdict.mean()


class LSGANLoss(nn.Module):
    name = 'lsgan'
    need_lipschitz_d = False

    def __init__(self, real_label=1.0, fake_label=0.0):
        super().__init__()
        self.register_buffer('real_label', torch.tensor(real_label))
        self.register_buffer('fake_label', torch.tensor(fake_label))
        self.loss = nn.MSELoss()

    def forward(self, verdict, target_is_real):
        if target_is_real:
            target = self.real_label
        else:
            target = self.fake_label
        return self.loss(verdict, target.expand_as(verdict))


class GaussianMLELoss(nn.Module):
    name = 'gaussian'

    def __init__(self, order=2, min_noise=0.):
        super().__init__()
        self.order = order
        self.min_noise = min_noise

    def forward(self, center, dispersion, y,
                log_dispersion=True, normalizer=None):
        squared = (center - y) ** 2

        if self.order == 1:
            return squared.mean()

        if log_dispersion:
            var = dispersion.exp()
            log_var = dispersion
        else:
            var = dispersion
            log_var = (dispersion + 1e-9).log()

        loss = ((squared + self.min_noise) / (var + 1e-9) + log_var) * 0.5

        if not normalizer:
            return loss.mean()
        else:
            return loss.sum() / normalizer


class LaplaceMLELoss(nn.Module):
    name = 'laplace'

    def __init__(self, order=2, min_noise=0.):
        super().__init__()
        self.order = order
        self.min_noise = min_noise

    def forward(self, center, dispersion, y,
                log_dispersion=True, normalizer=None):
        deviation = (center - y).abs()

        if self.order == 1:
            return deviation.mean()

        if log_dispersion:
            mad = dispersion.exp()
            log_mad = dispersion
        else:
            mad = dispersion
            log_mad = (dispersion + 1e-9).log()

        loss = (deviation + self.min_noise) / (mad + 1e-9) + log_mad

        if not normalizer:
            return loss.mean()
        else:
            return loss.sum() / normalizer


class MMDLoss(nn.Module):
    name = 'mmd'

    def __init__(self, kbws=None):
        super().__init__()
        self.kbws = (
            torch.Tensor(kbws).cuda() if kbws is not None else
            torch.cuda.ones(1).cuda()
        )

    def forward(self, x, y, sigmas=None, skip_y=True):
        # NOTE: should expand y to x.shape?
        rbf_kernel = partial(self._rbf_kernel_matrix, sigmas=self.kbws)
        return self._mmd(x, y, kernel=rbf_kernel, skip_y=skip_y)

    def _mmd(self, x, y, kernel=None, skip_y=True):
        kernel = kernel or self._rbf_kernel_matrix
        cost = kernel(x, x).mean()
        cost -= 2 * kernel(x, y).mean()
        if not skip_y:
            cost += kernel(y, y).mean()
        return cost

    def _rbf_kernel_matrix(self, x, y, sigmas):
        sigmas = sigmas.view(sigmas.shape[0], 1)
        beta = 1. / (2. * sigmas)
        dist = self._pdist(x, y).contiguous()
        dist_ = dist.view(1, -1)
        s = torch.matmul(beta, dist_)
        return torch.mean(torch.exp(-s), 0).view_as(dist)

    @staticmethod
    def _pdist(x, y):
        x_norm = (x**2).sum(2).view(x.shape[0], x.shape[1], 1)
        y_t = y.permute(0, 2, 1).contiguous()
        y_norm = (y**2).sum(2).view(y.shape[0], 1, y.shape[1])
        dist = x_norm + y_norm - 2.0 * torch.bmm(x, y_t)
        return dist


LOSSES = {
    SaturatingGANLoss.name: SaturatingGANLoss,
    NonSaturatingGANLoss.name: NonSaturatingGANLoss,
    WGANLoss.name: WGANLoss,
    LSGANLoss.name: LSGANLoss,
    GaussianMLELoss.name: GaussianMLELoss,
    LaplaceMLELoss.name: LaplaceMLELoss,
    MMDLoss.name: MMDLoss,
}
