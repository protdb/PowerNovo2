import torch
import torch.nn as nn


def LayerNorm(normalized_shape, eps=1e-5, elementwise_affine=True, export=False):
    if not export and torch.cuda.is_available():
        try:
            from apex.normalization import FusedLayerNorm
            return FusedLayerNorm(normalized_shape, eps, elementwise_affine)
        except ImportError:
            pass
    return nn.LayerNorm(normalized_shape, eps, elementwise_affine)



class LinearWeightNorm(nn.Module):

    def __init__(self, in_features, out_features, bias=True):
        super(LinearWeightNorm, self).__init__()
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.linear.weight, mean=0.0, std=0.05)
        if self.linear.bias is not None:
            nn.init.constant_(self.linear.bias, 0)
        self.linear = nn.utils.weight_norm(self.linear)

    def extra_repr(self):
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None
        )

    def init(self, x, init_scale=1.0):
        with torch.no_grad():

            out = self(x).view(-1, self.linear.out_features)
            mean = out.mean(dim=0)
            std = out.std(dim=0)
            inv_stdv = init_scale / (std + 1e-6)

            self.linear.weight_g.mul_(inv_stdv.unsqueeze(1))
            if self.linear.bias is not None:
                self.linear.bias.add_(-mean).mul_(inv_stdv)
            return self(x)

    def forward(self, input):
        return self.linear(input)
