"""
Glow: Generative Flow with Invertible 1x1 Convolutions
arXiv:1807.03039v2
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as D
import torchvision.transforms as T
from torchvision.utils import save_image, make_grid
from torch.utils.data import DataLoader
from torch.utils.checkpoint import checkpoint


import numpy as np
from tensorboardX import SummaryWriter

import os
import time
import math
import argparse
import pprint

# action
parser = argparse.ArgumentParser()
parser.add_argument('--train', action='store_true', help='Train a flow.')
parser.add_argument('--evaluate', action='store_true', help='Evaluate a flow.')
parser.add_argument('--generate', action='store_true', help='Generate samples from a model.')
parser.add_argument('--visualize', action='store_true', help='Visualize manipulated attribures.')
parser.add_argument('--restore_file', type=str, help='Path to model to restore.')
parser.add_argument('--seed', type=int, help='Random seed to use.')
# paths and reporting
parser.add_argument('--data_dir', default='/mnt/disks/data/', help='Location of datasets.')
parser.add_argument('--output_dir', default='./results/{}'.format(os.path.splitext(__file__)[0]))
parser.add_argument('--results_file', default='results.txt', help='Filename where to store settings and test results.')
parser.add_argument('--log_interval', type=int, default=2, help='How often to show loss statistics and save samples.')
parser.add_argument('--save_interval', type=int, default=50, help='How often to save during training.')
parser.add_argument('--eval_interval', type=int, default=1, help='Number of epochs to eval model and save model checkpoint.')
# data
parser.add_argument('--dataset', type=str, help='Which dataset to use.')
# model parameters
parser.add_argument('--depth', type=int, default=32, help='Depth of the network (cf Glow figure 2).')
parser.add_argument('--n_levels', type=int, default=3, help='Number of levels of of the network (cf Glow figure 2).')
parser.add_argument('--width', type=int, default=512, help='Dimension of the hidden layers.')
parser.add_argument('--z_std', type=float, help='Pass specific standard devition during generation/sampling.')
# training params
parser.add_argument('--batch_size', type=int, default=16, help='Training batch size.')
parser.add_argument('--batch_size_init', type=int, default=256, help='Batch size for the data dependent initialization.')
parser.add_argument('--n_epochs', type=int, default=10, help='Number of epochs to train.')
parser.add_argument('--n_epochs_warmup', type=int, default=2, help='Number of warmup epochs for linear learning rate annealing.')
parser.add_argument('--start_epoch', default=0, help='Starting epoch (for logging; to be overwritten when restoring file.')
parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate.')
parser.add_argument('--mini_data_size', type=int, default=None, help='Train only on this number of datapoints.')
parser.add_argument('--grad_norm_clip', default=50, type=float, help='Clip gradients during training.')
parser.add_argument('--checkpoint_grads', action='store_true', default=False, help='Whether to use gradient checkpointing in forward pass.')
parser.add_argument('--n_bits', default=5, type=int, help='Number of bits for input images.')
# distributed training params
parser.add_argument('--distributed', action='store_true', default=False, help='Whether to use DistributedDataParallels on multiple machines and GPUs.')
parser.add_argument('--world_size', type=int, default=1, help='Number of nodes for distributed training.')
parser.add_argument('--local_rank', type=int, help='When provided, run model on this cuda device. When None, used by torch.distributed.launch utility to manage multi-GPU training.')
# visualize
parser.add_argument('--vis_img', type=str, help='Path to image file to manipulate attributes and visualize.')
parser.add_argument('--vis_attrs', nargs='+', type=int, help='Which attribute to manipulate.')
parser.add_argument('--vis_alphas', nargs='+', type=float, help='Step size on the manipulation direction.')


best_eval_logprob = float('-inf')


# --------------------
# Data
# --------------------

def fetch_dataloader(args, train=True, data_dependent_init=False):
    args.input_dims = {'mnist': (3,32,32), 'celeba': (3,64,64)}[args.dataset]

    transforms = {'mnist': T.Compose([T.Pad(2),                                         # image to 32x32 same as CIFAR
                                      T.RandomAffine(degrees=0, translate=(0.1, 0.1)),  # random shifts to fill the padded pixels
                                      T.ToTensor(),
                                      T.Lambda(lambda t: t + torch.rand_like(t)/2**8),  # dequantize
                                      T.Lambda(lambda t: t.expand(3,-1,-1))]),          # expand to 3 channels

                  'celeba': T.Compose([T.CenterCrop(148),  # RealNVP preprocessing
                                       T.Resize(64),
                                       T.Lambda(lambda im: np.array(im, dtype=np.float32)),                     # to numpy
                                       T.Lambda(lambda x: np.floor(x / 2**(8 - args.n_bits)) / 2**args.n_bits), # lower bits
                                       T.ToTensor(),  # note: if input to this transform is uint8, it divides by 255 and returns float
                                       T.Lambda(lambda t: t + torch.rand_like(t) / 2**args.n_bits)])            # dequantize
                  }[args.dataset]

    dataset = {'mnist': MNIST, 'celeba': CelebA}[args.dataset]

    # load the specific dataset
    dataset = dataset(root=args.data_dir, train=train, transform=transforms)

    if args.mini_data_size:
        dataset.data = dataset.data[:args.mini_data_size]

    # load sampler and dataloader
    if args.distributed and train is True and not data_dependent_init:  # distributed training; but exclude initialization
        sampler = torch.utils.data.distributed.DistributedSampler(dataset)
    else:
        sampler = None

    batch_size = args.batch_size_init if data_dependent_init else args.batch_size  # if data dependent init use init batch size
    kwargs = {'num_workers': 1, 'pin_memory': True} if args.device.type is 'cuda' else {}
    return DataLoader(dataset, batch_size=batch_size, shuffle=(sampler is None), drop_last=True, sampler=sampler, **kwargs)


# --------------------
# Model component layers
# --------------------

class Actnorm(nn.Module):
    """ Actnorm layer; cf Glow section 3.1 """
    def __init__(self, param_dim=(1,3,1,1)):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(param_dim))
        self.bias = nn.Parameter(torch.zeros(param_dim))
        self.register_buffer('initialized', torch.tensor(0).byte())

    def forward(self, x):
        if not self.initialized:
            # per channel mean and variance where x.shape = (B, C, H, W)
            self.bias.squeeze().data.copy_(x.transpose(0,1).flatten(1).mean(1)).view_as(self.scale)
            self.scale.squeeze().data.copy_(x.transpose(0,1).flatten(1).std(1, False) + 1e-6).view_as(self.bias)
            self.initialized += 1

        z = (x - self.bias) / self.scale
        logdet = - self.scale.abs().log().sum() * x.shape[2] * x.shape[3]
        return z, logdet

    def inverse(self, z):
        return z * self.scale + self.bias, self.scale.abs().log().sum() * z.shape[2] * z.shape[3]


class Invertible1x1Conv(nn.Module):
    """ Invertible 1x1 convolution layer; cf Glow section 3.2 """
    def __init__(self, n_channels=3, lu_factorize=False):
        super().__init__()
        self.lu_factorize = lu_factorize

        # initiaize a 1x1 convolution weight matrix
        w = torch.randn(n_channels, n_channels)
        w = torch.qr(w)[0]  # note: nn.init.orthogonal_ returns orth matrices with dets +/- 1 which complicates the inverse call below

        if lu_factorize:
            # compute LU factorization
            p, l, u = torch.btriunpack(*w.unsqueeze(0).btrifact())
            # initialize model parameters
            self.p, self.l, self.u = nn.Parameter(p.squeeze()), nn.Parameter(l.squeeze()), nn.Parameter(u.squeeze())
            s = self.u.diag()
            self.log_s = nn.Parameter(s.abs().log())
            self.register_buffer('sign_s', s.sign())  # note: not optimizing the sign; det W remains the same sign
            self.register_buffer('l_mask', torch.tril(torch.ones_like(self.l), -1))  # store mask to compute LU in forward/inverse pass
        else:
            self.w = nn.Parameter(w)

    def forward(self, x):
        B,C,H,W = x.shape
        if self.lu_factorize:
            l = self.l * self.l_mask + torch.eye(C).to(self.l.device)
            u = self.u * self.l_mask.t() + torch.diag(self.sign_s * self.log_s.exp())
            self.w = self.p @ l @ u
            logdet = self.log_s.sum() * H * W
        else:
            logdet = torch.slogdet(self.w)[-1] * H * W

        return F.conv2d(x, self.w.view(C,C,1,1)), logdet

    def inverse(self, z):
        B,C,H,W = z.shape
        if self.lu_factorize:
            l = torch.inverse(self.l * self.l_mask + torch.eye(C).to(self.l.device))
            u = torch.inverse(self.u * self.l_mask.t() + torch.diag(self.sign_s * self.log_s.exp()))
            w_inv = u @ l @ self.p.inverse()
            logdet = - self.log_s.sum() * H * W
        else:
            w_inv = self.w.inverse()
            logdet = - torch.slogdet(self.w)[-1] * H * W

        return F.conv2d(z, w_inv.view(C,C,1,1)), logdet


class AffineCoupling(nn.Module):
    """ Affine coupling layer; cf Glow section 3.3; RealNVP figure 2 """
    def __init__(self, n_channels, width):
        super().__init__()
        # network layers;
        # per realnvp, network splits input, operates on half of it, and returns shift and scale of dim = half the input channels
        self.conv1 = nn.Conv2d(n_channels//2, width, kernel_size=3, padding=1, bias=False)  # input is split along channel dim
        self.actnorm1 = Actnorm(param_dim=(1, width, 1, 1))
        self.conv2 = nn.Conv2d(width, width, kernel_size=1, padding=1, bias=False)
        self.actnorm2 = Actnorm(param_dim=(1, width, 1, 1))
        self.conv3 = nn.Conv2d(width, n_channels, kernel_size=3)            # output is split into scale and shift components
        self.log_scale_factor = nn.Parameter(torch.zeros(n_channels,1,1))   # learned scale (cf RealNVP sec 4.1 / Glow official code

        # initialize last convolution with zeros, such that each affine coupling layer performs an identity function
        self.conv3.weight.data.zero_()
        self.conv3.bias.data.zero_()

    def forward(self, x):
        x_a, x_b = x.chunk(2, 1)  # split along channel dim

        h = F.relu(self.actnorm1(self.conv1(x_b))[0])
        h = F.relu(self.actnorm2(self.conv2(h))[0])
        h = self.conv3(h) * self.log_scale_factor.exp()
        t = h[:,0::2,:,:]  # shift; take even channels
        s = h[:,1::2,:,:]  # scale; take odd channels
        s = torch.sigmoid(s + 2.)  # at initalization, s is 0 and sigmoid(2) is near identity

        z_a = s * x_a + t
        z_b = x_b
        z = torch.cat([z_a, z_b], dim=1)  # concat along channel dim

        logdet = s.log().sum([1, 2, 3])

        return z, logdet

    def inverse(self, z):
        z_a, z_b = z.chunk(2, 1)  # split along channel dim

        h = F.relu(self.actnorm1(self.conv1(z_b))[0])
        h = F.relu(self.actnorm2(self.conv2(h))[0])
        h = self.conv3(h)  * self.log_scale_factor.exp()
        t = h[:,0::2,:,:]  # shift; take even channels
        s = h[:,1::2,:,:]  # scale; take odd channels
        s = torch.sigmoid(s + 2.)

        x_a = (z_a - t) / s
        x_b = z_b
        x = torch.cat([x_a, x_b], dim=1)  # concat along channel dim

        logdet = - s.log().sum([1, 2, 3])

        return x, logdet


class Squeeze(nn.Module):
    """ RealNVP squeezing operation layer (cf RealNVP section 3.6; Glow figure 2b):
    For each channel, it divides the image into subsquares of shape 2 × 2 × c, then reshapes them into subsquares of
    shape 1 × 1 × 4c. The squeezing operation transforms an s × s × c tensor into an s × s × 4c tensor """
    def __init__(self):
        super().__init__()

    def forward(self, x):
        B,C,H,W = x.shape
        x = x.reshape(B, C, H//2, 2, W//2, 2)   # factor spatial dim
        x = x.permute(0, 1, 3, 5, 2, 4)         # transpose to (B, C, 2, 2, H//2, W//2)
        x = x.reshape(B, 4*C, H//2, W//2)       # aggregate spatial dim factors into channels
        return x

    def inverse(self, x):
        B,C,H,W = x.shape
        x = x.reshape(B, C//4, 2, 2, H, W)      # factor channel dim
        x = x.permute(0, 1, 4, 2, 5, 3)         # transpose to (B, C//4, H, 2, W, 2)
        x = x.reshape(B, C//4, 2*H, 2*W)        # aggregate channel dim factors into spatial dims
        return x


class Split(nn.Module):
    """ Split layer; cf Glow figure 2 / RealNVP figure 4b
    Based on RealNVP multi-scale architecture: splits an input in half along the channel dim; half the vars are
    directly modeled as Gaussians while the other half undergo further transformations (cf RealNVP figure 4b).
    """
    def __init__(self, n_channels):
        super().__init__()
        self.gaussianize = Gaussianize(n_channels//2)

    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)  # split input along channel dim
        z2, logdet = self.gaussianize(x1, x2)
        return x1, z2, logdet

    def inverse(self, x1, z2):
        x2, logdet = self.gaussianize.inverse(x1, z2)
        x = torch.cat([x1, x2], dim=1)  # cat along channel dim
        return x, logdet


class Gaussianize(nn.Module):
    """ Gaussianization per ReanNVP sec 3.6 / fig 4b -- at each step half the variables are directly modeled as Gaussians.
    Model as Gaussians:
        x2 = z2 * exp(logs) + mu, so x2 ~ N(mu, exp(logs)^2) where mu, logs = f(x1)
    then to recover the random numbers z driving the model:
        z2 = (x2 - mu) * exp(-logs)
    Here f(x1) is a conv layer initialized to identity.
    """
    def __init__(self, n_channels):
        super().__init__()
        self.net = nn.Conv2d(n_channels, 2*n_channels, kernel_size=3, padding=1)  # computes the parameters of Gaussian
        self.log_scale_factor = nn.Parameter(torch.zeros(2*n_channels,1,1))       # learned scale (cf RealNVP sec 4.1 / Glow official code
        # initialize to identity
        self.net.weight.data.zero_()
        self.net.bias.data.zero_()

    def forward(self, x1, x2):
        h = self.net(x1) * self.log_scale_factor.exp()  # use x1 to model x2 as Gaussians; learnable scale
        m, logs = h[:,0::2,:,:], h[:,1::2,:,:]          # split along channel dims
        z2 = (x2 - m) * torch.exp(-logs)                # center and scale; log prob is computed at the model forward
        logdet = - logs.sum([1,2,3])
        return z2, logdet

    def inverse(self, x1, z2):
        h = self.net(x1) * self.log_scale_factor.exp()
        m, logs = h[:,0::2,:,:], h[:,1::2,:,:]
        x2 = m + z2 * torch.exp(logs)
        logdet = logs.sum([1,2,3])
        return x2, logdet


class Preprocess(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        logdet = - math.log(256) * x[0].numel() # processing each image dim from [0, 255] to [0,1]; per RealNVP sec 4.1 taken into account
        return x - 0.5, logdet                  # center x at 0

    def inverse(self, x):
        logdet = math.log(256) * x[0].numel()
        return x + 0.5, logdet

# --------------------
# Container layers
# --------------------

class FlowSequential(nn.Sequential):
    """ Container for layers of a normalizing flow """
    def __init__(self, *args, **kwargs):
        self.checkpoint_grads = kwargs.pop('checkpoint_grads', None)
        super().__init__(*args, **kwargs)

    def forward(self, x):
        sum_logdets = 0.
        for module in self:
            x, logdet = module(x) if not self.checkpoint_grads else checkpoint(module, x)
            sum_logdets = sum_logdets + logdet
        return x, sum_logdets

    def inverse(self, z):
        sum_logdets = 0.
        for module in reversed(self):
            z, logdet = module.inverse(z)
            sum_logdets = sum_logdets + logdet
        return z, sum_logdets


class FlowStep(FlowSequential):
    """ One step of Glow flow (Actnorm -> Invertible 1x1 conv -> Affine coupling); cf Glow Figure 2a """
    def __init__(self, n_channels, width, lu_factorize=False):
        super().__init__(Actnorm(param_dim=(1,n_channels,1,1)),
                         Invertible1x1Conv(n_channels, lu_factorize),
                         AffineCoupling(n_channels, width))


class FlowLevel(nn.Module):
    """ One depth level of Glow flow (Squeeze -> FlowStep x K -> Split); cf Glow figure 2b """
    def __init__(self, n_channels, width, depth, checkpoint_grads=False, lu_factorize=False):
        super().__init__()
        # network layers
        self.squeeze = Squeeze()
        self.flowsteps = FlowSequential(*[FlowStep(4*n_channels, width, lu_factorize) for _ in range(depth)], checkpoint_grads=checkpoint_grads)
        self.split = Split(4*n_channels)

    def forward(self, x):
        x = self.squeeze(x)
        x, logdet_flowsteps = self.flowsteps(x)
        x1, z2, logdet_split = self.split(x)
        logdet = logdet_flowsteps + logdet_split
        return x1, z2, logdet

    def inverse(self, x1, z2):
        x, logdet_split = self.split.inverse(x1, z2)
        x, logdet_flowsteps = self.flowsteps.inverse(x)
        x = self.squeeze.inverse(x)
        logdet = logdet_flowsteps + logdet_split
        return x, logdet


# --------------------
# Model
# --------------------

class Glow(nn.Module):
    """ Glow multi-scale architecture with depth of flow K and number of levels L; cf Glow figure 2; section 3"""
    def __init__(self, width, depth, n_levels, input_dims=(3,32,32), checkpoint_grads=False, lu_factorize=False):
        super().__init__()
        # calculate output dims
        in_channels, H, W = input_dims
        out_channels = int(in_channels * 4**(n_levels+1) / 2**n_levels)  # each Squeeze results in 4x in_channels (cf RealNVP section 3.6); each Split in 1/2x in_channels
        out_HW = int(H / 2**(n_levels+1))                                # each Squeeze is 1/2x HW dim (cf RealNVP section 3.6)
        self.output_dims = out_channels, out_HW, out_HW

        # preprocess images
        self.preprocess = Preprocess()

        # network layers cf Glow figure 2b: (Squeeze -> FlowStep x depth -> Split) x n_levels -> Squeeze -> FlowStep x depth
        self.flowlevels = nn.ModuleList([FlowLevel(in_channels * 2**i, width, depth, checkpoint_grads, lu_factorize) for i in range(n_levels)])
        self.squeeze = Squeeze()
        self.flowstep = FlowSequential(*[FlowStep(out_channels, width, lu_factorize) for _ in range(depth)], checkpoint_grads=checkpoint_grads)

        # gaussianize the final z output; initialize to identity
        self.gaussianize = Gaussianize(out_channels)

        # base distribution of the flow
        self.register_buffer('base_dist_mean', torch.zeros(1))
        self.register_buffer('base_dist_var', torch.ones(1))

    def forward(self, x):
        x, sum_logdets = self.preprocess(x)
        # pass through flow
        zs = []
        for m in self.flowlevels:
            x, z, logdet = m(x)
            sum_logdets = sum_logdets + logdet
            zs.append(z)
        x = self.squeeze(x)
        z, logdet = self.flowstep(x)
        sum_logdets = sum_logdets + logdet

        # gaussianize the final z
        z, logdet = self.gaussianize(torch.zeros_like(z), z)
        sum_logdets = sum_logdets + logdet
        zs.append(z)
        return zs, sum_logdets

    def inverse(self, zs=None, batch_size=None, z_std=1.):
        if zs is None:  # if no random numbers are passed, generate new from the base distribution
            assert batch_size is not None, 'Must either specify batch_size or pass a batch of z random numbers.'
            zs = [z_std * self.base_dist.sample((batch_size, *self.output_dims)).squeeze()]
        # pass through inverse flow
        z, sum_logdets = self.gaussianize.inverse(torch.zeros_like(zs[-1]), zs[-1])
        x, logdet = self.flowstep.inverse(z)
        sum_logdets = sum_logdets + logdet
        x = self.squeeze.inverse(x)
        for i, m in enumerate(reversed(self.flowlevels)):
            z = z_std * (self.base_dist.sample(x.shape).squeeze() if len(zs)==1 else zs[-i-2])  # if no z's are passed, generate new random numbers from the base dist
            x, logdet = m.inverse(x, z)
            sum_logdets = sum_logdets + logdet
        # postprocess
        x, logdet = self.preprocess.inverse(x)
        sum_logdets = sum_logdets + logdet
        return x, sum_logdets

    @property
    def base_dist(self):
        return D.Normal(self.base_dist_mean, self.base_dist_var)

    def log_prob(self, x, bits_per_pixel=False):
        zs, logdet = self.forward(x)
        log_prob = sum(self.base_dist.log_prob(z).sum([1,2,3]) for z in zs) + logdet
        if bits_per_pixel:
            log_prob /= (math.log(2) * x[0].numel())
        return log_prob
