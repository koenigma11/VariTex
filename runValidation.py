
import os
import sys
import warnings
warnings.filterwarnings('ignore')
import pytorch_lightning as pl
import torch
import numpy as np
from varitex.data.keys_enum import DataItemKey as DIK
pl.seed_everything(1234)
from mutil.object_dict import ObjectDict
from torch.nn.functional import normalize as normalizeT
from varitex.evaluation.inference import inference_ffhq
from varitex import validation
import sys
import json
sys.path.append('/home/matthias/ETH/Thesis/nFlows/nflows')
#### Nflows
from nflows.flows.base import Flow
from nflows.flows.realnvp import SimpleRealNVP
from nflows.distributions.normal import StandardNormal
from nflows.transforms.coupling import CouplingTransform, AffineCouplingTransform, AdditiveCouplingTransform
from nflows.flows import realnvp
sys.path.append('/home/matthias/ETH/Thesis/nFlows/pytorch-normalizing-flows/')

#### NfLib
# from nflib.flows import (
#     AffineConstantFlow, ActNorm, AffineHalfFlow,
#     SlowMAF, MAF, IAF, Invertible1x1Conv,
#     NormalizingFlow, NormalizingFlowModel,
# )
# from nflib.spline_flows import NSF_AR, NSF_CL




from torch import distributions
from torch.distributions import MultivariateNormal, Uniform, TransformedDistribution, SigmoidTransform
import itertools

from nflows.flows.base import Flow
from nflows.flows.realnvp import SimpleRealNVP
from nflows.flows.autoregressive import MaskedAutoregressiveFlow
from nflows.distributions.normal import StandardNormal
from nflows.transforms.base import CompositeTransform
from nflows.transforms.autoregressive import MaskedAffineAutoregressiveTransform, MaskedUMNNAutoregressiveTransform, MaskedPiecewiseLinearAutoregressiveTransform,MaskedPiecewiseQuadraticAutoregressiveTransform
from nflows.transforms.permutations import ReversePermutation, RandomPermutation
from nflows.transforms.nonlinearities import LeakyReLU
from nflows.transforms.normalization import ActNorm, BatchNorm
from nflows.transforms.coupling import CouplingTransform, AffineCouplingTransform, AdditiveCouplingTransform
from nflows.transforms.standard import PointwiseAffineTransform

import matplotlib.pyplot as plt



#def do_validation(model_name, opt):
ckptFolder ='/home/matthias/ETH/Thesis/VariTexLocal/output/ckpts/Res128_GLO_NoNorm'
filename = 'epoch=43-step=377079.ckpt'
Res128_nonorm_path = os.path.join(ckptFolder, filename)



def getModel(modelName, opt, test):
    mp_path = '/cluster/project/infk/hilliges/koenigma/Final_Models'
    if(test):
        mp_path = '/home/matthias/ETH/Thesis/Final_Models'
    if (modelName == 'default'):
        folder = os.path.join(mp_path, 'default/checkpoints')
        opt.update({"checkpoint": os.path.join(folder, 'epoch=43-step=377079.ckpt'),
                    "use_glo": False,
                    "use_NF": False,
                    "alternate": False,
                    "experiment_name": "eval_Default"})
    elif (modelName == 'norm'):
        folder = os.path.join(mp_path, 'GLO_Norm/checkpoints')
        opt.update({"checkpoint": os.path.join(folder, 'epoch=43-step=377079.ckpt'),
                    "use_glo": True,
                    "use_NF": False,
                    "alternate": False,
                    "experiment_name": "eval_norm"})
    elif (modelName == 'nonorm'):
        folder = os.path.join(mp_path, 'GLO_NoNorm/checkpoints')
        opt.update({"checkpoint": os.path.join(folder, 'epoch=43-step=377079.ckpt'),
                    "use_glo": True,
                    "use_NF": False,
                    "alternate": False,
                    "experiment_name": "eval_nonorm"})
    elif (modelName == 'nf_glo_alternate'):
        folder = os.path.join(mp_path, 'GLO_NF_ALTERNATE')
        opt.update({"checkpoint": os.path.join(folder, 'epoch=35-step=308519.ckpt'),
                    "use_glo": True,
                    "use_NF": True,
                    "alternate": True,
                    "experiment_name": "eval_nf_glo_alternate"})
    elif (modelName == 'nf_glo_joint'):
        folder = os.path.join(mp_path, 'GLO_NF_joint/Res128_GLO_NF_lambdae-3')
        opt.update({"checkpoint": os.path.join(folder, 'epoch=43-step=377079.ckpt'),
                    "use_glo": True,
                    "use_NF": True,
                    "alternate": False,
                    "experiment_name": "eval_nf_glo_joint"})
    else:
        print('Model not recognized!')
    return opt


def getStandards(valModel):
        print("Validating standard metrics...")
        valModel.inference_ffhq('standards', n=60000)
        psnr = np.array(vals.psnr).mean()
        ssim = np.array(vals.ssim).mean()
        lpips = np.array(vals.lpips).mean()
        valDict = {'PSNR': psnr,
                    'SSIM': ssim,
                   'LPIPS': lpips}
        writetoFile(valDict, 'standards.json')

def getFID(valModel, test=True, interpolated=None, shape='constant', sampling=None):
        filename = 'FID_'+shape+'_interpolated_'+str(interpolated)+'_sampling_'+str(sampling)+'.json'
        print(filename)
        n = 10000
        if(test):
            n=5
        valModel.inference_ffhq('fid',n=n,interpolated=interpolated, shape=shape, sampling=sampling)
        fid = vals.fid_std
        valDict = {'FID': fid}
        writetoFile(valDict, filename, test)

def writetoFile(valDict, metric, test):
        path = os.path.dirname(opt_new.checkpoint)
        JSON = json.dumps(valDict)
        # open file for writing, "w"
        filename = metric
        f = open(os.path.join(path, filename), "w")

        # write json object to file
        if(not test):
            f.write(JSON)
def getOpt():
    opt = {

        # "checkpoint": os.path.join(os.getenv("CP", "pretrained/ep44.ckpt")),
        # "checkpoint": Res128_nonorm_path,
        "dataroot_npy": os.path.join(os.getenv("DP", ""), 'FFHQ/preprocessed_dataset_new'),
        "path_bfm": os.path.join(os.getenv("FP", ""), "basel_facemodel/model2017-1_face12_nomouth.h5"),
        "path_uv": os.path.join(os.getenv("FP", ""), "basel_facemodel/face12.json"),
        "device": "cuda"
    }

    opt.update({"dataset": 'FFHQ',
                "dataroot_npy": os.path.join(os.getenv("DP"), 'FFHQ/preprocessed_dataset_new'),
                "image_folder": os.path.join(os.getenv("DP"), 'FFHQ/images'),
                "transform_mode": 'all',
                "image_h": 128,
                "image_w": 128,
                "batch_size": 1,
                "num_workers": 12,
                "semantic_regions": list(range(1, 16)),
                "keep_background": False,
                "bg_color": 'black',
                "latent_dim": 256,
                "texture_dim": 128,
                "texture_nc": 16,
                "nc_feature2image": 64,
                "feature2image_num_layers": 5,
                "use_glo": False,
                "glo_init": 'pca',
                "pca_file": os.path.join(os.getenv("BP"), 'datasets/pcaLatents.npy'),
                "experiment_name": 'eval_NoNorm',
                "use_NF": False,
                "eval": True})
    return opt
if __name__ == "__main__":
    model_names=['default', 'norm', 'nonorm', 'nf_glo_alternate', 'nf_glo_joint']
    model_names=['default', 'nf_glo_alternate', 'nf_glo_joint']
    """Run 1: Joint: [const, sampled][linear, spherical][normal]"""
    """Run 1: NoNorm: [const, sampled] [linear, spherical] [latent] ++ [const/sampled, sph, sampled]"""
    test = False
    if(test):
        print("Testing Run of validation")
    model_names=['default', 'nf_glo_joint']
    model_names=['nf_glo_joint']
    #model_names=['norm']
    for modelName in model_names:
            print("Validating Model "+ modelName + '...')
            opt = getOpt()
            opt = getModel(modelName,opt, test)
            opt_new = ObjectDict(opt)
            vals = validation.Validation(opt)
            #getStandards(vals)
            shapes = ['constant', 'sampled']
            shape = 'sampled'
            interpolateds = ['linear', 'spherical']
            samplings = ['latent', 'sampled']
            samplings = ['latent']
            for sampling in samplings:
                for interpolated in interpolateds:
                    pass
                    # getFID(vals, test= test, interpolated=interpolated, shape=shape, sampling= sampling)
            getFID(vals, test=test, interpolated='spherical', shape='sampled', sampling='sampled')
    #out = vals.sample()
