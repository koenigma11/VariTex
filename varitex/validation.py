import pdb

import imageio
import torch
import numpy as np
from mutil.files import mkdir
from tqdm import tqdm
from torch.utils.data import DataLoader
from mutil.pytorch_utils import ImageNetNormalizeTransformInverse, to_tensor, theta2rotation_matrix
from nflows.flows.base import Flow
from nflows.distributions.normal import StandardNormal
from nflows.transforms.base import CompositeTransform
from nflows.transforms.autoregressive import MaskedAffineAutoregressiveTransform
from nflows.transforms.permutations import ReversePermutation


try:
    from mutil.object_dict import ObjectDict
    from varitex.data.keys_enum import DataItemKey as DIK
    from varitex.data.uv_factory import BFMUVFactory
    from varitex.data.npy_dataset import NPYDataset
    from varitex.modules.pipeline import PipelineModule
    from varitex.visualization.batch import Visualizer
    from varitex.visualization.batch import CompleteVisualizer
    from varitex.options import varitex_default_options, varitex_glo_options
    from varitex.modules.metrics import PSNR, SSIM, LPIPS, FID
except ModuleNotFoundError:
    print("Have you added VariTex to your pythonpath?")
    print('To fix this error, go to the root path of the repository ".../VariTex/" \n '
          'and run \n'
          "export PYTHONPATH=$PYTHONPATH:$(pwd)")
    exit()

class Validation:
    def __init__(self, opt):
        #default_opt = varitex_default_options()
        default_opt = varitex_glo_options()
        default_opt.update(opt)
        self.opt = ObjectDict(default_opt)
        self.model = self.get_model(self.opt)

        self.dataset = NPYDataset(self.opt, augmentation=False, split='train')
        self.dataloader = DataLoader(self.dataset, batch_size=self.opt.batch_size, num_workers=self.opt.num_workers, shuffle=False, drop_last=True)
        self.datasetVal = NPYDataset(self.opt, augmentation=False, split='val')
        self.dataloaderVal = DataLoader(self.datasetVal, batch_size=self.opt.batch_size, num_workers=self.opt.num_workers, shuffle=False, drop_last=True)
        print("Datasets Loaded")
        self.device = self.model.device
        self.metric_psnr = PSNR()
        self.metric_ssim = SSIM()
        self.metric_lpips = LPIPS()

        self.metric_fid_std = FID()
        self.metric_fid_interpolated = FID()

        self.psnr = []
        self.ssim = []
        self.lpips = []
        self.fid_std = 0

        shapes, expressions = self.load_shape_expressions()
        index_id, index_sp, index_ep = 0, 0, 0
        self.sp =  torch.zeros((1,199)).to(self.device)
        self.ep = torch.zeros((1,100)).to(self.device)#expressions[index_ep].unsqueeze(0)
        self.theta = torch.Tensor((0, 0, 0)).to(self.device)
        self.t = torch.Tensor([0, -2, -57]).to(self.device)
        self.uv_factory_bfm = BFMUVFactory(opt=self.opt, use_bfm_gpu=self.opt.device == 'cuda')
        self.visualizer_complete = CompleteVisualizer(opt=self.opt, bfm_uv_factory=self.uv_factory_bfm)
        self.flow=None
        base_dist = StandardNormal(shape=[256])
        transforms = []
        for _ in range(4):
            transforms.append(ReversePermutation(features=256))
            transforms.append(MaskedAffineAutoregressiveTransform(features=256,
                                                                  hidden_features=256))
        transform = CompositeTransform(transforms)
        # if(self.opt.experiment_name=='eval_norm'):
        #     currentPath = '/home/matthias/ETH/Thesis/Final_Models/GLO_Norm/checkpoints/final_norm.ckpt'
        #     currentDicts = torch.load(currentPath)
        #     self.flow = Flow(transform, base_dist)
        #     self.flow.load_state_dict(currentDicts['model_state_dict'])
        # elif(self.opt.experiment_name=='eval_nonorm'):
        #     currentPath = '/home/matthias/ETH/Thesis/Final_Models/GLO_NoNorm/checkpoints/final_nonorm.ckpt'
        #     currentDicts = torch.load(currentPath)
        #     self.flow = Flow(transform, base_dist)
        #     self.flow.load_state_dict(currentDicts['model_state_dict'])



    def inference_ffhq(self, metric, n=3000, interpolated=None, shape=None, sampling=None):
        """
        Runs inference on FFHQ. Save the resulting images in the results_folder.
        Also saves the latent codes and distributions.
        """
        print(
            "Running inference on FFHQ. Using the extracted face model parameters and poses, and predicted latent codes.")

        self.metric_fid_std = FID()
        self.metric_fid_interpolated = FID()
        self.psnr = []
        self.ssim = []
        self.lpips = []
        self.fid_std = 0
        for i, batch in tqdm(enumerate(self.dataloader)):
            if(not(shape ==  None)):
                batch = self.getBatch(batch, mode=shape)
                if(not(interpolated ==  None)):
                    batch = self.interpolate(batch, interpolationMode=interpolated, sampling=sampling)
            if(metric=='fid' and self.opt.experiment_name=='eval_Default' and not(interpolated ==  None)):
                batch = self.model.forward_latent2image(batch, 0)
            else:
                batch = self.model.forward(batch, i, std_multiplier=0)
            fake = batch[DIK.IMAGE_OUT]
            real = batch[DIK.IMAGE_IN]

            if(metric=='standards'):
                """Standard Metrics"""
                psnr, ssim, lpips = self.standardVals(real, fake)
                self.psnr.append(psnr.item())
                self.ssim.append(ssim.item())
                self.lpips.append(lpips.item())
            elif(metric=='fid'):
                """FID"""
                self.metric_fid_std.fid.update(real.to(torch.uint8).cpu(), real=True)
                self.metric_fid_std.fid.update(fake.to(torch.uint8).cpu(), real=False)

            """PPL"""

            if(i*self.opt.batch_size>n):
                break
        if(metric=='fid'):
            self.fid_std = self.metric_fid_std.fid.compute().item()

    def get_model(self, opt):
        model = PipelineModule.load_from_checkpoint(opt.checkpoint, opt=opt, strict=False)
        model = model.eval()
        model = model.cuda()
        return model

    def getBatch(self, batch, mode=None):
        if(mode == 'constant'):
            self.sp = torch.zeros((1, 199)).to(self.device)
            self.ep = torch.zeros((1, 100)).to(self.device)  # expressions[index_ep].unsqueeze(0)
            self.theta = torch.Tensor((0, 0, 0))
            R = theta2rotation_matrix(theta_all=self.theta).to(self.device)
            batch[DIK.COEFF_SHAPE] = self.sp
            batch[DIK.COEFF_EXPRESSION] = self.ep
            batch[DIK.T] = self.t
            batch[DIK.R] = R.unsqueeze(0).expand(self.opt.batch_size,-1,-1)
            uv = self.uv_factory_bfm.getUV(sp=batch[DIK.COEFF_SHAPE], ep=batch[DIK.COEFF_EXPRESSION], R=R,
                                           t=batch[DIK.T],
                                           correct_translation=True)
            batch[DIK.UV_RENDERED] = uv.expand(self.opt.batch_size, -1, -1, -1).to(self.device)
        elif(mode == "sampled"):
            idx = np.random.randint(0,self.datasetVal.N,(1,1)).squeeze()
            batch2 = self.dataloaderVal.dataset.get_unsqueezed(idx)
            batch[DIK.COEFF_SHAPE] = batch2[DIK.COEFF_SHAPE]
            batch[DIK.COEFF_EXPRESSION] = batch2[DIK.COEFF_EXPRESSION]
            batch[DIK.T] = self.t
            batch[DIK.UV_RENDERED] = batch2[DIK.UV_RENDERED]
        return batch
    def interpolate(self,batch, interpolationMode='linear', sampling='latent',shape='constant'):
        batch1 = self.sample(sampling=sampling, shape=shape)
        batch2 = self.sample(sampling=sampling, shape=shape)
        t = torch.rand(1).to(self.device)
        if(interpolationMode=='linear'):
            batch[DIK.STYLE_LATENT] = torch.lerp(batch1[DIK.STYLE_LATENT],batch2[DIK.STYLE_LATENT],t)
        else:
            batch[DIK.STYLE_LATENT] = self.slerp(batch1[DIK.STYLE_LATENT],batch2[DIK.STYLE_LATENT],t)
        return batch

    def sample(self, sampling='latent', shape='constant'):
        """mode in [latent, ]"""
        modelName = self.opt.experiment_name
        if(modelName == 'eval_Default'):
            if(sampling == 'latent'):
                idx = np.random.randint(0,self.dataset.N-2,(1,1)).squeeze()
                batch = self.dataloader.dataset.get_unsqueezed(idx)
                batch = self.model.generator.forward_encode(batch, 0)  # Only encoding, not yet a distribution
                batch = self.model.generator.forward_encoded2latent_distribution(batch)  # Compute mu and std
                q = torch.distributions.Normal(batch[DIK.STYLE_LATENT_MU], batch[DIK.STYLE_LATENT_STD] * 1)
                z = q.rsample()
                batch[DIK.STYLE_LATENT] = z
            else:
                ## normal
                batch = {}
                batch[DIK.STYLE_LATENT] = torch.randn(size=(1,256)).to(self.device)
        elif(modelName == 'eval_norm'):
            if(sampling == 'latent'):
                idx = np.random.randint(0,self.dataset.N-2,(1,1)).squeeze()
                batch = self.dataloader.dataset.get_unsqueezed(idx)
                batch[DIK.STYLE_LATENT] = self.model.Z.weight[idx]
            else:
                batch = {}
                batch[DIK.STYLE_LATENT] = self.flow.sample(1).to(self.device)
        elif(modelName == 'eval_nonorm'):
            if(sampling == 'latent'):
                idx = np.random.randint(0,self.dataset.N-2,(1,1)).squeeze()
                batch = self.dataloader.dataset.get_unsqueezed(idx)
                batch[DIK.STYLE_LATENT] = self.model.Z.weight[idx]
            else:
                batch = {}
                batch[DIK.STYLE_LATENT] = self.flow.sample(1).to(self.device)
        elif(modelName == 'eval_nf_glo_alternate'):
            if(sampling == 'latent'):
                idx = np.random.randint(0,self.dataset.N-2,(1,1)).squeeze()
                batch = self.dataloader.dataset.get_unsqueezed(idx)
                batch[DIK.STYLE_LATENT] = self.model.Z.weight[idx]
            else:
                batch = {}
                batch[DIK.STYLE_LATENT] = self.model.flow.sample(1).to(self.device)
        elif(modelName == 'eval_nf_glo_joint'):
            if(sampling == 'latent'):
                idx = np.random.randint(0,self.dataset.N-2,(1,1)).squeeze()
                batch = self.dataloader.dataset.get_unsqueezed(idx)
                batch[DIK.STYLE_LATENT] = self.model.Z.weight[idx]
            else:
                batch = {}
                batch[DIK.STYLE_LATENT] = self.model.flow.sample(1).to(self.device)
        batch = self.getBatch(batch, mode = shape)
        return batch

    def sampleInterpolated(self, num_samples):
        # latent1 =
        # latent2 =
        # t = torch.rand(num_samples)
        # out = std1.lerp(std2, t)
        # imgs = sample()
        return None

    def to_image_tensor(self, batch_or_batch_list):
        if isinstance(batch_or_batch_list, list):
            out = torch.cat([batch_out[DIK.IMAGE_OUT][0] for batch_out in batch_or_batch_list], -1)
        elif isinstance(batch_or_batch_list, dict):
            out = batch_or_batch_list[DIK.IMAGE_OUT][0]
        return out

    def sampleFID(self, real, fake, mode):
        #assert mode in ['standard','interpolated']
        #If standard then just use all images, shuffle, compare to ground truth images return fid
        # true = true_images
        # fake = self.sampleInterpolated();
        # fids = FID(true, fake)
        return None# fids

    def standardVals(self, real, fake):
        real = real.detach().cpu()
        fake = fake.detach().cpu()
        psnr = self.metric_psnr(fake, real)
        ssim = self.metric_ssim(fake, real)
        lpips = self.metric_lpips(fake, real)
        return psnr, ssim, lpips

    def PPLSampler(self):
        return None

    def to_image(self, batch_or_batch_list):
        if isinstance(batch_or_batch_list, list):
            out = torch.cat([batch_out[DIK.IMAGE_OUT][0] for batch_out in batch_or_batch_list], -1)
        elif isinstance(batch_or_batch_list, dict):
            out = batch_or_batch_list[DIK.IMAGE_OUT][0]
        else:
            raise Warning("Invalid type: '{}'".format(type(batch_or_batch_list)))
        return self.visualizer_complete.tensor2image(out, return_format='pil')

    def to_video(self, batch_list, path_out, fps=15, quality=9, reverse=False):
        assert path_out.endswith(".mp4"), "Path should end with .mp4"
        frames = [self.to_image(batch) for batch in batch_list]
        if reverse:
            frames = frames + frames[::-1]
        imageio.mimwrite(path_out, frames, fps=fps, quality=quality)

    def load_shape_expressions(self):
        import numpy as np
        import os
        validation_indices = list(np.load(os.path.join(self.opt.dataroot_npy, "dataset_splits.npz"))["val"])
        sp = np.load(os.path.join(self.opt.dataroot_npy, "sp.npy"))[validation_indices]
        ep = np.load(os.path.join(self.opt.dataroot_npy, "ep.npy"))[validation_indices]
        return torch.Tensor(sp), torch.Tensor(ep)

    def run(self, z, sp, ep, theta, t=torch.Tensor([0, -2, -57])):
        batch = {
            DIK.STYLE_LATENT: z,
            DIK.COEFF_SHAPE: sp,
            DIK.COEFF_EXPRESSION: ep,
            DIK.T: t
        }
        batch = {k: v.to(self.device) for k, v in batch.items()}

        batch = self.visualizer_complete.visualize_single(self.pipeline, batch, 0, theta_all=theta,
                                                          forward_type='style2image')
        batch = {k: v.detach().cpu() for k, v in batch.items()}
        return batch


    def slerp(self, a, b, t):
        a = a / a.norm(dim=-1, keepdim=True)
        b = b / b.norm(dim=-1, keepdim=True)
        d = (a * b).sum(dim=-1, keepdim=True)
        p = t * torch.acos(d)
        c = b - d * a
        c = c / c.norm(dim=-1, keepdim=True)
        d = a * torch.cos(p) + c * torch.sin(p)
        d = d / d.norm(dim=-1, keepdim=True)
        return d