import pdb

import torch
import wandb
from torch.nn import BCELoss, Parameter, Embedding
import numpy as np
import os

from varitex.data.keys_enum import DataItemKey as DIK
from varitex.modules.discriminator import MultiscaleDiscriminator
from varitex.modules.generator import Generator
from varitex.modules.custom_module import CustomModule
from varitex.modules.loss import ImageNetVGG19Loss, kl_divergence, l2_loss, GANLoss
from varitex.modules.metrics import PSNR, SSIM, LPIPS, FID


from nflows.flows.base import Flow
from nflows.distributions.normal import StandardNormal
from nflows.transforms.base import CompositeTransform
from nflows.transforms.autoregressive import MaskedAffineAutoregressiveTransform
from nflows.transforms.permutations import ReversePermutation


class PipelineModule(CustomModule):

    def __init__(self, opt):
        super().__init__(opt)

        # Includes the full generation pipeline: encoder, texture decoder, additive decoder, neural renderer
        self.generator = Generator(opt)

        if getattr(self.opt, "lambda_gan", 0) > 0:
            self.discriminator = MultiscaleDiscriminator(opt)
            self.criterion_gan = GANLoss(opt.gan_mode)
            self.criterion_discriminator_features = torch.nn.L1Loss()
        if getattr(self.opt, "lambda_vgg", 0) > 0:
            self.loss_vgg = ImageNetVGG19Loss()
        if getattr(self.opt, "lambda_segmentation", 0) > 0:
            self.criterion_segmentation = BCELoss()

        self.metric_psnr = PSNR()
        self.metric_ssim = SSIM()
        self.metric_lpips = LPIPS()
        self.metric_fid = FID()

        if(self.opt.use_glo):
            #Quickly about embedding:
            #Takes in input shape (nSamples, latentDim)
            embedding_shape = np.array([getattr(self.opt, "nTrainSamples", 70000), getattr(self.opt, "latent_dim")])
            #self.Z = Embedding(embedding_shape[0],embedding_shape[1], max_norm=1.)
            self.Z = Embedding(embedding_shape[0],embedding_shape[1])
            if(self.opt.glo_init=='pca'):
                z = np.load(getattr(self.opt, "pca_file")).astype(np.single)
            else:
                #rnd initialization
                z = np.rand.randn(embedding_shape[0],embedding_shape[1])
            pcaFirst = z#self.project_l2_ball(z)
            pcaHalf = pcaFirst[:,:128]
            pcaFull = np.concatenate([pcaHalf, pcaHalf],axis=1)
            pcaFull = self.project_l2_ball(pcaFull)
            self.Z.weight = Parameter(torch.from_numpy(pcaFull))
            del z
        if(self.opt.use_NF):
            base_dist = StandardNormal(shape=[256])
            transforms = []
            for _ in range(4):
                transforms.append(ReversePermutation(features=256))
                transforms.append(MaskedAffineAutoregressiveTransform(features=256,
                                                                      hidden_features=256))
            transform = CompositeTransform(transforms)
            self.flow = Flow(transform, base_dist)

        

    def project_l2_ball(self, batch):
        """ project the vectors in z onto the l2 unit norm ball"""
        ##why even use cpu here? should I use to_device here? well no it should be a tensor right? embedding yes, zi's yes
        if not isinstance(batch, np.ndarray):
            batch = batch.data.cpu().numpy()
        return self.to_device(batch / np.maximum(np.sqrt(np.sum(batch**2, axis=1))[:, np.newaxis], 1))

    def normalize_l2_ball(self, input, dim: int = 1, eps: float = 1e-12):
        denom = torch.max(torch.linalg.norm(input, 2.0, dim=1, keepdim=True).clamp_min_(eps).expand_as(input),
                          self.to_device(torch.ones(input.shape)))
        return torch.div(input, denom)
    
    def to_device(self, o, device='cuda'):
        if isinstance(o, list):
            o = [self.to_device(o_i, device) for o_i in o]
        elif isinstance(o, dict):
            o = {k: self.to_device(v, device) for k, v in o.items()}
        elif isinstance(o, torch.Tensor):
            o = o.to(device)
        return o

    def forward(self, batch, batch_idx, std_multiplier=1):
        if(self.opt.use_glo and not batch_idx==-1):
            if(batch_idx==0):
                ## Take some norm snapshots for later visualization
                folder = 'norm_snapshots'
                path_folder = os.path.join(os.getenv("OP"), self.opt.experiment_name)
                os.makedirs(path_folder, exist_ok=True)
                path_folder = os.path.join(path_folder, folder)
                os.makedirs(path_folder, exist_ok=True)
                file_name = os.path.join(path_folder, 'norms_epoch_'+str(self.current_epoch))
                norms = torch.linalg.norm(self.Z.weight,dim=1)
                np.save(file_name, norms.detach().cpu().numpy())
            batch_idxs = torch.arange(batch_idx*self.opt.batch_size,(batch_idx+1)*self.opt.batch_size, dtype=torch.long)
            tmp = self.Z(batch_idxs.cuda())
            #batch[DIK.STYLE_LATENT] = self.normalize_l2_ball(tmp)
            batch[DIK.STYLE_LATENT] = tmp
        else:
            batch = batch
        batch = self.to_device(batch, self.opt.device)
        batch = self.generator(batch, batch_idx, std_multiplier)
        return batch

    def training_step(self, batch, batch_idx, optimizer_idx=0):
        if optimizer_idx == 0:
            loss = self._generator_step(batch, batch_idx)
        elif optimizer_idx == 1:
            loss = self._discriminator_step(batch, batch_idx)
        elif (optimizer_idx == 2 and self.opt.alternate):
            batch_idxs = torch.arange(batch_idx*self.opt.batch_size,(batch_idx+1)*self.opt.batch_size, dtype=torch.long)
            loss = -self.flow.log_prob(inputs=self.Z(batch_idxs.cuda())).mean() *self.opt.lambda_flow
        else:
            raise Warning("Invalid optimizer index: {}".format(optimizer_idx))
        return loss


    def validation_step(self, batch, batch_idx, std_multiplier=1):
        batch = self.forward(batch, batch_idx, std_multiplier=std_multiplier)
        fake = batch[DIK.IMAGE_OUT]
        real = batch[DIK.IMAGE_IN]

        # psnr = self.metric_psnr(fake, real)
        # ssim = self.metric_ssim(fake, real)
        # lpips = self.metric_lpips(fake, real)
        # fid = self.metric_fid(fake, real)

        latentMu = torch.mean(batch[DIK.STYLE_LATENT],dim=1).mean().item()
        latentStd = torch.std(batch[DIK.STYLE_LATENT],dim=1).mean().item()
        samples = self.flow.sample(self.opt.batch_size)
        nfMu = torch.mean(samples,dim=1).mean().item()
        nfStd = torch.std(samples,dim=1).mean().item()

        self.log_dict({
            #"val/psnr": psnr,
            #"val/ssim": ssim,
            #"val/lpips": lpips,
            #"val/fid": fid,
            "val/latentMu": latentMu,
            "val/latentStd": latentStd,
            "val/nfMu": nfMu,
            "val/nfStd": nfStd
        })

    # Below methods simply forward the calls to the generator
    def forward_encode(self, batch, batch_idx):
        return self.generator.forward_encode(batch, batch_idx)

    def forward_sample_style(self, *args, **kwargs):
        return self.generator.forward_sample_style(*args, **kwargs)

    def forward_latent2texture(self, batch, batch_idx):
        return self.generator.forward_latent2featureimage(batch, batch_idx)

    def forward_texture2image(self, *args, **kwargs):
        return self.generator.texture2image(*args, **kwargs)

    def forward_latent2image(self, *args, **kwargs):
        return self.generator.forward_latent2image(*args, **kwargs)

    def forward_interior2image(self, batch, batch_idx):
        batch = self.generator.sample_texture(batch)
        batch = self.generator.forward_latent2additive_featureimage(batch, batch_idx)
        batch = self.generator.forward_merge_textures(batch, batch_idx)
        batch = self.forward_texture2image(batch, batch_idx)
        return batch

    def configure_optimizers(self):
        # We use one optimizer for the generator and one for the discriminator.
        optimizers = list()
        # Important: Should have index 0
        if(not self.opt.alternate and self.opt.use_NF):
            optimizers.append(torch.optim.Adam(list(self.generator.parameters()) + list(self.Z.parameters()) + list(self.flow.parameters()), lr=self.opt.lr))
        elif(self.opt.use_glo):
            #optimizers.append(torch.optim.Adam(list(self.generator.parameters()) + list(self.Z.parameters()) + list(self.flow.parameters()), lr=self.opt.lr))
            optimizers.append(torch.optim.Adam(list(self.generator.parameters()) + list(self.Z.parameters()) , lr=self.opt.lr))
        else:
            optimizers.append(torch.optim.Adam(self.generator.parameters() , lr=self.opt.lr))

        if getattr(self.opt, "lambda_gan", 0) > 0:
            # Needs index 1
            optimizers.append(torch.optim.Adam(self.discriminator.parameters(),
                                               lr=self.opt.lr_discriminator))

        if self.opt.alternate:
            # Needs index 2
            optimizers.append(torch.optim.Adam(self.flow.parameters(),
                                               lr=self.opt.lr_flow))
        return optimizers, []

    def _generator_step(self, batch, batch_idx):
        batch = self.forward(batch, batch_idx)

        loss_gan = 0
        loss_gan_features = 0
        loss_l2 = 0
        loss_vgg = 0
        loss_segmentation = 0
        loss_rgb_texture = 0
        loss_flow = 0

        image_out = batch[DIK.IMAGE_OUT]
        image_in = batch[DIK.IMAGE_IN]
        if(self.opt.use_glo):
            loss_kl=0
        else:
            loss_kl = kl_divergence(batch[DIK.STYLE_LATENT_MU], batch[DIK.STYLE_LATENT_STD]).mean()
        if (not self.opt.alternate and self.opt.use_NF):
            loss_flow = -self.flow.log_prob(inputs=batch[DIK.STYLE_LATENT]).mean()

        if getattr(self.opt, "lambda_gan", 0) > 0:
            pred_fake, pred_real = self._forward_discriminate(image_out, image_in)

            loss_gan = self.criterion_gan(pred_fake, True,
                                          for_discriminator=False)

            # Feature loss
            num_D = len(pred_fake)
            GAN_Feat_loss = torch.FloatTensor(1).fill_(0).to(image_out.device)
            for i in range(num_D):  # for each discriminator
                # last output is the final prediction, so we exclude it
                num_intermediate_outputs = len(pred_fake[i]) - 1
                for j in range(num_intermediate_outputs):  # for each layer output
                    unweighted_loss = self.criterion_discriminator_features(
                        pred_fake[i][j], pred_real[i][j].detach())
                    GAN_Feat_loss += unweighted_loss
            loss_gan_features = GAN_Feat_loss / num_D

        if self.opt.lambda_l2 > 0:
            loss_l2 = l2_loss(image_out, image_in)

        if self.opt.lambda_vgg > 0:
            loss_vgg = self.loss_vgg(image_out.clone(), image_in.clone())

        if self.opt.lambda_segmentation > 0:
            loss_segmentation = self.criterion_segmentation(batch[DIK.SEGMENTATION_PREDICTED],
                                                            batch[DIK.SEGMENTATION_MASK])

        if self.opt.lambda_rgb_texture > 0:
            texture = batch[DIK.FACE_FEATUREIMAGE].clone()[:, :3]  # First three dimensions should be RGB only
            masked_image_in = image_in.clone()
            masked_image_in *= batch[DIK.MASK_UV].expand_as(image_in)
            texture *= batch[DIK.MASK_UV].expand_as(texture)
            loss_rgb_texture = l2_loss(texture, masked_image_in)

        loss_unweighted = loss_gan + loss_gan_features + loss_l2 + loss_kl + loss_vgg + loss_segmentation + loss_rgb_texture + loss_flow

        loss = self.opt.lambda_gan * loss_gan + \
               self.opt.lambda_discriminator_features * loss_gan_features + \
               self.opt.lambda_l2 * loss_l2 + \
               self.opt.lambda_kl * loss_kl + \
               self.opt.lambda_vgg * loss_vgg + \
               self.opt.lambda_segmentation * loss_segmentation + \
               self.opt.lambda_rgb_texture * loss_rgb_texture + \
               self.opt.lambda_flow * loss_flow

        data_log = {
            "train/generator": loss_gan,
            "train/gan_features": loss_gan_features,
            "train/reconstruction_l2": loss_l2,
            "train/kl": loss_kl,
            "train/vgg_l1": loss_vgg,
            "train/segmentation": loss_segmentation,
            "train/rgb_texture": loss_rgb_texture,
            "train/lossUnweighted": loss_unweighted,
            "train/losweighted": loss,
            "train/flow": loss_flow
        }
        # Filter out zero losses
        data_log = {k: v.clone().detach() for k, v in data_log.items() if v != 0}
        self.log_dict(data_log)
        return loss

    def _discriminator_step(self, batch, batch_idx):
        image_real = batch[DIK.IMAGE_IN]
        with torch.no_grad():
            batch = self.forward(batch, batch_idx)
            image_fake = batch[DIK.IMAGE_OUT].detach()
            image_fake.requires_grad = True

        fake_pred, real_pred = self._forward_discriminate(image_fake, image_real)

        real_loss = self.criterion_gan(real_pred, True,
                                       for_discriminator=True)
        fake_loss = self.criterion_gan(fake_pred, False,
                                       for_discriminator=True)
        loss_unweighted = (real_loss + fake_loss) / 2

        loss = self.opt.lambda_gan * loss_unweighted

        self.log_dict({
            "train/discriminator": loss_unweighted
        })
        return loss

    def _forward_discriminate(self, fake_image, real_image):
        # This method is from SPADE: https://github.com/NVlabs/SPADE
        # In Batch Normalization, the fake and real images are
        # recommended to be in the same batch to avoid disparate
        # statistics in fake and real images.
        # So both fake and real images are fed to D all at once.
        fake_and_real = torch.cat([fake_image, real_image], dim=0)
        discriminator_out = self.discriminator(fake_and_real)  # len(2); one per discriminator

        # Take the prediction of fake and real images from the combined batch
        def divide_pred(pred):
            # the prediction contains the intermediate outputs of multiscale GAN,
            # so it's usually a list
            if type(pred) == list:
                fake = []
                real = []
                for p in pred:
                    fake.append([tensor[:tensor.size(0) // 2] for tensor in p])
                    real.append([tensor[tensor.size(0) // 2:] for tensor in p])
            else:
                fake = pred[:pred.size(0) // 2]
                real = pred[pred.size(0) // 2:]

            return fake, real

        pred_fake, pred_real = divide_pred(discriminator_out)

        return pred_fake, pred_real
