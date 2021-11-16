from pytorch_lightning import LightningModule
from varitex.modules.GLOEncoder.GLOEncoder

class GLOModel(LightningModule):
    def __init__(self):
        super().__init__()
        # Important: This property activates manual optimization.
        # self.automatic_optimization = False

    def training_step(self, batch, batch_idx):
        
        optimizer = self.optimizers()
        optimizer.zero_grad()
        """change loss here"""
        loss = self.compute_loss(batch)
        self.manual_backward(loss)
        optimizer.step()
        """Add projection to Z"""
        prelimZ = batch[IMAGE_ENCODED]
        prelimZ = projectToL2Ball(prelimZ)
        batch[Image_Encoded] = prelimZ
        
        