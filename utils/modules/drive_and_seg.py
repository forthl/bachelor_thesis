import torch

import lightning.pytorch as pl


class DriveAndSeg(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.save_hyperparameters()

    def training_step(self, batch):
        pc, imgs = batch
        print(pc.shape, imgs.shape)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=2e-4)
        scheduler = torch.optim.lr_scheduler.PolynomialLR(optimizer)
        return [optimizer], [scheduler]