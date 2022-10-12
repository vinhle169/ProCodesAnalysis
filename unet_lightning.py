import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from unet import create_pretrained
from pytorch_lightning import Trainer, seed_everything
from dataset import ProCodesDataModule
import argparse
from pytorch_lightning.callbacks import ModelCheckpoint


class UnetLightning(pl.LightningModule):
    def __init__(self, unet, learning_rate=0.0005):
        super().__init__()
        self.model = unet
        self.loss_fn = nn.MSELoss(reduction='mean')
        self.learning_rate = learning_rate

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)
        loss = self.loss_fn(y_hat, y)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)
        return [optimizer], [lr_scheduler]

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)
        val_loss = self.loss_fn(y_hat, y)
        self.log("val_loss", val_loss)

    def predict_step(self, batch, batch_idx = 0, dataloader_idx = None):
        x, y = batch
        pred = self.encoder(x)
        return pred


# saves a file like: my/path/sample-mnist-epoch=02-val_loss=0.32.ckpt


def main(path, model_path, epochs = 1, batch_size = 1, gpus = 0):
    seed_everything(22, workers=True)
    unet, _ = create_pretrained('resnet34', None)
    unet_pl = UnetLightning(unet)
    z = ProCodesDataModule(data_dir=path, batch_size=batch_size,
                           test_size=0.2)
    train_loader = z.train_dataloader()
    val_loader = z.validation_dataloader()
    checkpoint_callback = ModelCheckpoint(
        every_n_epochs=200,
        save_top_k=-1,
        save_on_train_epoch_end=True,
        dirpath=model_path,
        filename="UNET-{epoch:02d}",
    )
    trainer = Trainer(gpus=gpus, max_epochs=epochs, deterministic=True, callbacks=[checkpoint_callback])
    trainer.fit(unet_pl, train_loader, val_loader)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('epochs')
    parser.add_argument('train_path')
    parser.add_argument('label_path')
    parser.add_argument('model_path')
    parser.add_argument('--batch_size')
    parser.add_argument('--gpus')
    args = parser.parse_args()
    batch_size = int(args.batch_size) if args.batch_size else 1
    epochs = int(args.epochs)
    model_path = args.model_path
    path = [args.train_path, args.label_path]
    gpus = int(args.gpus) if args.gpus else 1
    main(path, model_path, epochs, batch_size, gpus)






