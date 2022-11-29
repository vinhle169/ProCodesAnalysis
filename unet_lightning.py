import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from unet import create_pretrained
from pytorch_lightning import Trainer, seed_everything
from dataset import ProCodesDataModule
import argparse
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger


class UnetLightning(pl.LightningModule):
    def __init__(self, unet, learning_rate=0.0001):
        super().__init__()
        self.model = unet
        self.loss_fn = nn.MSELoss(reduction='mean')
        self.learning_rate = learning_rate
        self.running_loss = 0
        self.running_val_loss = 0
        self.num_batches = 0
        self.num_batches_val = 0

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)
        loss = self.loss_fn(y_hat, y)
        self.log_dict({"Loss": loss, "step": self.current_epoch + 1})
        return loss

    # def on_epoch_end(self):
    #     loss = self.running_loss / max(self.num_batches,1)
    #     self.log_dict({"Loss": loss, "step": self.current_epoch + 1})
    #     self.running_loss, self.num_batches = 0, 0
    #
    #     if self.running_val_loss > 0:
    #         val_loss = self.running_val_loss / max(self.num_batches_val, 1)
    #         self.log_dict({"Validation Loss": val_loss, "step": self.current_epoch + 1})
    #         self.running_val_loss, self.num_batches_val = 0, 0

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        # lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)
        lr_scheduler = None
        return [optimizer]

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)
        val_loss = self.loss_fn(y_hat, y)
        self.log_dict({"Validation Loss": val_loss, "step": self.current_epoch + 1})

    def predict(self, x):
        pred = self.model(x)
        return pred


def main(path, model_path, epochs=1, batch_size=1, gpus=0, checkpoint_location=None):
    start = time.time()
    seed_everything(22, workers=True)
    unet, _ = create_pretrained('resnet34', None)
    if checkpoint_location:
        unet_pl = UnetLightning.load_from_checkpoint("/path/to/checkpoint.ckpt")
    else:
        unet_pl = UnetLightning(unet, learning_rate=0.0001)

    z = ProCodesDataModule(data_dir=path, batch_size=batch_size,
                           test_size=0.2)
    train_loader = z.train_dataloader()
    val_loader = z.validation_dataloader()
    logger = TensorBoardLogger("runs/", name="unet_hpa")
    checkpoint_callback = ModelCheckpoint(
        monitor="Validation Loss",
        every_n_epochs=5,
        save_top_k=5,
        save_on_train_epoch_end=True,
        dirpath=model_path,
        auto_insert_metric_name=False,
        filename="UNET-hpa-{epoch:04d}",
    )

    trainer = Trainer(gpus=gpus, max_epochs=epochs, deterministic=True, callbacks=[checkpoint_callback],
                      check_val_every_n_epoch=1, logger=logger, strategy="ddp_find_unused_parameters_false")
    trainer.fit(unet_pl, train_loader, val_loader)
    print((time.time() - start)/60, 'minutes to run')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('epochs')
    parser.add_argument('train_path')
    parser.add_argument('label_path')
    parser.add_argument('model_path')
    parser.add_argument('--batch_size')
    parser.add_argument('--gpus')
    parser.add_argument('--checkpoint')
    args = parser.parse_args()
    batch_size = int(args.batch_size) if args.batch_size else 1
    epochs = int(args.epochs)
    model_path = args.model_path
    path = [args.train_path, args.label_path]
    gpus = int(args.gpus) if args.gpus else 1
    checkpoint = args.checkpoint if args.checkpoint else None
    main(path, model_path, epochs, batch_size, gpus, checkpoint)






