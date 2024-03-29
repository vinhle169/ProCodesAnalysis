import time
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from unet import create_pretrained
from pytorch_lightning import Trainer, seed_everything
from dataset import ProCodesDataModule
import argparse
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.loggers import WandbLogger

###### TODO
### add wandb, add classification accuracy
def classification_accuracy(img, label, mask_over_zero):
    # Get the "classifications" of each pixel
    img_max = img.max(axis=1, keepdim=True).indices
    label_max = label.max(axis=1, keepdim=True).indices
    # Check which pixels match between img and label
    acc = img_max == label_max
    # Compute accuracy
    acc = acc.float()
    acc *= mask_over_zero.float()
    return torch.sum(acc)/torch.sum(mask_over_zero)


class UnetLightning(pl.LightningModule):
    def __init__(self, unet, learning_rate=0.0001):
        super().__init__()
        self.model = unet
        # self.loss_fn = nn.MSELoss(reduction='mean')
        self.loss_fn = nn.L1Loss(reduction='mean')
        self.mse = nn.MSELoss(reduction='mean')
        self.learning_rate = learning_rate
        self.running_loss = 0
        self.running_val_loss = 0
        self.num_batches = 0
        self.num_batches_val = 0
        # need this to load trainable model again
        self.save_hyperparameters()


    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)
        loss = self.loss_fn(y_hat, y)
        # self.log_dict({"Loss": loss, "step": self.current_epoch + 1})
        self.log("L1 Loss", loss, on_step=False, on_epoch=True)
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
        val_mse = self.mse(y_hat, y)

        # code to get just the non-zero values of the prediction and truth
        pred_numpy = y_hat.clone().detach().cpu().numpy()
        truth_numpy = y.clone().detach().cpu().numpy()
        truth_mask = truth_numpy > 0
        pred_np = np.zeros_like(pred_numpy)
        pred_np[truth_mask] = pred_numpy[truth_mask]
        pred_nonzero = torch.from_numpy(pred_np).to('cuda:0')

        nonzero_loss = self.loss_fn(pred_nonzero, y)
        nonzero_mse = self.mse(pred_nonzero, y)

        # tensorboard logging
        # self.log_dict({"Validation Loss": val_loss, "step": self.current_epoch + 1})
        # self.log_dict({"Validation MSE": val_mse, "step": self.current_epoch + 1})
        # self.log_dict({"Validation Non-zero Loss": nonzero_loss, "step": self.current_epoch + 1})
        # self.log_dict({"Validation Non-zero MSE": nonzero_mse, "step": self.current_epoch + 1})

        # wandb logging
        self.log("Validation L1Loss", val_loss, on_step=False, on_epoch=True)
        self.log("Validation MSE", val_mse, on_step=False, on_epoch=True)
        self.log("Validation Non-zero L1Loss", nonzero_loss, on_step=False, on_epoch=True)
        self.log("Validation Non-zero MSE", nonzero_mse, on_step=False, on_epoch=True)

    def predict(self, x):
        pred = self.model(x)
        return pred


def train(data_path : list, model_path : str, epochs : int = 1, batch_size : int = 1, gpus : int = 0,
          checkpoint_location: str = None, metadata_path: str = None, in_chans = 4, out_chans = 4, activation = None):
    '''
    Training function for U-Net model
    :param data_paths: list of paths to train and truth folder, where train is first item and truth is second
                        ex input: ["train_data/", "truth_data"]
    :param model_path: path to model
    :param epochs: number of epochs the model trains for
    :param batch_size: batch size of data given to model
    :param gpus: number of gpus for pytorch to utilize
    :param checkpoint_location: if loading from a previous checkpoint, pass the directory of the checkpoint
    :return: Nothing, but the model should be saved every 5 epochs at model_path
    '''
    start = time.time()
    seed_everything(22, workers=True)

    # Initialize model, and load in checkpoint if necessary
    unet, _ = create_pretrained('resnet34', None, in_channels=in_chans, classes=out_chans, activation="sigmoid")
    if checkpoint_location:
        # checkpoint = torch.load(checkpoint_location)
        unet_pl = UnetLightning(unet, learning_rate=0.00015)
        unet_pl = unet_pl.load_from_checkpoint(unet=unet, checkpoint_path=checkpoint_location)
    else:
        unet_pl = UnetLightning(unet, learning_rate=0.00015)
    # Initialize data module for loading in data
    z = ProCodesDataModule(data_dir=data_path, batch_size=batch_size,
                           test_size=.3, image_size=(256,256), in_memory=True,
                           metadata=True, load_metadata=metadata_path)
    train_loader = z.train_dataloader()
    print('number of training batches:', len(train_loader))
    val_loader = z.validation_dataloader()
    print('number of validation batches:', len(val_loader))
    # Setup logger to track metrics for training over time
    # logger = TensorBoardLogger("runs/", name="unet_tuning")
    wandb = WandbLogger(project='UNET-Thesis', name='UNET_23channel',save_dir='wandb_runs/')
    checkpoint_callback = ModelCheckpoint(
        monitor="Validation Non-zero L1Loss",
        every_n_epochs=50,
        save_top_k=-1,
        save_on_train_epoch_end=False,
        dirpath=model_path,
        auto_insert_metric_name=False,
        filename="UNET_f16_tune_1gray_{epoch:04d}",
    )
    # Set up the trainer function and begin training
    trainer = Trainer(gpus=gpus, max_epochs=epochs, deterministic=True, callbacks=[checkpoint_callback],
                      benchmark=True, check_val_every_n_epoch=3, logger=wandb,
                      strategy="ddp_find_unused_parameters_false", precision="16", accelerator="gpu")
    trainer.fit(unet_pl, train_dataloaders = train_loader, val_dataloaders = val_loader)
    print((time.time() - start) / 60, 'minutes to run')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('epochs')
    parser.add_argument('train_path')
    parser.add_argument('label_path')
    parser.add_argument('model_path')
    parser.add_argument('--batch_size')
    parser.add_argument('--gpus')
    parser.add_argument('--checkpoint')
    parser.add_argument('--metadata_path')
    args = parser.parse_args()
    batch_size = int(args.batch_size) if args.batch_size else 1
    epochs = int(args.epochs)
    model_path = args.model_path
    metadata_path = args.metadata_path if args.metadata_path else None
    path = [args.train_path, args.label_path]
    gpus = int(args.gpus) if args.gpus else 1
    checkpoint = args.checkpoint if args.checkpoint else None
    train(path, model_path, epochs, batch_size, gpus, checkpoint, metadata_path=metadata_path, in_chans=23, out_chans=22)
