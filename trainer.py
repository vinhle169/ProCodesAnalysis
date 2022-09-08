import argparse
from colorization import *
from unet import *
from utils import classification_accuracy
from tqdm import tqdm
import torch.nn as nn
from torchvision import ops
from dataset import ProCodesDataModule
from torch.nn import MSELoss
from torch.utils.tensorboard import SummaryWriter


# noinspection PyUnboundLocalVariable,PyCallingNonCallable
def train(model: ColorizationNet, train_path: str, learning_rate: float, epochs: int, batch_size: int, model_path: str,
          loss_fn: object, continue_training: str = None, parallel: bool = True, data_type: str = 'unet'):
    """

    :param model: model object
    :param train_path: path to training, if unet this should be a list of two paths
    :param learning_rate: learning rate of the optimizer
    :param epochs:
    :param batch_size:
    :param model_path: path of where the model is to be saved
    :param continue_training: path of where the model is to be loaded from if training is continued
    :param loss_fn: loss function
    :param parallel: bool if parallel training
    :param data_type: determine type of model
    :return:
    """
    torch.cuda.empty_cache()
    start_time = time.time()
    writer = SummaryWriter(comment='RESNET50 1000 E 2')
    train_losses, epoch = [], 0
    if parallel:
        model = nn.DataParallel(model)
    if continue_training:
        model_data = torch.load(continue_training)
        train_losses = model_data['train_losses']
        epoch = model_data['epoch']
        model.load_state_dict(model_data['model_state_dict'])

    if torch.cuda.is_available():
        cuda0 = torch.device('cuda:0')
        model.to(cuda0)
    # setup hyper parameters
    criterion = loss_fn
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # repetitive if statement because of bug with cuda and model creation and optimizer creation
    if continue_training:
        optimizer.load_state_dict(model_data['optimizer_state_dict'])
    # if continuing from a previous process:

    # set up dataloader
    z = ProCodesDataModule(data_dir=train_path, batch_size=batch_size, test_size=0.2, data_type=data_type)
    train_loader = z.train_dataloader()
    length = len(train_loader)
    for e in tqdm(range(epoch, epochs)):
        running_loss = 0
        running_classification_acc = 0
        for i, image_label in enumerate(train_loader):
            image, label, zero_mask = image_label
            image = image.to(cuda0)
            label = label.to(cuda0)
            zero_mask = zero_mask.to(cuda0)
            # forward pass
            output = model(image)
            loss = criterion(output, label)
            running_loss += loss.item()
            running_classification_acc += classification_accuracy(output, label, zero_mask)
            del image
            del label
            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        classification_acc_per_epoch = running_classification_acc / length
        loss_per_epoch = running_loss / length
        writer.add_scalar("Loss/train", loss_per_epoch, e)
        writer.add_scalar("Accuracy/train", classification_acc_per_epoch, e)
        train_losses.append(loss_per_epoch)
        if (e + 1) % 100 == 0:
            torch.save({
                'epoch': e,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss,
                'train_losses': train_losses
            }, model_path + f'{e + 1}_test.tar')
        print('Epoch [{}/{}], Loss: {:.4f}, Accuracy: {:.2f}'.format(e + 1, epochs, loss.item(), classification_acc_per_epoch))
    print((time.time() - start_time)/60, ' minutes to finish')
    writer.flush()
    writer.close()
    return train_losses


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # batch_size
    parser.add_argument('batch_size')
    # epochs
    parser.add_argument('epochs')
    # path to data
    parser.add_argument('path')
    # path (optional) to pretrained model
    parser.add_argument('--model')
    # path if using unet to labeled data
    parser.add_argument('--label_path')
    args = parser.parse_args()
    batch_size = int(args.batch_size)
    epochs = int(args.epochs)
    path = args.path
    model_presave = args.model
    label_path = args.label_path
    MSE = MSELoss(reduction='mean')
    BCE = nn.BCELoss(reduction='mean')
    if label_path:
        path = [path, label_path]
    # unet = UNet(num_class=3, retain_dim=True, out_sz=(256, 256), dropout=0.05)
    unet, _ = create_pretrained('resnet50', 'swsl')
    print("BEGIN TRAINING")
    loss_data = train(unet, path, 0.0001, epochs, batch_size, 'models/resnet50_out/', loss_fn=MSE,
                      continue_training=model_presave, parallel=True)
