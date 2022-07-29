import argparse
from colorization import *
from unet import *
from tqdm import tqdm
import torch.nn as nn
from dataset import ProCodesDataModule


# noinspection PyUnboundLocalVariable,PyCallingNonCallable
def train(model: ColorizationNet, train_path: str, learning_rate: float, epochs: int, batch_size: int, model_path: str,
          loss_fn: object, continue_training: str = None, parallel: bool = True, data_type: str = 'unet'):
    '''

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
    '''
    torch.cuda.empty_cache()
    start_time = time.time()
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
    optimizer = torch.optim.Adam(model.parameters(), lr= learning_rate)

    # repetitive if statement because of bug with cuda and model creation and optimizer creation
    if continue_training:
        optimizer.load_state_dict(model_data['optimizer_state_dict'])
    # if continuing from a previous process:

    # set up dataloader
    z = ProCodesDataModule(data_dir=train_path, batch_size=batch_size, test_size=0.3, data_type=data_type)
    train_loader = z.train_dataloader()
    for e in tqdm(range(epoch, epochs)):
        running_loss = 0
        for i, image_label in enumerate(train_loader):
            image, label = image_label
            image = image.to(cuda0)
            label = label.to(cuda0)
            # forward pass
            output = model(image)
            loss = criterion(output, label)
            del image
            del label
            running_loss += loss.item()
            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        train_losses.append(running_loss / len(train_loader))
        if (e + 1) % 20 == 0:
            torch.save({
                'epoch': e,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss,
                'train_losses': train_losses
            }, model_path + f'{e + 1}_checkpoint.tar')
        print('Epoch [{}/{}], Loss: {:.4f}'.format(e + 1, epochs, loss.item()))
    print((time.time() - start_time)/60, ' minutes to finish')
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
    parser.add_argument('--label_path')
    args = parser.parse_args()
    batch_size = int(args.batch_size)
    epochs = int(args.epochs)
    path = args.path
    model_presave = args.model
    label_path = args.label_path
    if label_path:
        path = [path, label_path]
    unet = UNet(num_class = 3, retain_dim=True)
    print("BEGIN TRAINING")
    loss_data = train(unet, path, 0.00001, epochs, batch_size, 'models/unet/', loss_fn=FocalLoss(), continue_training=model_presave, parallel=False)
    plt.plot(loss_data, '-o')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.title('Loss Over Time')
    # noinspection SpellCheckingInspection
    plt.savefig('loss_procodes_unet_model.png')
    plt.show()
    #
    # with torch.no_grad():
    #     checkpoint = torch.load('models/90_checkpoint.tar')
    #     cnet.load_state_dict(checkpoint['model_state_dict'])
    #     img_path = 'fakedata/20240.jpg'
    #     img = cv2.imread(img_path)
    #     img = img.reshape((3,224,224))
    #     x,y,i = random_channel(img)
    #     x_ = x.copy()
    #     print(y.shape, i)
    #     y = np.insert(y, i, x, 0)
    #     print(y.shape)
    #     x= np.resize(x,(1,1,224,224))
    #     x = torch.Tensor(x)
    #     y2 = cnet(x)[0].cpu()
    #     y2 = y2.cpu()
    #     y2 = y2.numpy()
    #     y2 = np.insert(y2, i, x, 0)
    #     y.resize((224,224,3))
    #     y2.resize((224,224,3))
    #     print(y.shape, y2.shape, np.max(y))
    #     fig, axs = plt.subplots(1,3)
    #     y = cv2.cvtColor(y, cv2.COLOR_BGR2RGB)
    #     y2 = cv2.cvtColor(y2, cv2.COLOR_BGR2RGB)
    #     axs[0].set_title('Blue Channel')
    #     axs[0].imshow(x_)
    #     axs[1].set_title('Original Image')
    #     axs[1].imshow(y)
    #     axs[2].set_title('Colorized')
    #     axs[2].imshow(y2/255)
    #     plt.show()
