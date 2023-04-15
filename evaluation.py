from typing import List, Tuple
import matplotlib.pyplot as plt
import numpy as np
from unet import *
from tqdm import tqdm
from utils import *
import itertools
from collections import Counter
from skimage.metrics import structural_similarity as ssim
from unet_lightning import UnetLightning
from pytorch_lightning import Trainer, seed_everything
from graph_color import make_plotable_4chan
from torchvision.transforms import Resize
import torchvision
from dataset import ProCodesDataModule

def elementwise_accuracy(img_1, img_2, ignore_zeros=False, deviation=0.001):
    '''
    Calculates Error between two images elementwise
    :param img_1:
    :param img_2:
    :param ignore_zeros: bool which ignores zeros in calculation for elementwise accuracy
    :param deviation:
    :return: element wise accuracy between two images as a float
    '''
    print(img_1.shape, img_2.shape)
    assert img_1.shape == img_2.shape

    if ignore_zeros:
        img_1, img_2 = img_1.ravel(), img_2.ravel()
        nonzero_1 = set(np.flatnonzero(img_1))
        nonzero_2 = set(np.flatnonzero(img_2))
        nonzero = nonzero_1.intersection(nonzero_2)
        diffs = [np.abs(img_1[i] - img_2[i]) for i in nonzero]
        diffs = np.array(diffs)
        rounded_diff = np.where(diffs <= deviation, 1, 0)
        return np.sum(rounded_diff) / len(nonzero_1)
    else:
        img_diff = np.abs(img_1 - img_2)
        rounded_diff = np.where(img_diff <= deviation, 1, 0)
        return np.sum(rounded_diff) / img_diff.size


def mse(img_1, img_2):
    '''
    Mean Squared Error
    :param img_1:
    :param img_2:
    :return: mean squared error as a float
    '''
    diff = np.sum((img_1 - img_2) ** 2)
    return diff / img_1.size


def ssim_err(img_1, img_2, channel_axis=None, img=False):
    '''
    Structural Similarity Index
    :param img_1:
    :param img_2:
    :return: List containing  SSIM and Image of it if wanted, SSIM is a float [0,1]
    '''
    # need to reshape so that channels are at the end
    img1 = np.moveaxis(img_1.copy(), 0, -1)
    img2 = np.moveaxis(img_2.copy(), 0, -1)
    index = ssim(img1, img2, data_range=1, channel_axis=channel_axis, full=img)
    return [index]


def compare_images(imageA, imageB):
    '''
    Compute the mean squared error and structural similarity
    :param imageA:
    :param imageB:
    :return: None
    '''
    m = mse(imageA, imageB)
    s = ssim_err(imageA, imageB)

    print(f"MSE: {m}, SSIM: {s}")


class identifiable_submatrices:

    def __init__(self, codebook_path):
        self.path = codebook_path
        self.codebook = pd.read_csv(codebook_path, sep='.', index_col=0)
        self.idx_to_col = {i: self.codebook.columns[i] for i in range(len(self.codebook.columns))}
        self.idx_to_row = {i: self.codebook.index[i] for i in range(len(self.codebook.index))}
        self.codebook = pd.read_csv(codebook_path, sep='.', index_col=0).to_numpy()
        self.identifiable_submatrices(self.codebook)
        self.prettify_matrices()

    def identifiable_submatrices(self, codebook):
        '''
        Find all possible special matrices given codebook
        Special matrix in this case is a 6 x 4 matrix where:
        Condition 1:
            First 3 rows represent a smaller 3 x 4 subset where each 3 x 1 column is a different permutation
            (Can be in any order)
                [[1 0 1 1]
                 [0 1 1 1]
                 [1 1 0 1]]
        Condition 2:
            Last 3 rows represent 3/4th of an identity
                [[1 0 0 0]
                 [0 1 0 0]
                 [0 0 1 0]]
        :param codebook_path:
        :return: list of all possible matrices that fulfill above constraints
        '''

        def get_index_identity(column):
            '''
            Get the index of the identity portion of the column (condition 2 from above)
            :param column:
            :return:
            '''
            if tuple(column[0:3]) == (1, 1, 1):
                return 4
            return np.argmax(column[3:])

        def check_for_matrix(matrix):
            '''
            Finds all the submatrices in given matrix
            :param matrix:
            :return:
            '''
            mini = matrix[0:3]
            perm_dict = {(1, 1, 0): [], (0, 1, 1): [], (1, 0, 1): [], (1, 1, 1): []}
            keys = set(perm_dict.keys())
            for i in range(mini.shape[1]):
                if tuple(mini[:, i]) in keys:
                    perm_dict[tuple(mini[:, i])].append(i)
            # Does not satisfy is missing part of condition 1
            if [] in perm_dict.values():
                return None
            else:
                combs: List[(Tuple[int, int, int], List)] = sorted(list(perm_dict.items()), key=lambda x: len(x[1]))
                matrices = []
                # get all combinations of matrices that are possible
                for i in combs[0][1]:
                    col_i = matrix[:, i]
                    idx_i = get_index_identity(col_i)
                    chk = [idx_i]

                    for j in combs[1][1]:
                        col_j = matrix[:, j]
                        idx_j = get_index_identity(col_j)
                        if idx_j in chk:
                            continue
                        else:
                            chk2 = chk + [idx_j]

                        for k in combs[2][1]:
                            col_k = matrix[:, k]
                            idx_k = get_index_identity(col_k)
                            if idx_k in chk2:
                                continue
                            else:
                                chk3 = chk2 + [idx_k]

                            for l in combs[3][1]:
                                col_l = matrix[:, l]
                                idx_l = get_index_identity(col_l)
                                if idx_l in chk3:
                                    continue
                                else:
                                    cols = np.array([col_i, col_j, col_k, col_l]).T
                                    matrices.append([cols, [i, j, k, l]])
                return matrices

        # getting the permutations of all the possible row orderings
        permutations = list(itertools.permutations(list(range(codebook.shape[0]))))
        possible_matrices = {}
        print(len(permutations), "number of combinations of row orderings")
        nones = 0
        for perm in permutations:
            codebook_copy = codebook[perm, :]
            matrices = check_for_matrix(codebook_copy)
            if not matrices:
                continue
            nones += len(matrices)
            possible_matrices[perm] = matrices
        print(nones, "number of possible submatrices")
        self.possible_matrices = possible_matrices
        return possible_matrices

    def prettify_matrices(self):
        '''
        Make sure the matrix appears correctly
        [[x x x 1]
        [x x x 1]
        [x x x 1]
        [1 x x 0]
        [x 1 x 0]
        [x x 1 0]
        :return:
        '''

        # returns new matrix without the empty row and the row index that was deleted
        def delete_empty(mat, row_ordering):
            tf = np.all(mat == 0, axis=1)
            empty = np.argmax(tf)
            tf = np.invert(tf)
            new_mat = mat[tf]
            row_ordering.pop(empty)
            return new_mat, tuple(row_ordering)

        # returns new matrix and the new shifted column order
        def shift_matrices(mat, cols):
            cols = np.array(cols)
            copy_mat = np.copy(mat).T
            new_positions = [0 for i in range(len(copy_mat))]
            for i in range(len(copy_mat)):
                if tuple(copy_mat[i][0:3]) == (1, 1, 1):
                    new_positions[3] = i
                else:
                    new_positions[np.argmax(copy_mat[i][3:])] = i
            new_mat = copy_mat[new_positions, :].T
            new_cols = cols[new_positions]
            return new_mat, new_cols

        matrices = self.possible_matrices
        self.clean_matrices = {}
        for row_ordering, vals in self.possible_matrices.items():
            for v in vals:
                m = v[0]
                c = v[1]
                mat, row_order = delete_empty(m, list(row_ordering))
                new_mat, col_order = shift_matrices(mat, c)
                self.clean_matrices.setdefault(row_order, list())
                # check for repeats
                flag = 0
                for elem in self.clean_matrices[row_order]:
                    if Counter(elem[1]) == Counter(col_order):
                        flag = 1
                if flag == 0:
                    self.clean_matrices[row_order].append([new_mat, col_order])


def generate_model_outputs(model_directory, model_list, data_path, input_img_list, img_size=(256, 256), output_path='',
                           parallel=False, in_channels=3, out_channels=3):
    """
    :param model_directory: location of all models being used
    :param model_list: list of models by name being used in above location
    :param input_img_list: list of paths for images being used
    :param output_path: where you want these outputs to be saved
    :return:
    """
    assert torch.cuda.is_available(), "Need GPU to run this function"
    print(data_path, 'in funct in eval')
    z = ProCodesDataModule(data_dir=data_path, batch_size=1,
                           test_size=0, image_size=img_size)
    for model_name in tqdm(model_list):
        model_path = model_directory + model_name
        # unet = UNet(num_class=3, retain_dim=True, out_sz=(256, 256), dropout=.10)
        unet, _ = create_pretrained('resnet34', None, in_channels=in_channels, classes=out_channels)
        unet = UnetLightning(unet)
        checkpoint = torch.load(model_path)
        unet.load_state_dict(checkpoint["state_dict"])
        unet.eval()
        cuda0 = torch.device('cuda:0')
        unet.to(cuda0)

        train_loader = z.train

        for filename in input_img_list:
            img, truth = train_loader.get_item(filename)
            img_shape = list(img.shape)
            img = img.view([1] + img_shape).to(cuda0)
            output = unet.predict(img)
            torch.save(output, f'{output_path}{model_name[:model_name.rfind(".")]}_{filename}')
            del img
            del output
        del unet


def plot_different_outputs_HPA(filenames: list, checkpoint_path: str, data_path: str, plot_name: str ='example', dimensions: tuple = (512, 512)):
    '''
    :param filenames: filenames
    :param checkpoint: path to checkpoint
    :param data_path: path to data, this folder should contain a train/ folder and a truth/ folder
    :param plot_name: name of plot to be saved, optional
    :param dimensions: dimensions of the images, [width, height]
    :return:
    '''
    u, _ = create_pretrained('resnet34', None, in_channels=4, classes=4)
    model = UnetLightning(u)
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    dims = (1, 4) + dimensions
    results = []
    loss_fn = nn.MSELoss(reduction='mean')
    for file in tqdm(filenames):
        x = torch.load(f'{data_path}train_gc_512/{file}')
        y = torch.load(f'{data_path}truth_gc_512/{file}')
        x = x.clone().detach().type(torch.float)
        y = y.clone().detach().type(torch.float)
        y = y.view(dims)
        x = x.view(dims)
        y_hat = model.predict(x)
        loss = loss_fn(y_hat, y)
        x = make_plotable_4chan(x[0], train=True)
        y_hat = make_plotable_4chan(y_hat[0])
        y = make_plotable_4chan(y[0])
        results.append([x, y_hat, y, loss, file])



    results.sort(key=lambda x: x[3])
    fig, ax = plt.subplots(len(filenames), 3, figsize=(15, 22))

    for i in tqdm(range(len(filenames))):
        for j in range(3):
            if j == 2:
                ax[i][j].set_xlabel(f'Loss {round(results[i][3].item(), 4)}', fontsize=16)
            ax[i][j].imshow(results[i][j])
    column_titles = ['Test Image', 'Result', 'Ground Truth']

    for a, row in zip(ax[:, 0], range(len(filenames))):
        a.set_ylabel(results[row][4][0:10], rotation=0, fontsize=16)

    for i, a in enumerate(ax.flatten()[:3]):
        a.set_title(f'{column_titles[i]}', fontsize=20)
    fig.suptitle('Results, Sorted By Loss', fontsize=24)
    fig.subplots_adjust(top=0.92, wspace=0.05, hspace=0.20, left=0.100, right=0.900)
    plt.savefig(f'{plot_name}.png')


def hpa_classification_accuracy(test_files: list, checkpoint_path: str, test_path: str, cell_seg_path: str, metadata_path: str, dimensions: tuple = (512, 512)):
    '''
    :param test_files: filenames
    :param checkpoint_path: path to checkpoint
    :param test_path: path to test data
    :param cell_seg_path: path to cell segmentation masks
    :param metadata_path: path to metadata json file
    :param dimensions: dimensions of the images, [width, height]
    :return:
    '''
    metadata = open(metadata_path)
    metadata = json.load(metadata)
    def mapping_function(x, filename):
        if x!=0:
            return metadata[filename][str(x)]+1
        else:
            return 0
    v_color = np.vectorize(mapping_function)

    u, _ = create_pretrained('resnet34', None, in_channels=4, classes=4)
    model = UnetLightning(u)
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    accuracies = []
    aug = Resize(dimensions, interpolation=torchvision.transforms.InterpolationMode.NEAREST,
                 antialias=False)
    for test_f in tqdm(test_files):
        x = torch.load(f'{test_path}train_gc_512/{test_f}')
        x = x.clone().detach().type(torch.float)
        x = x.view((1,4) + dimensions)
        y_hat = model.predict(x)


        colored_mask = torch.load(f'{test_path}truth_gc_512/{test_f}')
        mask_over_zero = colored_mask > 0
        color_mask = colored_mask.max(axis=1, keepdim=True).indices + 1

        y_hat = y_hat.max(axis=1, keepdim=True).indices + 1
        acc = y_hat == color_mask
        acc = acc.float()
        acc *= mask_over_zero.float()
        accuracies.append(torch.sum(acc)/torch.sum(mask_over_zero))
    accs = np.array(accuracies)
    return np.mean(accs)


def single_image_result(in_channels, out_channels, checkpoint_path, paths, image_size=(2048,2048)):
    u, _ = create_pretrained('resnet34', None, in_channels=in_channels, classes=out_channels)
    model = UnetLightning(u)
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    dataloader = ProCodesDataModule(paths, test_size=2, image_size=image_size)
    training = dataloader.train_dataloader()
    for item in training:
        x,y = item
        break
    y_hat = model.predict(x)
    y = make_plotable(y[0])
    y_hat = make_plotable(y_hat[0])
    fig,ax = plt.subplots(1,2)
    fig.set_figheight(15)
    fig.set_figwidth(15)
    ax[0].imshow(y)
    ax[1].imshow(y_hat)
    ax[0].set_title('Truth')
    ax[1].set_title('Pred')
    plt.savefig('single_test.png')


def plot_outputs(output_path, filenames, data_path, model_names, result_img_path='output_images/', cpu=True, rgb=False,
                 channels=[], luminance_scale=1):
    if rgb:
        assert len(channels) == 3, 'Need to specify 3 channels in a list to display as rgb'
    for file in tqdm(filenames):
        input_filename = data_path + 'train/' + file
        truth_filename = data_path + 'truth/' + file
        if cpu:
            img_x = torch.load(input_filename, map_location=torch.device('cpu'))
            img_y = torch.load(truth_filename, map_location=torch.device('cpu'))
        else:
            img_x = torch.load(input_filename)
            img_y = torch.load(truth_filename)
        img_x = make_plotable(img_x)
        img_y = make_plotable(img_y)
        if rgb:
            img_x = np.stack([img_x[:,:,i] for i in channels], axis=-1)
            img_y = np.stack([img_y[:,:,i] for i in channels], axis=-1)
        else:
            img_x = np.where(img_x.max(-1) == 0, np.nan, img_x.argmax(-1))
            img_y = np.where(img_y.max(-1) == 0, np.nan, img_y.argmax(-1))
        for model_name in model_names:
            output_filename = output_path + model_name[:model_name.rfind(".")] + '_' + file
            if cpu:
                img_pred = torch.load(output_filename, map_location=torch.device('cpu'))[0]
            else:
                img_pred = torch.load(output_filename)[0]

            img_pred = make_plotable(img_pred)
            fig, ax = plt.subplots(1, 3)
            fig.set_figwidth(15)
            fig.set_figwidth(20)
            if rgb:
                img_pred = np.stack([img_pred[:,:,i] for i in channels], -1)
                ax[0].imshow(img_x * luminance_scale)
                ax[1].imshow(img_pred * luminance_scale)
                ax[2].imshow(img_y * luminance_scale)
            else:
                img_pred = np.where(img_pred.max(-1) == 0, np.nan, img_pred.argmax(-1))
                ax[0].matshow(img_x * luminance_scale)
                ax[1].matshow(img_pred * luminance_scale)
                ax[2].matshow(img_y * luminance_scale)
            plt.savefig(f'{result_img_path}{file[:file.rfind(".")]}_{model_name[:model_name.rfind(".")]}.png')
            plt.clf()




if __name__ == '__main__':
    data_path = ['/nobackup/users/vinhle/data/procodes_data/unet_train/train/',
                 '/nobackup/users/vinhle/data/procodes_data/unet_train/truth/']

    # files = ['2_F196_mp_score_max.pt',
    #      '1_F170_mp_score_max.pt',
    #      '3_F077_mp_score_max.pt',
    #      ]
    files = ['2_01_F196_mp_score_max.pt',
             '1_00_F170_mp_score_max.pt',
             '1_10_F170_mp_score_max.pt',
             '1_01_F170_mp_score_max.pt',
             '1_11_F170_mp_score_max.pt',
             '3_01_F077_mp_score_max.pt',
             ]
    # "/nobackup/users/vinhle/data/procodes_data/unet_train/train/2_01_F196_mp_score_max.pt"
    model_names=['UNET_22_procodes_patches_0284.ckpt', 'UNET_22_procodes_patches_0314.ckpt']
    models_path = 'models/unet/'
    output_path = 'outputs/'
    # single_image_result(in_channels,out_channels,checkpoint_path,paths,image_size)
    generate_model_outputs(models_path, model_names, data_path, files, output_path='outputs/',
                           in_channels=22, out_channels=22)
    # plot_outputs(output_path, files, '/nobackup/users/vinhle/data/procodes_data/unet_train/', model_names, rgb=True,
    #              channels=[2,4,6], luminance_scale=3)

