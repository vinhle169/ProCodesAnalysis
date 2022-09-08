from typing import List, Tuple
import matplotlib.pyplot as plt
import numpy as np
from unet import *
from tqdm import tqdm
from utils import *
import itertools
from collections import Counter
from skimage.metrics import structural_similarity as ssim


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


def generate_model_outputs(model_directory, model_list, input_img_list, output_path='', parallel=True):
    """
    :param model_directory: location of all models being used
    :param model_list: list of models by name being used in above location
    :param input_img_list: list of paths for images being used
    :param output_path: where you want these outputs to be saved
    :return:
    """
    for model_name in tqdm(model_list):
        model_path = model_directory + model_name
        checkpoint = torch.load(model_path)
        # unet = UNet(num_class=3, retain_dim=True, out_sz=(256, 256), dropout=.10)
        unet, _ = create_pretrained('resnet50', 'swsl')
        if parallel:
            unet = nn.DataParallel(unet)
        unet.load_state_dict(checkpoint['model_state_dict'])
        unet.eval()
        cuda0 = torch.device('cuda:0')
        unet.to(cuda0)
        for i in range(len(input_img_list)):
            img = torch.load(input_img_list[i]).to(cuda0)
            img = img.view((1, 3, 256, 256))
            output = unet(img)
            print(model_name, model_name[:-4])
            torch.save(output, f'{output_path}{model_name[:-4]}_{i}.pt')
            del img
            del output
        del unet
        del checkpoint


def plot_different_outputs(file_paths, org_img_paths, ground_truth, name, img_size=[256, 256, 3]):
    """
    :param file_paths: list of paths of output images
    :param org_img_paths: list of input images
    :param ground_truth: list of ground truth images
    :return:
    """

    row_count = 0
    output_files = {}
    for img_name in file_paths:
        output_files.setdefault(img_name[-4],[])
        output_files[img_name[-4]].append(img_name)
    print('num rows:', (1 + 1 + len(output_files[img_name[-4]])) * len(org_img_paths))
    fig, axs = plt.subplots((1 + 1 + len(output_files[img_name[-4]])) * len(org_img_paths), 4)
    fig.set_figheight(10*len(axs))
    fig.set_figwidth(50)
    for i in range(len(org_img_paths)):
        print(i, 'ith image---------------------')
        # plot test input
        curr_img = torch.load(org_img_paths[i])
        axs[row_count][0].set_ylabel(f"input image", fontsize=40)
        for channel in range(len(curr_img)+1):
            if channel != 0:
                mini = curr_img[channel-1].view(img_size[0:2]).cpu()
            else:
                mini = curr_img.cpu()
            mini = mini.detach()
            mini = mini.numpy()
            # mini *= 10
            mini = normalize_array(mini)
            if channel == 0:
                mini = np.stack([i for i in mini], axis=-1)
                axs[row_count][channel].imshow(mini, vmin=0, vmax=1)
            else:
                one_channel = np.zeros(img_size)
                one_channel[:, :, channel-1] = mini
                axs[row_count][channel].imshow(one_channel, vmin=0, vmax=1)
        del curr_img
        row_count += 1

        # plot ground truth
        curr_img = torch.load(ground_truth[i])
        axs[row_count][0].set_ylabel(f"ground truth image", fontsize=40)
        combined = []
        for channel in range(len(curr_img)):
            one_channel = np.zeros(img_size)
            mini = curr_img[channel].view(img_size[0:2]).cpu()
            mini = mini.detach()
            mini = mini.numpy()
            # mini *= 10
            print(mini.shape)
            mini = normalize_array(mini)

            one_channel[:, :, channel] = mini
            axs[row_count][channel+1].imshow(one_channel, vmin=0, vmax=1)
            combined.append(mini)
        combined = np.stack(combined, -1)
        axs[row_count][0].imshow(combined)
        del curr_img
        del combined

        # plot resulting images
        row_count += 1
        outputs = sorted(output_files[str(i)], key=lambda x: int(x[x.find('/')+1:x.find('_',x.find('/'))]))
        print(outputs)
        for img_p in outputs:
            curr_img = torch.load(img_p)[0].cpu()
            combined = []
            print(img_p[0:img_p.find('.')])
            axs[row_count][0].set_ylabel(img_p[img_p.find('/')+1:img_p.find('.')], fontsize=50)
            for channel in range(len(curr_img)):
                one_channel = np.zeros(img_size)
                mini = curr_img[channel].view(img_size[0:2]).detach()

                mini = mini.numpy()

                if mini.max() != 0 and mini.min() != mini.max():
                    mini = normalize_array(mini)
                one_channel[:, :, channel] = mini
                im = axs[row_count][channel + 1].imshow(one_channel, vmin=0, vmax=1)
                mini = np.reshape(mini, img_size[0:2])
                combined.append(mini)
            combined = np.stack(combined, -1)
            print(combined.min(), combined.max(), 'combined')
            axs[row_count][0].imshow(combined, vmin=0, vmax=1)
            del curr_img
            row_count += 1
    cols = ['Combined', 'Channel 0', 'Channel 1', 'Channel 2']
    for ax, col in zip(axs[0], cols):
        ax.set_title(col, fontsize=50)
    # fig.subplots_adjust(right=0.8)
    # cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
    # fig.colorbar(im, cax=cbar_ax)
    plt.setp(axs, xticks=[], yticks=[])
    plt.savefig(f'{name}.png')
    print('Done~~~~~~~~~~~~~~~~~~~~')


if __name__ == '__main__':

    model_names = [
                    '1000_test.tar',
                   ]

    model_directory = 'models/resnet50_out/'

    actual_images = ['data/three_grayscale_truth/F039_trim_manual_9.pt',
                    ]
    input_image_list = ['data/three_grayscale/F039_trim_manual_9.pt', ]
    generate_model_outputs(model_directory, model_names, input_image_list, 'outputs/', parallel=True)
    print('---------------generating done---------------------')
    output_images = ['outputs/'+i for i in os.listdir('outputs/')]
    plot_different_outputs(output_images, input_image_list, actual_images, 'RESNET50', img_size=[256, 256, 3])
    print('Done')
