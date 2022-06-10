from typing import List, Tuple

from utils import *
import itertools
from skimage.metrics import structural_similarity as ssim

def elementwise_accuracy(img_1, img_2, deviation=0.001):
    '''
    Calculates Error between two images elementwise
    :param img_1:
    :param img_2:
    :param deviation:
    :return: element wise accuracy between two images as a float
    '''
    assert img_1.shape == img_2.shape
    img_diff = np.abs(img_1 - img_2)
    rounded_diff = np.where(img_diff <= deviation, 1, 0)
    return np.sum(rounded_diff)/img_diff.size


def mse(img_1, img_2):
    '''
    Mean Squared Error
    :param img_1:
    :param img_2:
    :return: mean squared error as a float
    '''
    diff = np.sum((img_1 - img_2)**2)
    return diff / img_1.size


def ssim_err(img_1, img_2, img=False):
    '''
    Structural Similarity Index
    :param img_1:
    :param img_2:
    :return: List containing  SSIM and Image of it if wanted
    '''
    # need to reshape
    img1 = np.moveaxis(img_1.copy(), 0, -1)
    img2 = np.moveaxis(img_2.copy(), 0, -1)
    index = ssim(img1, img2, data_range=1, channel_axis=-1, full=img)
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


def find_special_matrices(codebook_path):
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
            [[1 0 0 1]
             [0 1 0 1]
             [0 0 1 1]]
    :param codebook_path:
    :return: list of all possible matrices that fulfill above constraints
    '''
    def check_for_matrix(matrix):
        mini = matrix[0:3]
        perm_dict = {(1,1,0):[], (0,1,1):[], (1,0,1):[], (1,1,1):[]}
        keys = set(perm_dict.keys())
        for i in range(mini.shape[1]):
            if tuple(mini[:,i]) in keys:
                perm_dict[tuple(mini[:,i])].append(i)
        # Does not satisfy is missing part of condition 1
        if [] in perm_dict.values():
            return None
        else:
            cnt=0
            combs: List[(Tuple[int, int, int], List)] = sorted(list(perm_dict.items()), key=lambda x: len(x[1]))
            matrices = []
            # get all combinations of matrices that are possible
            for i in combs[0][1]:
                col_i = matrix[:,i]
                idx_i = np.argmax(col_i[3:])
                if tuple(col_i[0:3]) == (1, 1, 1):
                    idx_i = 4
                chk  = [idx_i]

                for j in combs[1][1]:
                    col_j = matrix[:, j]
                    idx_j = np.argmax(col_j[3:])
                    if tuple(col_j[0:3]) == (1, 1, 1):
                        idx_j = 4
                    if idx_j in chk:
                        continue
                    else: chk2 = chk + [idx_j]

                    for k in combs[2][1]:
                        col_k = matrix[:, k]
                        idx_k = np.argmax(col_k[3:])
                        if tuple(col_k[0:3]) == (1, 1, 1):
                            idx_k = 4
                        if idx_k in chk2:
                            continue
                        else: chk3 = chk2 + [idx_k]

                        for l in combs[3][1]:
                            cnt += 1
                            col_l = matrix[:, l]
                            if tuple(col_l[0:3]) == (1, 1, 1):
                                idx_l = 4
                            idx_l = np.argmax(col_l[3:])
                            if idx_l in chk3:
                                continue
                            else:
                                cols = np.array([col_i, col_j, col_k, col_l]).T
                                matrices.append([cols, [i,j,k,l]])
            return matrices



    codebook = pd.read_csv('codebook.csv', sep='.', index_col=0).to_numpy()
    permutations = list(itertools.permutations(list(range(codebook.shape[0]))))
    cnt = 0
    possible_matrices = {}
    print(len(permutations), "number of combinations of row orderings")
    nones = 0
    for perm in permutations:
        codebook_copy = codebook[perm, :]
        matrices = check_for_matrix(codebook_copy)
        if matrices == None:
            continue
        nones += len(matrices)
        possible_matrices[perm] = matrices
    print(nones, "number of possible submatrices")
    return possible_matrices


if __name__ == '__main__':
    # img = load('F000_max.tif')
    # markers = pd.read_csv('markers.csv')
    # print("image shape", img.shape)
    # reshape based on codebook
    # img = img[markers['marker_name'].isin(codebook.index)]
    # print("image shape", img.shape)
    x = find_special_matrices('codebook.csv')
    y = [[k,v] for k,v in x.items()][0]
    print('Row ordering: ', y[0])
    print('Possible submatrices and the column numbers they are associated with: ')
    for i in y[1]:
        print("column numbers:", i[1])
        print("submatrix: \n", i[0])


    # x = img.copy()
    # print(elementwise_accuracy(img, x))
    # noise = np.random.normal(0, 0.1, img.shape)
    # x += noise
    # compare_images(img, x)
    # max_components = 3
    # thr_factor = 0.8
    # z = matching_pursuit(x, A, max_components, thr_factor)
    # cm = plt.get_cmap('tab20')
    # cm.set_bad('k')
    # plt.figure(figsize=(12, 12))
    # plt.gca().matshow(np.where(z.max(0) == 0, np.nan, z.argmax(0)), cmap=cm)
    # plt.show()


