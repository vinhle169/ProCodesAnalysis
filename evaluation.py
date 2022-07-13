from typing import List, Tuple

from utils import *
import itertools
from collections import Counter
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


class identifiable_submatrices:

    def __init__(self, codebook_path):
        self.path = codebook_path
        self.codebook = pd.read_csv(codebook_path, sep='.', index_col=0)
        self.idx_to_col = {i:self.codebook.columns[i] for i in range(len(self.codebook.columns))}
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
            perm_dict = {(1,1,0):[], (0,1,1):[], (1,0,1):[], (1,1,1):[]}
            keys = set(perm_dict.keys())
            for i in range(mini.shape[1]):
                if tuple(mini[:,i]) in keys:
                    perm_dict[tuple(mini[:,i])].append(i)
            # Does not satisfy is missing part of condition 1
            if [] in perm_dict.values():
                return None
            else:
                combs: List[(Tuple[int, int, int], List)] = sorted(list(perm_dict.items()), key=lambda x: len(x[1]))
                matrices = []
                # get all combinations of matrices that are possible
                for i in combs[0][1]:
                    col_i = matrix[:,i]
                    idx_i = get_index_identity(col_i)
                    chk  = [idx_i]

                    for j in combs[1][1]:
                        col_j = matrix[:, j]
                        idx_j = get_index_identity(col_j)
                        if idx_j in chk:
                            continue
                        else: chk2 = chk + [idx_j]

                        for k in combs[2][1]:
                            col_k = matrix[:, k]
                            idx_k = get_index_identity(col_k)
                            if idx_k in chk2:
                                continue
                            else: chk3 = chk2 + [idx_k]

                            for l in combs[3][1]:
                                col_l = matrix[:, l]
                                idx_l = get_index_identity(col_l)
                                if idx_l in chk3:
                                    continue
                                else:
                                    cols = np.array([col_i, col_j, col_k, col_l]).T
                                    matrices.append([cols, [i,j,k,l]])
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
            break
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
            tf = np.all(mat == 0, axis = 1)
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
                if tuple(copy_mat[i][0:3]) == (1,1,1):
                    new_positions[3] = i
                else:
                    new_positions[np.argmax(copy_mat[i][3:])] = i
            new_mat = copy_mat[new_positions,:].T
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


if __name__ == '__main__':
    pass
