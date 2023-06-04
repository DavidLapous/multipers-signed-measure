## This code was written by Luis Scoccola
import numpy as np
from scipy.sparse import coo_array
from scipy.ndimage import convolve1d


def signed_betti(hilbert_function, threshold=False, sparse=False):
    n = len(hilbert_function.shape)
    res = np.copy(hilbert_function)
    # zero out the "end" of the Hilbert function
    if threshold:
        for dimension in range(n):
            slicer = tuple([slice(None) if i != dimension else -1 for i in range(n)])
            res[slicer] = 0
    weights = np.array([0, 1, -1], dtype=int)
    for i in range(n):
        res = convolve1d(res, weights, axis=i, mode="constant", cval=0)
    if sparse:
        return coo_array(res)
    else:
        return res

def rank_decomposition_by_rectangles(rank_invariant, threshold=False):
    # takes as input the rank invariant of an n-parameter persistence module
    #   M  :  [0, ..., s_1 - 1] x ... x [0, ..., s_n - 1]  --->  Vec
    # on a grid with dimensions of sizes s_1, ..., s_n. The input is assumed to be
    # given as a tensor of dimensions (s_1, ..., s_n, s_1, ..., s_n), so that,
    # at index [i_1, ..., i_n, j_1, ..., j_n] we have the rank of the structure
    # map M(i) -> M(j), where i = (i_1, ..., i_n) and j = (j_1, ..., j_n), and
    # i <= j, meaning that i_1 <= j_1, ..., i_n <= j_n.
    # NOTE :
    #   - About the input, we assume that, if not( i <= j ), then at index
    #     [i_1, ..., i_n, j_1, ..., j_n] we have a zero.
    #   - Similarly, the output at index [i_1, ..., i_n, j_1, ..., j_n] only
    #     makes sense when i <= j. For indices where not( i <= j ) the output
    #     may take arbitrary values and they should be ignored.
    n = len(rank_invariant.shape) // 2
    if threshold:
        rank_invariant = rank_invariant.copy()
        # print(rank_invariant)
        # zero out the "end"
        for dimension in range(n):
            slicer = tuple(
                [slice(None) for _ in range(n)]
                + [slice(None) if i != dimension else -1 for i in range(n)]
            )
            rank_invariant[slicer] = 0
        # print(rank_invariant)
    to_flip = tuple(range(n, 2 * n))
    return np.flip(signed_betti(np.flip(rank_invariant, to_flip)), to_flip)


# only tests rank functions with 1 and 2 parameters
def test_rank_decomposition():
    # rank of an interval module in 1D on a grid with 2 elements
    ri = np.array(
        [
            [
                1,  # 0,0
                1,  # 0,1
            ],
            [0, 1],  # 1,0  # 1,1
        ]
    )
    expected_rd = np.array(
        [
            [
                0,  # 0,0
                1,  # 0,1
            ],
            [0, 0],  # 1,0  # 1,1
        ]
    )
    rd = rank_decomposition_by_rectangles(ri)
    for i in range(2):
        for i_ in range(i, 2):
            assert rd[i, i_] == expected_rd[i, i_]

    # rank of a sum of two rectangles in 2D on a grid of 2 elements
    ri = np.array(
        [
            [
                [
                    [1, 1],  # (0,0), (0,0)  # (0,0), (0,1)
                    [1, 1],  # (0,0), (1,0)  # (0,0), (1,1)
                ],
                [
                    [0, 1],  # (0,1), (0,0)  # (0,1), (0,1)
                    [0, 1],  # (0,1), (1,0)  # (0,1), (1,1)
                ],
            ],
            [
                [
                    [0, 0],  # (1,0), (0,0)  # (1,0), (0,1)
                    [2, 2],  # (1,0), (1,0)  # (1,0), (1,1)
                ],
                [
                    [0, 0],  # (1,1), (0,0)  # (1,1), (0,1)
                    [0, 2],  # (1,1), (1,0)  # (1,1), (1,1)
                ],
            ],
        ]
    )
    expected_rd = np.array(
        [
            [
                [
                    [0, 0],  # (0,0), (0,0)  # (0,0), (0,1)
                    [0, 1],  # (0,0), (1,0)  # (0,0), (1,1)
                ],
                [
                    [0, 0],  # (0,1), (0,0)  # (0,1), (0,1)
                    [0, 0],  # (0,1), (1,0)  # (0,1), (1,1)
                ],
            ],
            [
                [
                    [0, 0],  # (1,0), (0,0)  # (1,0), (0,1)
                    [0, 1],  # (1,0), (1,0)  # (1,0), (1,1)
                ],
                [
                    [0, 0],  # (1,1), (0,0)  # (1,1), (0,1)
                    [0, 0],  # (1,1), (1,0)  # (1,1), (1,1)
                ],
            ],
        ]
    )

    rd = rank_decomposition_by_rectangles(ri)
    for i in range(2):
        for i_ in range(i, 2):
            for j in range(2):
                for j_ in range(j, 2):
                    assert rd[i, j, i_, j_] == expected_rd[i, j, i_, j_]


# only tests Hilbert functions with 1, 2, 3, and 4 parameters
def test_signed_betti():
    np.random.seed(0)
    N = 4

    # test 1D
    for _ in range(N):
        a = np.random.randint(10, 30)

        f = np.random.randint(0, 40, size=(a))
        sb = signed_betti(f)

        check = np.zeros(f.shape)
        for i in range(f.shape[0]):
            for i_ in range(0, i + 1):
                check[i] += sb[i_]

        assert np.allclose(check, f)

    # test 2D
    for _ in range(N):
        a = np.random.randint(10, 30)
        b = np.random.randint(10, 30)

        f = np.random.randint(0, 40, size=(a, b))
        sb = signed_betti(f)

        check = np.zeros(f.shape)
        for i in range(f.shape[0]):
            for j in range(f.shape[1]):
                for i_ in range(0, i + 1):
                    for j_ in range(0, j + 1):
                        check[i, j] += sb[i_, j_]

        assert np.allclose(check, f)

    # test 3D
    for _ in range(N):
        a = np.random.randint(10, 20)
        b = np.random.randint(10, 20)
        c = np.random.randint(10, 20)

        f = np.random.randint(0, 40, size=(a, b, c))
        sb = signed_betti(f)

        check = np.zeros(f.shape)
        for i in range(f.shape[0]):
            for j in range(f.shape[1]):
                for k in range(f.shape[2]):
                    for i_ in range(0, i + 1):
                        for j_ in range(0, j + 1):
                            for k_ in range(0, k + 1):
                                check[i, j, k] += sb[i_, j_, k_]

        assert np.allclose(check, f)

    # test 4D
    for _ in range(N):
        a = np.random.randint(5, 10)
        b = np.random.randint(5, 10)
        c = np.random.randint(5, 10)
        d = np.random.randint(5, 10)

        f = np.random.randint(0, 40, size=(a, b, c, d))
        sb = signed_betti(f)

        check = np.zeros(f.shape)
        for i in range(f.shape[0]):
            for j in range(f.shape[1]):
                for k in range(f.shape[2]):
                    for l in range(f.shape[3]):
                        for i_ in range(0, i + 1):
                            for j_ in range(0, j + 1):
                                for k_ in range(0, k + 1):
                                    for l_ in range(0, l + 1):
                                        check[i, j, k, l] += sb[i_, j_, k_, l_]

        assert np.allclose(check, f)

    for threshold in [True, False]:
        for _ in range(N):
            a = np.random.randint(5, 10)
            b = np.random.randint(5, 10)
            c = np.random.randint(5, 10)
            d = np.random.randint(5, 10)
            e = np.random.randint(5, 10)
            f = np.random.randint(5, 10)

            f = np.random.randint(0, 40, size=(a, b, c, d,e,f))
            sb = signed_betti(f, threshold=threshold)
            sb_ = signed_betti(f, threshold=threshold)

            assert np.allclose(sb, sb_)


if __name__ == "__main__":
    test_signed_betti()
