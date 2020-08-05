import unittest
import torch
import torch_cusparse._C as _C


class TestSparseMM(unittest.TestCase):
    def spmm_int32(self):
        size = (2, 4)
        rows = torch.IntTensor([0, 1, 1, 0, 0, ]).cuda()
        cols = torch.IntTensor([3, 0, 1, 1, 2, ]).cuda()
        vals = torch.FloatTensor([1, 1, 1, 1, 1]).cuda()
        mat = torch.FloatTensor([[0, 0], [1, 0], [0, 1], [1, 1]]).cuda()

        # Works with unsorted coo
        out = _C.coo_spmm_int32(rows, cols, vals, size[0], size[1], mat, 1)
        print(out)

        # does not works with unsorted coo
        out = _C.coo_spmm_int32(rows, cols, vals, size[0], size[1], mat, 2)
        print(out)

        # does not works with unsorted coo
        out = _C.coo_spmm_int32(rows, cols, vals, size[0], size[1], mat, 3)
        print(out)

        # Fails
        # out = _C.coo_spmm_int32(rows, cols, vals, size[0], size[1], mat, 4)
        # print(out)

    def spmm_int64(self):

        size = (2, 4)
        rows = torch.LongTensor([0, 0, 0, 1, 1]).cuda()
        cols = torch.LongTensor([1, 2, 3, 0, 1]).cuda()
        vals = torch.FloatTensor([1, 1, 1, 1, 1]).cuda()
        mat = torch.FloatTensor([[0, 0], [1, 0], [0, 1], [1, 1]]).cuda()

        # Fails. SPMM_COO_ALG4 works only with ROW major
        out = _C.coo_spmm_int64(rows, cols, vals, size[0], size[1], mat, 4)
        print(out)
