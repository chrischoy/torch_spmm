import unittest
import torch
import torch_cusparse._C as _C


class TestSparseMM(unittest.TestCase):
    def spmm_int32(self):
        size = (2, 4)
        rows = torch.IntTensor([0, 0, 0, 1, 1]).cuda()
        cols = torch.IntTensor([1, 2, 3, 0, 1]).cuda()
        vals = torch.FloatTensor([1, 1, 1, 1, 1]).cuda()
        mat = torch.FloatTensor([[0, 0], [1, 0], [0, 1], [1, 1]]).cuda()

        out = _C.coo_spmm_int32(rows, cols, vals, size[0], size[1], mat, 1)
        print(out)

        out = _C.coo_spmm_int32(rows, cols, vals, size[0], size[1], mat, 2)
        print(out)

        out = _C.coo_spmm_int32(rows, cols, vals, size[0], size[1], mat, 3)
        print(out)

    def spmm_int64(self):

        size = (2, 4)
        rows = torch.LongTensor([0, 0, 0, 1, 1]).cuda()
        cols = torch.LongTensor([1, 2, 3, 0, 1]).cuda()
        vals = torch.FloatTensor([1, 1, 1, 1, 1]).cuda()
        mat = torch.FloatTensor([[0, 0], [1, 0], [0, 1], [1, 1]]).cuda()

        out = _C.coo_spmm_int64(rows, cols, vals, size[0], size[1], mat, 1)
        print(out)

        out = _C.coo_spmm_int64(rows, cols, vals, size[0], size[1], mat, 2)
        print(out)

        out = _C.coo_spmm_int64(rows, cols, vals, size[0], size[1], mat, 3)
        print(out)
