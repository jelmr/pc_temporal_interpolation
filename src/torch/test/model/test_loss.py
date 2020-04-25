import torch
import unittest
import numpy as np
from model.loss import (one_way_matching_distortion,
                        symmetric_matching_distortion,
                        point_to_plane)


class TestLoss(unittest.TestCase):

    def test_one_way_matching_distortion(self):
        f1 = torch.tensor([[0, 0], [0, 1], [0, 2], [1, 0], [1, 1], [2, 0]], dtype=torch.float32)
        f2 = torch.tensor([[2, 2], [2, 1], [2, 0], [1, 2], [1, 1], [0, 2]], dtype=torch.float32)

        loss_ = one_way_matching_distortion(f2, f1).item()

        self.assertAlmostEqual(loss_, 0.6666667, places=4)


    def test_symmetric_matching_distortion_small(self):
        f1 = torch.tensor([[0, 0], [0, 1], [0, 2], [1, 0], [1, 1], [2, 0]], dtype=torch.float32)
        f2 = torch.tensor([[2, 2], [2, 1], [2, 0], [1, 2], [1, 1], [0, 2]], dtype=torch.float32)

        loss_1 = symmetric_matching_distortion(f2, f1).item()
        self.assertAlmostEqual(loss_1, 0.6666667, places=4)

        loss_2 = symmetric_matching_distortion(f1, f2).item()
        self.assertAlmostEqual(loss_2, 0.6666667, places=4)

        self.assertAlmostEqual(loss_1, loss_2, places=5)

    def test_symmetric_matching_distortion_large(self):
        f1 = torch.rand(1000, 6)
        f2 = torch.rand(1200, 6)

        loss_1 = symmetric_matching_distortion(f2, f1).item()
        loss_2 = symmetric_matching_distortion(f1, f2).item()

        self.assertAlmostEqual(loss_1, loss_2, places=5)
        return True

    def test_point_to_plane_small(self):
        sq2 = 2 ** -0.5
        f1 = torch.FloatTensor([[0.,0.,0.], [101,102,103], [201,202,203]])
        f2 = torch.FloatTensor([[5.,4.,10.], [106,107,103], [206-sq2, 207+sq2, 203]])
        f1_normals = torch.FloatTensor([[0.,0.,1.], [-sq2, sq2, 0], [-sq2, sq2, 0]])

        loss = point_to_plane(f1, f2, f1_normals).squeeze()
        np.testing.assert_almost_equal(loss, (100.+ 0.+1.)/ 3, decimal=5)

    def test_point_to_plane_identity(self):
        f1 = torch.rand(1000, 6)
        f2 = f1.clone()
        f1_normals = torch.rand(1000, 3)

        loss = point_to_plane(f1, f2, f1_normals).squeeze()
        np.testing.assert_almost_equal(loss.data.numpy(), 0, decimal=5)




