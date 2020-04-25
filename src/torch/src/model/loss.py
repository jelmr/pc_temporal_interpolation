import torch.nn.functional as F
import torch
from torch_cluster import knn


def one_way_matching_distortion(pc1: torch.Tensor, pc2: torch.Tensor) -> torch.Tensor:
    """ Calculates the one way matching distortion from pc1 -> pc2

    See `"Dynamic Polygon Clouds: Representation and Compression for VR/AR" <https://arxiv.org/abs/1610.00402>` paper.

    :param pc1: Tensor representing Point cloud 1
    :param pc2: Tensor representing Point cloud 2
    :return: Tensor shape [] containing the matching distortion value.
    """

    # for knn search we only consider the spatial coordinates
    spatial_f1 = pc1[..., :3] if pc1.shape[-1] > 3 else pc1
    spatial_f2 = pc2[..., :3] if pc2.shape[-1] > 3 else pc2

    # Find the nearest neighbour
    nn_f1, nn_f2 = knn(spatial_f2.contiguous(), spatial_f1.contiguous(), 1)

    # Calculate the sum of squared errors
    nn_error = pc2[nn_f2, ...] - pc1[nn_f1, ...]
    nn_squared_error = nn_error ** 2
    sum_squared_error = nn_squared_error.sum(dim=-1)

    return sum_squared_error.mean()


def symmetric_matching_distortion(pc1: torch.Tensor, pc2: torch.Tensor) -> torch.Tensor:
    """ Calculates the symmetric matching distortion from pc1 -> pc2

    See `"Dynamic Polygon Clouds: Representation and Compression for VR/AR" <https://arxiv.org/abs/1610.00402>` paper.

    :param pc1: Tensor representing Point cloud 1
    :param pc2: Tensor representing Point cloud 2
    :return: Tensor shape [] containing the matching distortion value.
    """
    loss1 = one_way_matching_distortion(pc1, pc2)
    loss2 = one_way_matching_distortion(pc2, pc1)

    return (loss1 + loss2) / 2

def point_to_plane(pc1: torch.Tensor, pc2: torch.Tensor, pc1_normals) -> torch.Tensor:
    """ Calculates the point-to-plane distortion from pc1 -> pc2

    See `"Geometric Distortion Metrics for Point Cloud Compression" <https://www.merl.com/publications/docs/TR2017-113.pdf>` paper.

    :param pc1: Tensor [bn x d] representing Point cloud 1
    :param pc2: Tensor [bn x d] representing Point cloud 2
    :param pc1_normals: Tensor [bn x 3] representing the normalized surface normals for pc1 in x/y/z.
    :return: Tensor shape [] containing the point-to-plane distortion
    """

    # for knn search we only consider the spatial coordinates
    spatial_f1 = pc1[..., :3] if pc1.shape[-1] > 3 else pc1
    spatial_f2 = pc2[..., :3] if pc2.shape[-1] > 3 else pc2

    # Find the nearest neighbour
    nn_f1, nn_f2 = knn(spatial_f2.contiguous(), spatial_f1.contiguous(), 1)

    # Calculate the sum of squared errors
    nn_error = pc2[nn_f2, :3] - pc1[nn_f1, :3]

    batch_errors = nn_error.view(-1, 1, 3)
    batch_normals = pc1_normals.view(-1, 3, 1)


    projected_error = torch.bmm(batch_errors, batch_normals)
    projected_error = projected_error ** 2

    return projected_error.mean()

