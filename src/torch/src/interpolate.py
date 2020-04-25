import os
import os.path as osp
import argparse
import datasets.data_loaders as module_data
import model.loss as module_loss
import model.metric as module_metric
import model.model as module_arch
from train import get_instance
from data.dynamic_point_cloud import FileBackedDynamicPointCloud
import torch_geometric.transforms as T
from transforms import UniformSample, NormalizeScale
from utils.open3d_util import open3d_to_np_pc
import torch
from torch_geometric.data import Data
from tqdm import tqdm


def main(config, resume):
    # Input Dynamic Point Cloud
    source_dpc = FileBackedDynamicPointCloud("/home/jelmer/pc_interpolation/data/basic_sequence",
                                             "/home/jelmer/pc_interpolation/data/basic_sequence/frames.dpc")

    # Output Dynamic Point Cloud
    base_dir = osp.join("/home/jelmer/pc_interpolation/data/basic_interpolated")
    output_dpc = FileBackedDynamicPointCloud(base_dir, osp.join(base_dir, "frames.dpc"))
    naming_scheme = osp.join(base_dir, "frame_{0:03d}.ply")

    # Perform the interpolation!
    interpolate(source_dpc, output_dpc, naming_scheme, config, args)

    # Write results back to disk
    output_dpc.write_to_disk()


def interpolate(source, output, naming_scheme, config, args):
    # build model architecture
    model = get_instance(module_arch, 'arch', config)
    #model.summary()

    # load state dict
    checkpoint = torch.load(args.resume)
    state_dict = checkpoint['state_dict']
    if config['n_gpu'] > 1:
        model = torch.nn.DataParallel(model)
    model.load_state_dict(state_dict)

    # prepare model for testing
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()

    pre_transform = T.Compose([
        UniformSample(config["data_loader"]["args"]["num_points"]),
        NormalizeScale()
    ])

    with torch.no_grad():
        f1 = None
        f2 = None
        for i, open3d_pc in enumerate(tqdm(source)):
            np_pc = open3d_to_np_pc(open3d_pc)
            pc = torch.FloatTensor(np_pc)

            # If this is first iteration
            if f2 is None:
                f2 = pc
                continue

            # Shift the frames
            f1 = f2
            f2 = pc

            pos_cat = torch.cat([f1, f2])
            data = Data(pos=pos_cat)
            graph_id = torch.zeros(data.pos.size()[0])
            graph_id[f1.size()[0]:] = 1
            data.graph_id = graph_id

            data = pre_transform(data)

            pc1 = data.pos[data.graph_id == 0].to(device)
            pc2 = data.pos[data.graph_id == 1].to(device)
            batch1 = torch.zeros(pc1.size()[0], dtype=torch.int64).to(device)
            batch2 = torch.zeros(pc2.size()[0], dtype=torch.int64).to(device)

            out = model(pc1, pc2, batch1, batch2)

            idx = 2 * (i-1)
            # print("f1 min ", torch.min(f1, dim=0))
            # print("f1 max ", torch.max(f1, dim=0))
            # print("pc1 min ", torch.min(pc1, dim=0))
            # print("pc1 max ", torch.max(pc1, dim=0))
            output.add_np_pc(naming_scheme.format(idx+0), pc1.cpu().data.numpy())
            output.add_np_pc(naming_scheme.format(idx+1), out.cpu().data.numpy())
        output.add_np_pc(naming_scheme.format(idx+2), pc2.cpu().data.numpy())





if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch Template')

    parser.add_argument('-r', '--resume', default=None, type=str,
                        help='path to latest checkpoint (default: None)')
    parser.add_argument('-d', '--device', default=None, type=str,
                        help='indices of GPUs to enable (default: all)')

    args = parser.parse_args()

    if args.resume:
        config = torch.load(args.resume)['config']
    if args.device:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.device

    main(config, args)
