import argparse
import os
import random
import numpy as np
import torch
import shapenet_edge as shapenet_dataset
import model_edge as models
import util_edge as util
import ops

parser = argparse.ArgumentParser()
parser.add_argument('--remove_point_num', type=int, default=512)
parser.add_argument('--cal_edge', action='store_true')
parser.add_argument('--test_unseen', action='store_true')
parser.add_argument('--train_seen', action='store_true') # action='store_true'
parser.add_argument('--loss_type', type=str,default='topnet')
parser.add_argument('--train_pcn', action='store_true')
parser.add_argument('--n_in_points', type=int, default=2048)
parser.add_argument('--n_gt_points', type=int, default=2048)
parser.add_argument('--n_out_points', type=int, default=2048)
parser.add_argument('--eval_path', default='data/shapenetcore_partanno_segmentation_benchmark_v0_test2_edge_200_5.000000.h5')
# data/topnet_dataset2019/val_edge.h5
# data/shapenetcore_partanno_segmentation_benchmark_v0_test2_edge_200_5.000000.h5
parser.add_argument('--gpu', type=str,default='1')
parser.add_argument('--run_name', default='test')
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--num_gpus', type=int, default=1)

parser.add_argument('--normalize_ratio', type=float,default=0.5)
parser.add_argument('--pre_trained_model', default='model_best.pth.tar')
parser.add_argument('--grid_size', type=int,default=32)

parser.add_argument('--augment', action='store_true') #### set to true if use scale etc...
parser.add_argument('--pc_augm_scale', default=1.0, type=float)
parser.add_argument('--pc_augm_rot', default=0, type=int,help='Training augmentation: Bool, random rotation around z-axis')
parser.add_argument('--pc_augm_mirror_prob', default=0.0, type=float,help='Training augmentation: Probability of mirroring about x or y axes')
parser.add_argument('--pc_augm_jitter', default=0, type=int)

parser.add_argument('--random_seed', type=int, default=42)

args = parser.parse_args()
if args.num_gpus==1:
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

chamfer_index=ops.chamferdist_index.chamferdist.ChamferDistance()

torch.manual_seed(args.random_seed)
np.random.seed(args.random_seed)
random.seed(args.random_seed)

def apply_net(net, dataloader, args):
    eval_loss=0

    for train_id,batch_data in enumerate(dataloader):
        inp, target, _, _,_ = batch_data
        inp = inp.cuda().transpose(2, 1).contiguous()
        target = target.cuda().transpose(2, 1).contiguous()

        pred, dens, dens_cls, reg, voxels,  pred_edge,reg_edge,dens_cls_edge,dens_edge = net(inp, n_points=args.n_out_points)

        dist1_fine, dist2_fine, _, _ = chamfer_index(pred.transpose(2, 1).contiguous(), target.transpose(2, 1).contiguous())
        if args.loss_type == 'pcn':
            out_cd = (torch.mean(torch.sqrt(dist2_fine), dim=1) + torch.mean(torch.sqrt(dist1_fine), dim=1)) / 2
        else:
            out_cd = torch.mean(dist2_fine, dim=1) + torch.mean(dist1_fine, dim=1)
        chamfer_loss = torch.sum(out_cd)#*pred.shape[0]
        eval_loss+=chamfer_loss.item()

    return eval_loss / len(dataloader.dataset), pred.detach().cpu()


def eval_net(net, dataloader, args):
    net.eval()
    with torch.no_grad():
        return apply_net(net, dataloader, args)

def test(args):
    if args.train_pcn:
        val_data = shapenet_dataset.PartDatasetPCN(args, path=args.eval_path, training=False)
    else:
        val_data = shapenet_dataset.PartDataset(args, path=args.eval_path, training=False)

    val_dataloader = torch.utils.data.DataLoader(val_data, num_workers=0,
                                                 batch_size=args.batch_size,
                                                 shuffle=False,
                                                 drop_last=False)

    net = models.GridAutoEncoderAdaIN(args,rnd_dim=2, adain_layer=3, ops=ops)
    if args.num_gpus > 1:
        net = torch.nn.DataParallel(net)  # ,device_ids=[0,1]
    net=net.cuda()
    net.apply(util.init_weights)

    if os.path.isfile(os.path.join('runs/{}/{}'.format(args.run_name,args.pre_trained_model))):
        checkpoint = torch.load(os.path.join('runs/{}/{}'.format(args.run_name,args.pre_trained_model)))
        net.load_state_dict(checkpoint['state_dict'])
        print("=> loaded checkpoint '{}'".format(args.pre_trained_model))

    args.training = False
    chamfer_avg, _ = eval_net(net, val_dataloader, args)
    print(chamfer_avg)

if __name__ == '__main__':
    test(args)
