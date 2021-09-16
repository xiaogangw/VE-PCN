import argparse
import os
import random
import numpy as np
import torch
import torch.nn.functional as F
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
parser.add_argument('--train_path', default='data/shapenetcore_partanno_segmentation_benchmark_v0_train2_edge_200_5.000000.h5')
# data/shapenet/train_pcn_16384_2048_edge.h5
# data/topnet_dataset2019/train_edge.h5
# data/shapenetcore_partanno_segmentation_benchmark_v0_train2_edge_200_5.000000.h5
parser.add_argument('--eval_path', default='data/shapenetcore_partanno_segmentation_benchmark_v0_test2_edge_200_5.000000.h5')
# data/shapenet/valid_data.h5
# data/topnet_dataset2019/val.h5
# data/shapenetcore_partanno_segmentation_benchmark_v0_test2_edge_200_5.000000.h5
parser.add_argument('--gpu', type=str,default='1')
parser.add_argument('--run_name', default='test')
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--num_gpus', type=int, default=1)

parser.add_argument('--pre_trained_model', default='model_best.pth.tar')
parser.add_argument('--normalize_ratio', type=float,default=0.5)
parser.add_argument('--density_weight', type=float, default=1e10)
parser.add_argument('--dense_cls_weight', type=float, default=100)
parser.add_argument('--p_norm', type=int, default=5)
parser.add_argument('--p_norm_weight', type=float, default=300)
parser.add_argument('--dist_regularize_weight', type=float, default=0.3)
parser.add_argument('--chamfer_weight', type=float, default=1e4)
parser.add_argument('--grid_size', type=int,default=32)

parser.add_argument('--augment', action='store_true') #### set to true if use scale etc...
parser.add_argument('--pc_augm_scale', default=1.0, type=float)
parser.add_argument('--pc_augm_rot', default=0, type=int,help='Training augmentation: Bool, random rotation around z-axis')
parser.add_argument('--pc_augm_mirror_prob', default=0.0, type=float,help='Training augmentation: Probability of mirroring about x or y axes')
parser.add_argument('--pc_augm_jitter', default=0, type=int)

parser.add_argument('--lr', type=float, default=0.0007)
parser.add_argument('--lr_decay', action='store_true')
parser.add_argument('--decay_epochs', type=int, default=40)
parser.add_argument('--lr_decay_rate', type=float, default=0.5)
parser.add_argument('--lr_min', type=float, default=0.000001)
parser.add_argument('--steps_per_print', type=int, default=200)

parser.add_argument('--n_epochs', type=int, default=200)
parser.add_argument('--best_loss',type=float,  default=1000)
parser.add_argument('--random_seed', type=int, default=42)

args = parser.parse_args()
if args.num_gpus==1:
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

chamfer_index=ops.chamferdist_index.chamferdist.ChamferDistance()

os.makedirs('runs/{}'.format(args.run_name),exist_ok=True)
LOG_FOUT = open(os.path.join('runs/{}'.format(args.run_name), 'log_train.txt'), 'a')
LOG_FOUT.write(str(args) + '\n')
def log_string(out_str):
    LOG_FOUT.write(out_str + '\n')
    LOG_FOUT.flush()
    print(out_str)

torch.manual_seed(args.random_seed)
np.random.seed(args.random_seed)
random.seed(args.random_seed)
start_epoch=0

def apply_net(epoch,net, dataloader, optimizer, args,log_string):
    chamfer_sum = 0
    p_norm_sum = 0
    density_sum = 0
    dense_cls_sum = 0
    reg_sum = 0
    eval_loss=0

    for train_id,batch_data in enumerate(dataloader):
        if optimizer is not None:
            inp, target, edge_labels, label,edge_pcds = batch_data
        else:
            inp, target, _, _, _ = batch_data

        inp = inp.cuda().transpose(2, 1).contiguous()
        target = target.cuda().transpose(2, 1).contiguous()

        if optimizer is not None:
            if args.cal_edge:
                edge_labels = edge_labels.cuda()
                edge_pcds = util.segCalc(target, edge_labels)
            else:
                edge_pcds = edge_pcds.cuda().transpose(2, 1).contiguous()

            targetdens = util.densCalc(target, args.grid_size, args.normalize_ratio)  # b*1*32*32*32
            mask = torch.squeeze(targetdens > 0, 1)
            pos_exp = mask.view(target.shape[0], -1).float().sum(dim=-1)
            neg_exp = targetdens.shape[-1] ** 3 - pos_exp
            pos_weight = neg_exp / pos_exp
            weight = torch.ones(mask.shape).float().cuda()

            edge_pcds_dens = util.densCalc(edge_pcds, args.grid_size, args.normalize_ratio)  # b*1*32*32*32
            edge_mask = torch.squeeze(edge_pcds_dens > 0, 1)
            pos_exp_edge = edge_mask.view(edge_pcds_dens.shape[0], -1).float().sum(dim=-1)
            neg_exp_edge = edge_pcds_dens.shape[-1] ** 3 - pos_exp_edge
            pos_weight_edge = neg_exp_edge / pos_exp_edge
            weight_edge = torch.ones(edge_mask.shape).float().cuda()

            for i in range(weight_edge.shape[0]):
                weight[i, mask[i]] = pos_weight[i]
                weight_edge[i, edge_mask[i]] = pos_weight_edge[i]

        pred, dens, dens_cls, reg, voxels,  pred_edge,reg_edge,dens_cls_edge,dens_edge = net(inp, n_points=args.n_out_points)

        assert pred.shape[-1]==args.n_out_points

        if optimizer is not None:
            dense_cls_loss = F.binary_cross_entropy_with_logits(dens_cls,
                                                                mask.float(),
                                                                weight=weight,
                                                                reduction='none').mean() * args.dense_cls_weight
            dense_cls_loss += F.binary_cross_entropy_with_logits(dens_cls_edge,
                                                                 edge_mask.float(),
                                                                 weight=weight_edge,
                                                                 reduction='none').mean() * args.dense_cls_weight

            if args.p_norm_weight >= 0:
                p_loss = util.dist_norm(pred, target, p=args.p_norm, chamferdist=chamfer_index).mean() * args.p_norm_weight
            else:
                p_loss = torch.zeros(1).to(pred)

            dist1_fine, dist2_fine, _, _ = chamfer_index(pred_edge.transpose(2, 1).contiguous(),
                                                         edge_pcds.transpose(2, 1).contiguous())
            chamfer_loss_edge = (torch.mean(torch.sqrt(dist2_fine)) + torch.mean(
                torch.sqrt(dist1_fine))) / 2 * args.chamfer_weight
            dist1_fine, dist2_fine, _, _ = chamfer_index(pred.transpose(2, 1).contiguous(),
                                                         target.transpose(2, 1).contiguous())
            chamfer_loss = (torch.mean(torch.sqrt(dist2_fine)) + torch.mean(
                torch.sqrt(dist1_fine))) / 2 * args.chamfer_weight \
                           + chamfer_loss_edge

            density_loss = F.mse_loss(dens, targetdens, reduction='mean') * args.density_weight #1e10
            density_loss += F.mse_loss(dens_edge, edge_pcds_dens, reduction='mean') * args.density_weight  # 1e10
            dist_regularization = torch.mean(torch.sum(reg,dim=1))*args.dist_regularize_weight
            dist_regularization += torch.mean(torch.sum(reg_edge, dim=1)) * args.dist_regularize_weight

            loss = chamfer_loss + density_loss + dense_cls_loss + dist_regularization + p_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            reg_sum += dist_regularization.item()
            chamfer_sum += chamfer_loss.item()
            p_norm_sum += p_loss.item()
            density_sum += density_loss.item()
            dense_cls_sum += dense_cls_loss.item()

        else:
            dist1_fine, dist2_fine, _, _ = chamfer_index(pred.transpose(2, 1).contiguous(),
                                                         target.transpose(2, 1).contiguous())
            if args.loss_type == 'pcn':
                out_cd = (torch.mean(torch.sqrt(dist2_fine), dim=1) + torch.mean(torch.sqrt(dist1_fine), dim=1)) / 2
            else:
                out_cd = torch.mean(dist2_fine, dim=1) + torch.mean(dist1_fine, dim=1)
            chamfer_loss = torch.sum(out_cd)  # *pred.shape[0]
            eval_loss += chamfer_loss.item()

        if train_id%args.steps_per_print==0 and optimizer is not None:
            log_string("(Training) Epoch:{} Iter:{} Chamfer Loss: {:.{prec}f} P Norm Loss: {:.{prec}f} Density Loss: {:.{prec}f}  "
                                "Density BCE {:.{prec}f} Regularization: {:.{prec}f}".
                       format(epoch,train_id,
                    chamfer_sum / (train_id+1), p_norm_sum / (train_id+1), density_sum / (train_id+1),
                    dense_cls_sum / (train_id+1), reg_sum / (train_id+1),prec=5))

    chamfer_avg = float(chamfer_sum) / (train_id+1)
    p_norm_avg = float(p_norm_sum) / (train_id+1)
    density_avg = float(density_sum) / (train_id+1)
    dense_cls_avg = dense_cls_sum / (train_id+1)
    reg_avg = reg_sum / (train_id+1)

    if not args.training:
        return eval_loss / len(dataloader.dataset), p_norm_avg, density_avg, dense_cls_avg, reg_avg, pred.detach().cpu()
    else:
        return chamfer_avg, p_norm_avg, density_avg, dense_cls_avg, reg_avg, pred.detach().cpu()

def train_net(epoch,net, dataloader, optimizer, args,log_string):
    net.train()
    return apply_net(epoch,net, dataloader, optimizer, args,log_string)


def eval_net(net, dataloader, args,log_string,epoch=0):
    net.eval()
    with torch.no_grad():
        return apply_net(epoch,net, dataloader, None, args,log_string)

def save_checkpoint(state, is_best, filename=None,epoch=0):
    if is_best:
        torch.save(state, os.path.join(filename,'model_best.pth.tar'))
    else:
        torch.save(state, os.path.join(filename,'checkpoint_%d.pth.tar'%(epoch)))

def train(args,log_string):
    global start_epoch
    if args.train_pcn:
        train_data = shapenet_dataset.PartDatasetPCN(args, path=args.train_path, training=True)
        val_data = shapenet_dataset.PartDatasetPCN(args, path=args.eval_path, training=False)
    else:
        train_data = shapenet_dataset.PartDataset(args, path=args.train_path, training=True)
        val_data = shapenet_dataset.PartDataset(args, path=args.eval_path, training=False)

    train_dataloader = torch.utils.data.DataLoader(train_data, num_workers=0,
                                                   batch_size=args.batch_size,
                                                   shuffle=True,
                                                   drop_last=False)

    val_dataloader = torch.utils.data.DataLoader(val_data, num_workers=0,
                                                 batch_size=args.batch_size+12,
                                                 shuffle=False,
                                                 drop_last=False)
    print(len(train_dataloader.dataset),len(val_dataloader.dataset))

    net = models.GridAutoEncoderAdaIN(args,rnd_dim=2, adain_layer=3, ops=ops)
    if args.num_gpus > 1:
        net = torch.nn.DataParallel(net)  # ,device_ids=[0,1]
    net=net.cuda()
    net.apply(util.init_weights)

    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr, amsgrad=True)

    if os.path.isfile(os.path.join('runs/{}/{}'.format(args.run_name,args.pre_trained_model))):
        checkpoint = torch.load(os.path.join('runs/{}/{}'.format(args.run_name,args.pre_trained_model)))
        start_epoch = checkpoint['epoch']
        net.load_state_dict(checkpoint['state_dict'])
        print("successfully load the pre-trained model")

    for epoch in range(start_epoch,args.n_epochs):

        if args.lr_decay and (epoch+1)%args.decay_epochs==0: # lrate / 10.0**((epoch+1) //50)
            cur_lr=max(args.lr/10.0**((epoch+1)//args.decay_epochs),args.lr_min)
            optimizer = torch.optim.Adam(net.parameters(), lr=cur_lr, amsgrad=True)

        # Training
        args.training=True
        chamfer_loss, p_norm_loss, density_loss, density_bce_loss, reg_loss, preds = \
            train_net(epoch,net, train_dataloader, optimizer, args,log_string)

        log_string("(Training) Epoch:{} Chamfer Loss: {:.{prec}f} P Norm Loss: {:.{prec}f} Density Loss: {:.{prec}f}  "
                   "Density BCE {:.{prec}f} Regularization: {:.{prec}f}".
            format(epoch,chamfer_loss, p_norm_loss, density_loss, density_bce_loss, reg_loss,prec=5))

        args.training = False
        val_chamfer_loss, val_p_norm_loss, val_density_loss, val_density_bce_loss, val_reg_loss, val_preds = \
            eval_net(net, val_dataloader, args,log_string,epoch=epoch)

        if epoch % 10==0 and epoch>0:
            save_checkpoint({
                'epoch': epoch,
                'state_dict': net.state_dict(),
                'optimizer': optimizer.state_dict()
            },False, filename=os.path.join('runs/{}'.format(args.run_name)))

        log_string(
            "(Validation) Epoch:{} Chamfer: {:.{prec}f} P Norm: {:.{prec}f} "
            "Density: {:.{prec}f}  Density BCE {:.{prec}f} Regularization: {:.{prec}f}".format(
                epoch,val_chamfer_loss, val_p_norm_loss, val_density_loss, val_density_bce_loss, val_reg_loss, prec=5))

    save_checkpoint({'epoch': args.n_epochs,
                    'state_dict': net.state_dict(),
                    'optimizer': optimizer.state_dict()
                    }, True, filename=os.path.join('runs/{}'.format(args.run_name)))

    log_string('\ndone')


if __name__ == '__main__':
    train(args,log_string)
