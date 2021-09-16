import math
import torch
import torch.nn as nn
import util_edge as util
import numpy as np
import ghalton

class GridEncoder(nn.Module):
    def __init__(self,args, prep, grid_size,ops=None):
        super(self.__class__, self).__init__()

        self.grid_size = grid_size
        self.preprocessing = prep
        self.args=args
        self.ops=ops

    def initialize_grid_ball(self, x,grid_size):
        if x.dim() == 2:
            x = x.unsqueeze(0)
        assert (x.min() >= -self.args.normalize_ratio)
        assert (x.max() <= self.args.normalize_ratio)
        # bring vector into range -self.args.normalize_ratio - grid_size-self.args.normalize_ratio
        if self.args.normalize_ratio==0.5:
            reshaped = (x + self.args.normalize_ratio) * grid_size - self.args.normalize_ratio
        else:
            reshaped = (x + self.args.normalize_ratio)/2 * grid_size - self.args.normalize_ratio/2
        ind1 = reshaped.floor().clamp(0.0, grid_size - 1)
        ind2 = reshaped.ceil().clamp(0.0, grid_size - 1)
        ind = [torch.cat([ind1[:, 0, :].unsqueeze(1), ind1[:, 1, :].unsqueeze(1), ind1[:, 2, :].unsqueeze(1)], dim=1),
               torch.cat([ind1[:, 0, :].unsqueeze(1), ind1[:, 1, :].unsqueeze(1), ind2[:, 2, :].unsqueeze(1)], dim=1),
               torch.cat([ind1[:, 0, :].unsqueeze(1), ind2[:, 1, :].unsqueeze(1), ind1[:, 2, :].unsqueeze(1)], dim=1),
               torch.cat([ind1[:, 0, :].unsqueeze(1), ind2[:, 1, :].unsqueeze(1), ind2[:, 2, :].unsqueeze(1)], dim=1),
               torch.cat([ind2[:, 0, :].unsqueeze(1), ind1[:, 1, :].unsqueeze(1), ind1[:, 2, :].unsqueeze(1)], dim=1),
               torch.cat([ind2[:, 0, :].unsqueeze(1), ind1[:, 1, :].unsqueeze(1), ind2[:, 2, :].unsqueeze(1)], dim=1),
               torch.cat([ind2[:, 0, :].unsqueeze(1), ind2[:, 1, :].unsqueeze(1), ind1[:, 2, :].unsqueeze(1)], dim=1),
               torch.cat([ind2[:, 0, :].unsqueeze(1), ind2[:, 1, :].unsqueeze(1), ind2[:, 2, :].unsqueeze(1)], dim=1)]
        ind = torch.stack(ind, dim=-1).cuda()
        res = reshaped.unsqueeze(-1).repeat([1, 1, 1, 8]) - ind # B*3*2500*8
        ind = ind[:, 0, :, :] * grid_size * grid_size + ind[:, 1, :, :] * grid_size + ind[:, 2, :, :]
        ind = ind.long() # B*2500*8
        # binary weight to check wether point is in gridball
        dist = res.norm(dim=1).detach()
        weight = (dist < 0.87).float().detach()  # half the diagonal of a grid cube
        return res, weight, ind

    def propose_grid(self,res,indices,weight,grid_size,per_point_features=False):
        b, _, n,_ = res.size()
        c = res.shape[1]
        weight = weight.unsqueeze(1).expand_as(res) #  B*32*2500*8
        res = res * weight  # zero out weights of points outside of ball
        # sum up features of points inside ball
        x = torch.zeros(b, c, grid_size * grid_size * grid_size).to(res.device)
        count = torch.zeros(b, c, grid_size * grid_size * grid_size).to(x)

        indices = indices.view(b, -1).contiguous()
        indices.clamp_(0, grid_size ** 3)
        res = res.view(b, c, 8 * n).contiguous()
        weight = weight.view(b, c, 8 * n).contiguous()
        for i in range(b):
            x[i].index_add_(1, indices[i], res[i]) # dim, index, tensor
            count[i].index_add_(1, indices[i], weight[i])
        # number of points should have no effect

        count = torch.max(count, torch.tensor([1.0]).to(weight.device))
        x /= count
        x = x.view(b, -1, grid_size, grid_size, grid_size).contiguous()  # b x c x grid_size x grid_size x grid_size
        return x

    def forward(self, x):
        b, _, n = x.size()
        x_all=[]
        for i in range(int(np.log2(self.args.grid_size))):
            p1_idx = self.ops.pointnet2.pointnet2_utils.furthest_point_sample(x.transpose(1, 2).contiguous(),
                                                                              n // 2 ** i)
            x = self.ops.pointnet2.pointnet2_utils.gather_operation(x, p1_idx)
            grid_size = self.grid_size // 2 ** i
            res, weight, indices = self.initialize_grid_ball(x, grid_size)  # b x 3 x n x 8
            res = self.preprocessing[i](res)  # b x k x n x 8
            x1 = self.propose_grid(res, indices, weight, grid_size)
            x_all.append(x1)
        return x_all

class EdgePointCloudGenerator(nn.Module):
    def __init__(self, generator, rnd_dim=2, res=16,ops=None,normalize_ratio=0.5,args=None):
        super(self.__class__, self).__init__()
        self.base_dim = rnd_dim
        self.generator = generator
        self.ops=ops
        grid = util.meshgrid(res)
        if normalize_ratio == 0.5:
            self.o = (((grid + normalize_ratio) / res) - normalize_ratio).view(3, -1).contiguous()
        elif normalize_ratio == 1:
            self.o = (((grid + normalize_ratio / 2) * 2 / res) - normalize_ratio).view(3, -1).contiguous()
        self.s = res
        self.args=args

    def forward_fixed_pattern(self, x, dens, n,ratio,global_feature=None,local_feature=None):
        b, c, g, _, _ = x.shape
        grid_o = self.o.to(x.device)  # 3d meshgrid : 3* ()
        N = util.densSample(dens, n)
        N = N.view(b, -1).contiguous()
        x = x.view(b, c, -1).contiguous()

        b_rnd = torch.tensor(ghalton.GeneralizedHalton(2).get(n), dtype=torch.float32).t()
        b_rnd = b_rnd.to(x) * ratio - ratio / 2
        b_rnd = b_rnd.unsqueeze(0).repeat(b, 1, 1)

        N = N.view(-1).contiguous()
        x = x.permute(1, 0, 2).contiguous().view(c, -1).contiguous()

        ind = (N > 0).nonzero().squeeze(-1)
        x_ind = torch.repeat_interleave(x[:, ind], N[ind].long(), dim=-1)
        x_ind = x_ind.view(c, b, n).contiguous().permute(1, 0, 2).contiguous()
        b_inp = torch.cat([x_ind, b_rnd], dim=1)
        grid_o = grid_o.unsqueeze(0).repeat(b, 1, 1).permute(1, 0, 2).contiguous().view(3, -1).contiguous()
        o_ind = torch.repeat_interleave(grid_o[:, ind], N[ind].long(), dim=-1)
        o_ind = o_ind.view(3, b, n).contiguous().permute(1, 0, 2).contiguous()

        out = self.generator(b_inp)
        norm = out.norm(dim=1)
        reg = (norm - (math.sqrt(3) / (self.s))).clamp(0)  # twice the size needed to cover a gridcell
        return out + o_ind, reg

class PointCloudGenerator(nn.Module):
    def __init__(self, generator, rnd_dim=2, res=16,ops=None,normalize_ratio=0.5,args=None):
        super(self.__class__, self).__init__()

        self.base_dim = rnd_dim
        self.generator = generator
        self.ops=ops

        grid = util.meshgrid(res)
        if normalize_ratio == 0.5:
            self.o = (((grid + normalize_ratio) / res) - normalize_ratio).view(3, -1).contiguous()
        elif normalize_ratio == 1:
            self.o = (((grid + normalize_ratio/2)*2 / res) - normalize_ratio).view(3, -1).contiguous()
        self.s = res
        self.args=args

    def forward_fixed_pattern(self, x, dens, n,ratio,global_feature=None,local_feature=None):
        b, c, g, _, _ = x.shape
        grid_o = self.o.to(x.device)  # 3d meshgrid : 3* ()
        N = util.densSample(dens, n)
        N = N.view(b, -1).contiguous()
        x = x.view(b, c, -1).contiguous()

        b_rnd = torch.tensor(ghalton.GeneralizedHalton(2).get(n), dtype=torch.float32).t()
        b_rnd = b_rnd.to(x) * ratio - ratio / 2
        b_rnd = b_rnd.unsqueeze(0).repeat(b, 1, 1)

        N = N.view(-1).contiguous()
        x = x.permute(1, 0, 2).contiguous().view(c, -1).contiguous()

        ind = (N > 0).nonzero().squeeze(-1)
        x_ind = torch.repeat_interleave(x[:, ind], N[ind].long(), dim=-1)
        x_ind = x_ind.view(c, b, n).contiguous().permute(1, 0, 2).contiguous()
        b_inp = torch.cat([x_ind, b_rnd], dim=1)
        grid_o = grid_o.unsqueeze(0).repeat(b, 1, 1).permute(1, 0, 2).contiguous().view(3, -1).contiguous()
        o_ind = torch.repeat_interleave(grid_o[:, ind], N[ind].long(), dim=-1)
        o_ind = o_ind.view(3, b, n).contiguous().permute(1, 0, 2).contiguous()

        out = self.generator(b_inp)
        norm = out.norm(dim=1)
        reg = (norm - (math.sqrt(3) / (self.s))).clamp(0)  # twice the size needed to cover a gridcell
        return out + o_ind, reg

class AdaptiveDecoder(nn.Module):
    def __init__(self, decoder, n_classes=None,args=None, max_layer=None):
        super(self.__class__, self).__init__()
        self.args=args
        self.decoder = decoder
        self.norm_indices = []
        self.conditional = n_classes is not None

        if self.args.grid_size==32:
            self.norm_indices = [0, 3, 8, 12, 17, 21, 26, 30, 35, 39]
            self.slices=[128 * 2,128 * 2,
                         256 * 2,128 * 2,
                         192 * 2,64 * 2,
                         129 * 2,32 * 2,
                         65 * 2,62 * 2]
        elif self.args.grid_size==64:
            self.norm_indices=[0,3,8,12,17,21,26,30,35,39,44,48]
            self.slices=[128 * 2,128 * 2,
                        256 * 2,128 * 2,
                         192 * 2,128 * 2,
                         192 * 2,64 * 2,
                         97 * 2,32 * 2,
                         65 * 2,62 * 2]
        self.max_layer = len(self.norm_indices)

    def forward(self, w,x, contour_16=None,contour_32=None):
        size = 0
        if self.args.grid_size == 32:
            jj=[0,8,17,26,35]
        elif self.args.grid_size == 64:
            jj = [0, 8, 17, 26, 35,44]

        for i, l in enumerate(self.decoder):
            if i==0:
                x1 = x[int(np.log2(self.args.grid_size))-1-i]
            else:
                x1=torch.cat((x1,x[int(np.log2(self.args.grid_size))-1-i]),dim=1)

            if self.args.grid_size == 32:
                if i == 3:
                    x1 = torch.cat((x1, contour_16), dim=1)
                if i == 4:
                    x1 = torch.cat((x1, contour_32), dim=1)
            elif self.args.grid_size == 64:
                if i == 4:
                    x1 = torch.cat((x1, contour_16), dim=1)
                if i == 5:
                    x1 = torch.cat((x1, contour_32), dim=1)

            jjj = 0
            for j,ll in enumerate(l):
                x1=ll(x1)
                if j+jj[i] in self.norm_indices:
                    s = w[:, size:size + self.slices[i*2+jjj], None, None, None]
                    size += self.slices[i*2+jjj]

                    x1 = x1 * s[:, :self.slices[i*2+jjj] // 2]
                    x1 = x1 + s[:, self.slices[i*2+jjj] // 2:]
                    jjj += 1
        return x1