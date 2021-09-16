import numpy as np
import torch

def densSample(d, n):
    b, _, g, _, _ = d.shape
    out = []
    for i in range(b):
        N = torch.zeros(g, g, g).to(d.device)
        add = torch.ones([n]).to(d.device)
        d_ = d[i, 0, :, :, :].view(-1).contiguous()
        d_sum = d_.sum().item()
        assert (np.isfinite(d_sum))
        if d_sum < 1e-12:
            d_ = torch.ones_like(d_)
        ind = torch.multinomial(d_, n, replacement=True)
        N.put_(ind, add, accumulate=True)
        out.append(N.int())
    out = torch.stack(out, dim=0)
    return out

def densCalc(x, grid_size,normalize_ratio=0.5,args=None):
    n = x.size(2)
    if normalize_ratio == 0.5:
        ind = ((x + normalize_ratio) * grid_size - normalize_ratio).round().clamp(0, grid_size - 1).long()
    elif normalize_ratio == 1:
        ind = ((x + normalize_ratio) / 2 * grid_size - normalize_ratio / 2).round().clamp(0, grid_size - 1).long()
    resf1 = torch.zeros((x.size(0),grid_size ** 3)).to(x)
    ind=ind[:,2, :] + (grid_size * ind[:,1, :]) + (grid_size * grid_size * ind[:,0, :])
    values=torch.ones(ind.size()).to(x)
    resf1.scatter_add_(1, ind, values)
    resf1=resf1.reshape((-1, 1, grid_size, grid_size, grid_size)) / n
    return resf1


def segCalc(x,edge_labels):
    out_edge_points=[]
    for i in range(x.size(0)):
        inp = x[i, :, :]
        ids=(edge_labels[i] == 1).nonzero()
        ids1 = np.random.choice(ids.cpu().numpy()[:,0], 2048 - ids.shape[0])
        ids1=torch.from_numpy(ids1).cuda()
        out_edge_points.append(torch.cat((inp[:, ids1], inp[:, ids[:,0]]), dim=-1))
    return torch.stack(out_edge_points,dim=0)

def meshgrid(s):
    r = torch.arange(s).float()
    x = r[:, None, None].expand(s, s, s)
    y = r[None, :, None].expand(s, s, s)
    z = r[None, None, :].expand(s, s, s)
    return torch.stack([x, y, z], 0)

def init_weights(m): # ,init_type='normal', gain=0.02
    '''
    initialize network's weights
    init_type: normal | xavier | kaiming | orthogonal
    https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/9451e70673400885567d08a9e97ade2524c700d0/models/networks.py#L39
    '''

    # def init_func(m):
    classname = m.__class__.__name__
    if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
        # if init_type == 'normal':
        #     torch.nn.init.normal_(m.weight.data, 0.0, gain)
        # elif init_type == 'xavier':
        #     torch.nn.init.xavier_normal_(m.weight.data, gain=gain)
        # elif init_type == 'kaiming':
        #     torch.nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
        # elif init_type == 'orthogonal':
        #     torch.nn.init.orthogonal_(m.weight.data, gain=gain)
        torch.nn.init.xavier_uniform_(m.weight)

        if hasattr(m, 'bias') and m.bias is not None:
            torch.nn.init.constant_(m.bias.data, 0.0)

    elif classname.find('BatchNorm') != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)

def dist_norm(x, y, p=2, chamferdist=None):
    _,_,dist1, dist2 = chamferdist(x.transpose(2, 1).contiguous(),y.transpose(2, 1).contiguous()) # nearest neighbor from 1->2; 2->1
    pc1 = torch.gather(y, dim=2, index=dist1.long()[:, :].unsqueeze(1).repeat(1, 3, 1))
    pc1 = (x - pc1).norm(dim=1, p=p)  # .norm(p=p, dim=-1)
    pc2 = torch.gather(x, dim=2, index=dist2.long()[:, :].unsqueeze(1).repeat(1, 3, 1))
    pc2 = (y - pc2).norm(dim=1, p=p)  # .norm(p=p, dim=-1)
    result2=pc1.norm(p=p, dim=-1) + pc2.norm(p=p, dim=-1)
    return result2