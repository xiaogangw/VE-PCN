# from __future__ import print_function
import torch.utils.data as data
import torch
import numpy as np
import h5py
import transforms3d
import random
import math

class PartDataset(data.Dataset):
    def __init__(self, args, path,training=True):
        self.npoints = args.n_in_points
        self.args=args
        # with h5py.File(path, 'r') as f:
        f=h5py.File(path, 'r')
        self.complete_pcds = f['complete_pcds'][()]  # torch.from_numpy(f['complete_pcds'])#[()])
        if args.n_gt_points==2048:
            self.gt_pcds = self.complete_pcds  # torch.from_numpy(f['complete_pcds'])#[()])
        elif args.n_gt_points==16384:
            self.gt_pcds = f['complete_pcds_16384'][()]  # torch.from_numpy(f['complete_pcds'])#[()])
        self.labels = f['labels'][()] #torch.from_numpy(f['labels'])#[()])
        self.edge_labels = f['edge_labels'][()] #torch.from_numpy(f['edge_labels'])#[()])
        self.edge_pcds = f['complete_edge_pcds'][()]
        f.close()

        if args.train_seen:
            data_ids = np.isin(self.labels, np.asarray([1,2,3,4,5,6,8,9,10,11,14,15]))
            self.complete_pcds = self.complete_pcds[data_ids]
            self.gt_pcds = self.gt_pcds[data_ids]
            self.labels = self.labels[data_ids]
            self.edge_labels =self.edge_labels[data_ids]
            self.edge_pcds=self.edge_pcds[data_ids]

        if args.test_unseen:
            data_ids = np.isin(self.labels, np.asarray([0,7,12,13]))
            self.complete_pcds = self.complete_pcds[data_ids]
            self.gt_pcds = self.gt_pcds[data_ids]
            self.labels = self.labels[data_ids]
            self.edge_labels =self.edge_labels[data_ids]
            self.edge_pcds = self.edge_pcds[data_ids]

        self.model_ids = torch.tensor(range(self.complete_pcds.shape[0]))
        self.training=training

    def __getitem__(self, index):
        complete = self.complete_pcds[self.model_ids[index]]
        complete=torch.from_numpy(complete)#[()])

        partial,sel_ids = self.del_ratio_pts(self.args, complete, delete=self.args.remove_point_num, training=self.training)
        cls = self.labels[self.model_ids[index]]
        edge_labels = self.edge_labels[self.model_ids[index]]
        edge_labels = torch.from_numpy(edge_labels)

        edge = self.edge_pcds[self.model_ids[index]]
        edge = torch.from_numpy(edge)

        if self.args.n_gt_points == 16384:
            complete=self.gt_pcds[self.model_ids[index]]
            complete = torch.from_numpy(complete)

        return partial, complete, edge_labels, cls,edge

    def del_ratio_pts(self,args, batch_data, delete=512, training=True):
        if training:
            seed = batch_data[np.random.choice(batch_data.shape[0], 1)[0], :].unsqueeze(0)
        else:
            seed = batch_data[0, :].unsqueeze(0)
        seed = torch.repeat_interleave(seed, batch_data.shape[0], dim=0)
        diff = batch_data - seed
        dist_sq = torch.sum(diff * diff, 1)
        dist_sq_id = torch.argsort(dist_sq)  # ascending order
        sel_id = dist_sq_id[delete:]
        sel_id = self.pad_cloudN(sel_id, args.n_in_points)  # tmp_pt[choice, :]
        return batch_data[sel_id], sel_id

    def pad_cloudN(self,P, Nin):
        """ Pad or subsample 3D Point cloud to Nin number of points """
        N = P.shape[0]
        ii = np.random.choice(N, Nin - N)
        choice = np.concatenate([range(N), ii])
        choice = torch.from_numpy(choice)  # .to(P.device)
        P = P[choice]
        return P

    def __len__(self):
        return len(self.model_ids)



class PartDatasetPCN(data.Dataset):
    def __init__(self, args, path,training=True):
        self.npoints = args.n_in_points
        self.args=args
        f=h5py.File(path, 'r')
        self.incomplete_pcds = f['incomplete_pcds']#[()]
        self.complete_pcds = f['complete_pcds']#[()]  # torch.from_numpy(f['complete_pcds'])##[()])
        self.edge_pcds = f['complete_edge_pcds']#[()]
        self.gt_pcds = self.complete_pcds
        self.labels = f['labels']
        self.edge_labels = f['edge_labels']#[()]

        self.model_ids = torch.tensor(range(self.incomplete_pcds.shape[0]))
        if args.cal_edge:
            self.model_ids = self.model_ids[self.model_ids != 73479]
            self.model_ids = self.model_ids[self.model_ids != 73477]

        self.training=training

    def augment_cloud(self,Ps, args):
        """" Augmentation on XYZ and jittering of everything """
        M = transforms3d.zooms.zfdir2mat(1)
        if args.pc_augm_scale > 1:
            s = random.uniform(1 / args.pc_augm_scale, 1)
            M = np.dot(transforms3d.zooms.zfdir2mat(s), M)
        if args.pc_augm_rot > 0:
            angle = random.uniform(0, math.pi / 180 * args.pc_augm_rot)
            M = np.dot(transforms3d.axangles.axangle2mat([0, 1, 0], angle), M)  # y=upright assumption
        if args.pc_augm_mirror_prob > 0:  # mirroring x&z, not y
            if random.random() < args.pc_augm_mirror_prob / 2:
                M = np.dot(transforms3d.zooms.zfdir2mat(-1, [np.random.random() - 0.5, 0, np.random.random() - 0.5]), M)
        result = []
        for P in Ps:
            tmp = np.dot(P[:, :3], M.T)
            if args.pc_augm_jitter:
                sigma, clip = 0.01, 0.05  # https://github.com/charlesq34/pointnet/blob/master/provider.py#L74
                tmp = tmp + np.clip(sigma * np.random.randn(*P.shape), -1 * clip, clip).astype(np.float32)
            result.append(tmp)
        return result

    def __getitem__(self, index):
        complete=self.gt_pcds[self.model_ids[index]]
        edge = self.edge_pcds[self.model_ids[index]]
        partial = self.incomplete_pcds[self.model_ids[index]]

        if self.args.augment and self.training:
            complete, partial,edge = self.augment_cloud([complete, partial,edge], self.args)

        complete=torch.from_numpy(complete).float()#.clamp(-0.5, 0.5)
        edge = torch.from_numpy(edge).float()#.clamp(-0.5, 0.5)
        partial = torch.from_numpy(partial).float().clamp(-self.args.normalize_ratio, self.args.normalize_ratio)

        cls = self.labels[self.model_ids[index]]

        edge_labels = self.edge_labels[self.model_ids[index]]
        edge_labels = torch.from_numpy(edge_labels)

        return partial, complete, edge_labels, cls, edge

    def __len__(self):
        return len(self.model_ids)