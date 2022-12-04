import torch
import h5py

def knn(x, k):
    inner = -2 * torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x ** 2, dim=1, keepdim=True)
    pairwise_distance = torch.sqrt(xx + inner + xx.transpose(2, 1))
    values, idx=torch.topk(pairwise_distance, k, dim=2, largest=False)
    return values, idx


def get_graph_feature(x, k=20, idx=None):
    batch_size = x.size(0)
    num_points = x.size(2)
    x = x.view(batch_size, -1, num_points)
    values, idx = knn(x, k=k+1)  # (batch_size, num_points, k)
    idx_base = torch.arange(0, batch_size).cuda().view(-1, 1, 1) * num_points
    idx = idx + idx_base#.type(torch.cuda.LongTensor)
    idx = idx.view(-1)
    _, num_dims, _ = x.size()
    x = x.transpose(2,1).contiguous()  # (batch_size, num_points, num_dims)  -> (batch_size*num_points, num_dims) #   batch_size * num_points * k + range(0, batch_size*num_points)
    neighbours = x.view(batch_size * num_points, -1)[idx, :]
    neighbours = neighbours.view(batch_size, num_points, k+1, num_dims)
    centroid = torch.mean(neighbours[:,:,1:,:], dim=2, keepdim=False)  # B*N*3
    return values, centroid, idx # B*N*(k+1)


def points_select(k,points,bs,edge_labels,w):
    for i in range(0, points.shape[0], bs):
        print(i)
        end_id = min(i + bs, points.shape[0])

        if points[i:end_id].shape[0] > 0:
            point = torch.from_numpy(points[i:end_id]).cuda()  # B*N*3
            point=point.permute(0,2,1).contiguous()

            values,centroid,idx=get_graph_feature(point, k=k, idx=None)
            point = point.transpose(2,1).contiguous()
            edge_labels[i:end_id, :] = (torch.sqrt(torch.sum((point - centroid) ** 2, 2)) >
                                                w * values[:, :, 1]).data.cpu().numpy()

    return edge_labels


def edge_detection(path='',k=100,w=1.8,bs=100):
    data_file = h5py.File(path, 'a')
    pcds = data_file['pcds']  # [()]
    edge_labels = data_file.create_dataset('edge_labels', (pcds.shape[0],pcds.shape[1]), 'i',
                                            compression='gzip', chunks=(1,pcds.shape[1]))

    edge_labels=points_select(k,pcds,bs,edge_labels,w)

    data_file.close()
    
if __name__ == '__main__':
    edge_detection(path='')
