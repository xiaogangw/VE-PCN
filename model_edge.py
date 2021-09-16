import torch
import torch.nn as nn
import torch.nn.functional as fn
import layer_edge as layer

class ResnetBlock(nn.Module):
    def __init__(self, dim, dilation=1):
        super(ResnetBlock, self).__init__()
        self.conv_block = nn.Sequential(
            nn.Conv3d(in_channels=dim, out_channels=dim, kernel_size=3, padding=2, dilation=dilation, bias=True),
            nn.InstanceNorm3d(dim, track_running_stats=False),
            nn.ReLU(True),
            nn.Conv3d(in_channels=dim, out_channels=dim, kernel_size=3, padding=1, dilation=1, bias=True),
            nn.InstanceNorm3d(dim, track_running_stats=False),
        )

    def forward(self, x):
        out = x + self.conv_block(x)
        return out


class EdgeGenerator(nn.Module):
    def __init__(self, residual_blocks=3):  # , init_weights=True
        super(EdgeGenerator, self).__init__()

        input_dim = 32
        self.encoder = nn.Sequential(
            nn.Conv3d(in_channels=input_dim, out_channels=64, kernel_size=4, stride=2, padding=1),  # 16*16*16
            nn.InstanceNorm3d(64, track_running_stats=False),
            nn.ReLU(True),
            nn.Conv3d(in_channels=64, out_channels=128, kernel_size=4, stride=2, padding=1),  # 8*8*8
            nn.InstanceNorm3d(128, track_running_stats=False),
            nn.ReLU(True),
        )

        blocks = []
        for _ in range(residual_blocks):
            block = ResnetBlock(128, 2)
            blocks.append(block)

        self.middle = nn.Sequential(*blocks)
        self.decoder11 = nn.Sequential(
            nn.ConvTranspose3d(in_channels=128, out_channels=64, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm3d(64, track_running_stats=False),
            nn.ReLU(True))
        self.decoder12 = nn.Conv3d(in_channels=64, out_channels=1, kernel_size=3, padding=1)
        self.decoder21 = nn.Sequential(
            nn.ConvTranspose3d(in_channels=64, out_channels=32, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm3d(32, track_running_stats=False),
            nn.ReLU(True),
        )
        self.decoder22 = nn.Conv3d(in_channels=32, out_channels=1, kernel_size=3, padding=1)

    def forward(self, x):
        x = self.encoder(x)
        x = self.middle(x)
        x1 = self.decoder11(x)
        x16 = self.decoder12(x1)
        x32 = self.decoder21(x1)
        x32 = self.decoder22(x32)
        return x16, x32


class GridAutoEncoderAdaIN(nn.Module):
    def __init__(self, args, rnd_dim=2, h_dim=62, dec_p=0, adain_layer=None, filled_cls=True, ops=None):
        super().__init__()

        self.grid_size = args.grid_size
        self.filled_cls = filled_cls
        self.args = args
        activation_func = torch.nn.ReLU()

        self.edge_generator = EdgeGenerator()
        self.decoder_edge = nn.Sequential(nn.Conv3d(1, 32, 3, padding=1, bias=True),
                                            nn.BatchNorm3d(32),
                                            activation_func,
                                            nn.Conv3d(32, 62, 3, padding=1, bias=True),
                                            nn.BatchNorm3d(62),
                                            activation_func,
                                            nn.Conv3d(62, 62, 3, padding=1, bias=True))
        self.density_estimator_edge = nn.Sequential(
                                                    nn.Conv3d(62, 16, 1, bias=True),
                                                    nn.BatchNorm3d(16),
                                                    activation_func,
                                                    nn.Conv3d(16, 8, 1, bias=True),
                                                    nn.BatchNorm3d(8),
                                                    activation_func,
                                                    nn.Conv3d(8, 4, 1, bias=True),
                                                    nn.BatchNorm3d(4),
                                                    activation_func,
                                                    nn.Conv3d(4, 2, 1),
                                                )
        self.generator_edge = layer.EdgePointCloudGenerator(
                                            nn.Sequential(nn.Conv1d(62 + rnd_dim, 64, 1),
                                                          activation_func,
                                                          nn.Conv1d(64, 64, 1),
                                                          activation_func,
                                                          nn.Conv1d(64, 32, 1),
                                                          activation_func,
                                                          nn.Conv1d(32, 32, 1),
                                                          activation_func,
                                                          nn.Conv1d(32, 16, 1),
                                                          activation_func,
                                                          nn.Conv1d(16, 16, 1),
                                                          activation_func,
                                                          nn.Conv1d(16, 8, 1),
                                                          activation_func,
                                                          nn.Conv1d(8, 3, 1)),
                                                        rnd_dim=rnd_dim, res=self.grid_size, ops=ops, normalize_ratio=self.args.normalize_ratio, args=args)

        self.grid_encoder = layer.GridEncoder(self.args,
                                              nn.ModuleList([
                                                  nn.Sequential(
                                                      nn.Conv2d(3, 16, 1, bias=True),
                                                      nn.BatchNorm2d(16),
                                                      activation_func,
                                                      nn.Conv2d(16, 32, 1, bias=True)),  # 32*32*32
                                                  nn.Sequential(
                                                      nn.Conv2d(3, 32, 1, bias=True),
                                                      nn.BatchNorm2d(32),
                                                      activation_func,
                                                      nn.Conv2d(32, 64, 1, bias=True)),  # 16*16*16
                                                  nn.Sequential(
                                                      nn.Conv2d(3, 32, 1, bias=True),
                                                      nn.BatchNorm2d(32),
                                                      activation_func,
                                                      nn.Conv2d(32, 64, 1, bias=True)),  # 8*8*8
                                                  nn.Sequential(
                                                      nn.Conv2d(3, 64, 1, bias=True),
                                                      nn.BatchNorm2d(64),
                                                      activation_func,
                                                      nn.Conv2d(64, 128, 1, bias=True)),  # 4*4*4
                                                  nn.Sequential(
                                                      nn.Conv2d(3, 64, 1, bias=True),
                                                      nn.BatchNorm2d(64),
                                                      activation_func,
                                                      nn.Conv2d(64, 128, 1, bias=True)),  # 2*2*2
                                              ]),
                                              self.grid_size, ops=ops)
        input_dim = 32
        self.encoder = nn.Sequential(
            nn.Conv3d(input_dim, 64, 3, padding=1, bias=True),
            nn.BatchNorm3d(64),
            activation_func,
            nn.Conv3d(64, 64, 3, padding=1, bias=True),
            nn.BatchNorm3d(64),
            activation_func,
            nn.Conv3d(64, 64, 3, padding=1, bias=True),
            nn.BatchNorm3d(64),
            activation_func,
            nn.MaxPool3d(2),  # 16

            nn.Conv3d(64, 128, 3, padding=1, bias=True),
            nn.BatchNorm3d(128),
            activation_func,
            nn.Conv3d(128, 128, 3, padding=1, bias=True),
            nn.BatchNorm3d(128),
            activation_func,
            nn.MaxPool3d(2),  # 8

            nn.Conv3d(128, 256, 3, padding=1, bias=True),
            nn.BatchNorm3d(256),
            activation_func,
            nn.Conv3d(256, 256, 3, padding=1, bias=True),
            nn.BatchNorm3d(256),
            activation_func,
            nn.MaxPool3d(2),  # 4

            nn.Conv3d(256, 512, 3, padding=1, bias=True),
            nn.BatchNorm3d(512),
            activation_func,
            nn.Conv3d(512, 512, 3, padding=1, bias=True),
            nn.BatchNorm3d(512),
            activation_func,
            nn.MaxPool3d(2),  # 2

            nn.Conv3d(512, 512, 3, padding=1, bias=True),
            nn.BatchNorm3d(512),
            activation_func,
            nn.Conv3d(512, 1024, 2, padding=0, bias=True),
            nn.BatchNorm3d(1024),
            activation_func,
        )

        self.decoder = layer.AdaptiveDecoder(nn.ModuleList([
            nn.ModuleList([
                nn.InstanceNorm3d(128),  # 0
                nn.Dropout3d(dec_p),
                nn.Conv3d(128, 128, 3, padding=1, bias=True),

                nn.InstanceNorm3d(128),  # 3
                activation_func,
                nn.Dropout3d(dec_p),
                nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True),  # 4
                nn.Conv3d(128, 128, 3, padding=1, bias=True)
            ]),

            nn.ModuleList([
                nn.InstanceNorm3d(256),  # 8
                activation_func,
                nn.Dropout3d(dec_p),
                nn.Conv3d(256, 128, 3, padding=1, bias=True),

                nn.InstanceNorm3d(128),  # 12
                activation_func,
                nn.Dropout3d(dec_p),
                nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True),  # 8
                nn.Conv3d(128, 128, 3, padding=1, bias=True)
            ]),

            nn.ModuleList([
                nn.InstanceNorm3d(128 + 64),  # 17
                activation_func,
                nn.Dropout3d(dec_p),
                nn.Conv3d(128 + 64, 64, 3, padding=1, bias=True),

                nn.InstanceNorm3d(64),  # 21
                activation_func,
                nn.Dropout3d(dec_p),
                nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True),  # 16
                nn.Conv3d(64, 64, 3, padding=1, bias=True)
            ]),

            nn.ModuleList([
                nn.InstanceNorm3d(64 + 64 + 1),  # 26
                activation_func,
                nn.Dropout3d(dec_p),
                nn.Conv3d(64 + 64 + 1, 32, 3, padding=1, bias=True),

                nn.InstanceNorm3d(32),  # 30
                activation_func,
                nn.Dropout3d(dec_p),
                nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True),  # 32
                nn.Conv3d(32, 32, 3, padding=1, bias=True)
            ]),

            nn.ModuleList([
                nn.InstanceNorm3d(32 + 32 + 1),  # 35
                activation_func,
                nn.Dropout3d(dec_p),
                nn.Conv3d(32 + 32 + 1, h_dim, 3, padding=1, bias=True),

                nn.InstanceNorm3d(h_dim),  # 39
                # nn.Dropout3d(dec_p)
            ])
        ]), args=self.args, max_layer=adain_layer)

        self.generator = layer.PointCloudGenerator(
            nn.Sequential(nn.Conv1d(h_dim + rnd_dim, 64, 1),
                          activation_func,
                          nn.Conv1d(64, 64, 1),
                          activation_func,
                          nn.Conv1d(64, 32, 1),
                          activation_func,
                          nn.Conv1d(32, 32, 1),
                          activation_func,
                          nn.Conv1d(32, 16, 1),
                          activation_func,
                          nn.Conv1d(16, 16, 1),
                          activation_func,
                          nn.Conv1d(16, 8, 1),
                          activation_func,
                          nn.Conv1d(8, 3, 1)),
            rnd_dim=rnd_dim, res=self.grid_size, ops=ops, normalize_ratio=self.args.normalize_ratio, args=args)

        self.density_estimator = nn.Sequential(
            nn.Conv3d(h_dim, 16, 1, bias=True),
            nn.BatchNorm3d(16),
            activation_func,
            nn.Conv3d(16, 8, 1, bias=True),
            nn.BatchNorm3d(8),
            activation_func,
            nn.Conv3d(8, 4, 1, bias=True),
            nn.BatchNorm3d(4),
            activation_func,
            nn.Conv3d(4, 2, 1),
        )

        self.adaptive = nn.Sequential(
            nn.Linear(1024, sum(self.decoder.slices))
        )


    def encode(self, x, partial_contour=None):
        b = x.shape[0]
        x = self.grid_encoder(x)  # different grid_size of x
        if partial_contour is not None:
            encoder_input = torch.cat((x[0], partial_contour.unsqueeze(1)), dim=1)
        else:
            encoder_input = x[0]
        z = self.encoder(encoder_input).view(b, -1).contiguous()
        return z, x

    def generate_points(self, w, x, n_points=5000, partial_contour=None):
        b = w.shape[0]
        if partial_contour is not None:
            contour_16, contour_32 = self.edge_generator(torch.cat((x[0], partial_contour.unsqueeze(1)), dim=1))
        else:
            contour_16, contour_32 = self.edge_generator(x[0])
        contour_16 = torch.sigmoid(contour_16)
        contour_32 = torch.sigmoid(contour_32)

        #### edge points generation ###
        edge_rec = self.decoder_edge(contour_32)
        est_edge = self.density_estimator_edge(edge_rec)
        dens_edge = fn.relu(est_edge[:, 0])
        dens_cls_edge = est_edge[:, 1].unsqueeze(1)
        dens_edge = dens_edge.view(b, -1).contiguous()
        dens_s = dens_edge.sum(-1).unsqueeze(1)
        mask = dens_s < 1e-12
        ones = torch.ones_like(dens_s)
        dens_s[mask] = ones[mask]
        dens_edge = dens_edge / dens_s
        dens_edge = dens_edge.view(b, 1, self.grid_size, self.grid_size, self.grid_size).contiguous()

        filled = torch.sigmoid(dens_cls_edge).round()
        dens_ = filled * dens_edge

        cloud_edge,reg_edge = self.generator_edge.forward_fixed_pattern(edge_rec, dens_, n_points, 2)
        #### edge points generation ###

        x_rec = self.decoder(w, x, contour_16=contour_16, contour_32=contour_32)
        est = self.density_estimator(x_rec)
        dens = fn.relu(est[:, 0])
        dens_cls = est[:, 1].unsqueeze(1)
        dens = dens.view(b, -1).contiguous()

        dens_s = dens.sum(-1).unsqueeze(1)
        mask = dens_s < 1e-12
        ones = torch.ones_like(dens_s)
        dens_s[mask] = ones[mask]
        dens = dens / dens_s
        dens = dens.view(b, 1, self.grid_size, self.grid_size, self.grid_size).contiguous()

        filled = torch.sigmoid(dens_cls).round()
        dens_ = filled * dens

        cloud, reg = self.generator.forward_fixed_pattern(x_rec, dens_, n_points, 2)

        return cloud, dens, torch.squeeze(dens_cls, 1), reg, filled, cloud_edge,reg_edge,torch.squeeze(dens_cls_edge, 1),dens_edge

    def decode(self, z, x, n_points=5000, partial_contour=None):
        b = z.shape[0]
        w = self.adaptive(z.view(b, -1).contiguous())
        return self.generate_points(w, x, n_points, partial_contour=partial_contour)

    def forward(self, x, n_points=5000, partial_contour=None):
        z, x = self.encode(x, partial_contour=partial_contour)
        return self.decode(z, x, n_points, partial_contour=partial_contour)
