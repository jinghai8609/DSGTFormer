import torch
import torch.nn as nn
# from model.module.trans import Transformer as Transformer_s
# from model.module.trans_hypothesis import Transformer
import numpy as np
from einops import rearrange
from collections import OrderedDict
from torch.nn import functional as F
from torch.nn import init
import scipy.sparse as sp
from functools import partial
from timm.models.layers import DropPath
import math


class Model(nn.Module):
    def __init__(self, args):
        super().__init__()

        layers, d_hid, frames = args.layers, args.d_hid, args.frames
        num_joints_in, num_joints_out = args.n_joints, args.out_joints

        # layers, length,d_hid = layers, frames, d_hid
        # num_joints_in, num_joints_out = 17,17

        self.pose_emb = nn.Linear(2, d_hid, bias=False)
        self.gelu = nn.GELU()
        self.stcformer = STCFormer(layers, frames, num_joints_in, d_hid)
        self.regress_head = nn.Linear(d_hid, 3, bias=False)

    def forward(self, x):
        # b, t, s, c = x.shape  #batch,frame,joint,coordinate
        # dimension tranfer
        x = x.squeeze(-1)
        x = x.permute(0, 2, 3, 1)
       # print(f"输入 x 的形状: {x.shape}")
        x = self.pose_emb(x)
        x = self.gelu(x)
        # spatio-temporal correlation
        x = self.stcformer(x)
        # regression head
        x = self.regress_head(x)

        return x


CONNECTIONS = {10: [9], 9: [8, 10], 8: [7, 9], 14: [15, 8], 15: [16, 14], 11: [12, 8], 12: [13, 11],
               7: [0, 8], 0: [1, 7], 1: [2, 0], 2: [3, 1], 4: [5, 0], 5: [6, 4], 16: [15], 13: [12], 3: [2], 6: [5]}


class GCN(nn.Module):
    """
    空间图卷积网络（GCN）模块。

    :param dim_in: 输入通道维度
    :param dim_out: 输出通道维度
    :param num_nodes: 节点数量
    :param connections: 图边的空间连接关系（可选）
    """

    def __init__(self, dim_in, dim_out, num_nodes, connections=None):
        super().__init__()
        self.relu = nn.ReLU()
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.num_nodes = num_nodes
        self.connections = connections

        # 定义线性层
        self.U = nn.Linear(self.dim_in, self.dim_out)
        self.V = nn.Linear(self.dim_in, self.dim_out)
        # 定义批归一化层
        self.batch_norm = nn.BatchNorm1d(self.num_nodes)

        # 初始化 GCN 参数
        self._init_gcn()
        # 初始化空间邻接矩阵
        self.adj = self._init_spatial_adj()
        self.joint_att = nn.Sequential(
            nn.Linear(dim_out, max(4, dim_out // 4)),  # 保证最小维度为4
            nn.ReLU(),
            nn.Linear(max(4, dim_out // 4), 1),
            nn.Sigmoid()
        )

        self._init_gcn()
        self.adj = self._init_spatial_adj()

    def _init_gcn(self):
        """
        初始化 GCN 模块的权重和偏置。
        """
        # 初始化线性层 U 的权重
        self.U.weight.data.normal_(0, math.sqrt(2. / self.dim_in))
        # 初始化线性层 V 的权重
        self.V.weight.data.normal_(0, math.sqrt(2. / self.dim_in))
        # 初始化批归一化层的权重
        self.batch_norm.weight.data.fill_(1)
        # 初始化批归一化层的偏置
        self.batch_norm.bias.data.zero_()

    def _init_spatial_adj(self):
        """
        初始化空间邻接矩阵。

        :return: 空间邻接矩阵
        """
        # 初始化全零邻接矩阵
        adj = torch.zeros((self.num_nodes, self.num_nodes))
        # 使用传入的连接关系，若未传入则使用默认的 CONNECTIONS
        connections = self.connections if self.connections is not None else CONNECTIONS

        for i in range(self.num_nodes):
            connected_nodes = connections[i]
            for j in connected_nodes:
                adj[i, j] = 1
        return adj

    @staticmethod
    def normalize_digraph(adj):
        """
        对有向图的邻接矩阵进行归一化处理。

        :param adj: 邻接矩阵，形状为 [b, n, c]
        :return: 归一化后的邻接矩阵
        """
        b, n, c = adj.shape

        # 计算节点的度
        node_degrees = adj.detach().sum(dim=-1)
        # 计算度的逆平方根
        deg_inv_sqrt = node_degrees ** -0.5
        # 初始化单位矩阵
        norm_deg_matrix = torch.eye(n)
        # 获取邻接矩阵所在的设备
        dev = adj.get_device()
        if dev >= 0:
            # 将单位矩阵移动到相同的设备
            norm_deg_matrix = norm_deg_matrix.to(dev)
        # 计算归一化的度矩阵
        norm_deg_matrix = norm_deg_matrix.view(1, n, n) * deg_inv_sqrt.view(b, n, 1)
        # 计算归一化的邻接矩阵
        norm_adj = torch.bmm(torch.bmm(norm_deg_matrix, adj), norm_deg_matrix)

        return norm_adj

    def change_adj_device_to_cuda(self, adj):
        """
        将邻接矩阵移动到与线性层 V 相同的设备（如果是 CUDA 设备）。

        :param adj: 邻接矩阵
        :return: 移动后的邻接矩阵
        """
        # 获取线性层 V 的权重所在的设备
        dev = self.V.weight.get_device()
        if dev >= 0 and adj.get_device() < 0:
            # 如果线性层 V 在 CUDA 设备上，且邻接矩阵不在 CUDA 设备上，则移动邻接矩阵
            adj = adj.to(dev)
        return adj

    def forward(self, x):
        """
        前向传播方法。

        :param x: 输入张量，形状为 [B, T, J, C]
        :return: 输出张量
        """
        b, t, j, c = x.shape
        x = x.reshape(-1, j, c)
        adj = self.adj
        adj = self.change_adj_device_to_cuda(adj)
        adj = adj.repeat(b * t, 1, 1)

        norm_adj = self.normalize_digraph(adj)
        aggregate = norm_adj @ self.V(x)

        if self.dim_in == self.dim_out:
            x = self.relu(x + self.batch_norm(aggregate + self.U(x)))
        else:
            x = self.relu(self.batch_norm(aggregate + self.U(x)))

        x = x.reshape(-1, t, j, self.dim_out)
        att = self.joint_att(x)  # [B, T, J, 1]
        return x * att
        # return x# 输出形状为 [B, T, J, dim_out]


class TGCN(nn.Module):
    def __init__(self, dim_in, dim_out, num_nodes, connections=None):
        """
        :param dim_in: Channel input dimension
        :param dim_out: Channel output dimension
        :param num_nodes: Number of nodes
        :param connections: Spatial connections for graph edges (Optional)
        """
        super().__init__()

        self.relu = nn.ReLU()
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.num_nodes = num_nodes
        self.connections = connections

        self.U = nn.Linear(self.dim_in, self.dim_out)
        self.V = nn.Linear(self.dim_in, self.dim_out)
        self.batch_norm = nn.BatchNorm1d(self.num_nodes)
        self.joint_att = nn.Sequential(
            nn.Linear(dim_out, max(4, dim_out // 4)),  # 保证最小维度为4
            nn.ReLU(),
            nn.Linear(max(4, dim_out // 4), 1),
            nn.Sigmoid()
        )

        self._init_gcn()
        self.adj = self._init_spatial_adj()

    def _init_gcn(self):
        self.U.weight.data.normal_(0, math.sqrt(2. / self.dim_in))
        self.V.weight.data.normal_(0, math.sqrt(2. / self.dim_in))
        self.batch_norm.weight.data.fill_(1)
        self.batch_norm.bias.data.zero_()

    def _init_spatial_adj(self):
        adj = torch.zeros((self.num_nodes, self.num_nodes))
        connections = self.connections if self.connections is not None else CONNECTIONS

        for i in range(self.num_nodes):
            connected_nodes = connections[i]
            for j in connected_nodes:
                adj[i, j] = 1
        return adj

    @staticmethod
    def normalize_digraph(adj):
        b, n, c = adj.shape

        node_degrees = adj.detach().sum(dim=-1)
        deg_inv_sqrt = node_degrees ** -0.5
        norm_deg_matrix = torch.eye(n)
        dev = adj.get_device()
        if dev >= 0:
            norm_deg_matrix = norm_deg_matrix.to(dev)
        norm_deg_matrix = norm_deg_matrix.view(1, n, n) * deg_inv_sqrt.view(b, n, 1)
        norm_adj = torch.bmm(torch.bmm(norm_deg_matrix, adj), norm_deg_matrix)

        return norm_adj

    def change_adj_device_to_cuda(self, adj):
        dev = self.V.weight.get_device()
        if dev >= 0 and adj.get_device() < 0:
            adj = adj.to(dev)
        return adj

    def forward(self, x):
        """
        x: tensor with shape [B, T, J, C]
        """
        b, t, j, c = x.shape
        x = x.reshape(-1, j, c)
        adj = self.adj
        adj = self.change_adj_device_to_cuda(adj)
        adj = adj.repeat(b * t, 1, 1)

        norm_adj = self.normalize_digraph(adj)
        aggregate = norm_adj @ self.V(x)

        if self.dim_in == self.dim_out:
            x = self.relu(x + self.batch_norm(aggregate + self.U(x)))
        else:
            x = self.relu(self.batch_norm(aggregate + self.U(x)))

        x = x.reshape(-1, t, j, self.dim_out)
        att = self.joint_att(x)  # [B, T, J, 1]
        return x * att


class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16):
        super(ChannelAttention, self).__init__()
        # 平均池化和最大池化
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        # MLP层
        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction_ratio, kernel_size=1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_channels // reduction_ratio, in_channels, kernel_size=1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # 平均池化和最大池化的输出
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))

        # 将两个输出相加
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        # 卷积层
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=(kernel_size // 2), bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # 平均池化和最大池化
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)

        # 拼接后通过卷积层
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv(x)
        return self.sigmoid(x)


class CBAM(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16, kernel_size=7):
        super(CBAM, self).__init__()

        # 通道注意力和空间注意力
        self.channel_attention = ChannelAttention(in_channels, reduction_ratio=reduction_ratio)
        self.spatial_attention = SpatialAttention(kernel_size=kernel_size)

    def forward(self, x):
        # 通道注意力
        out = self.channel_attention(x) * x

        # 空间注意力
        out = self.spatial_attention(out) * out

        return out


class Cross_Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.linear_q = nn.Linear(dim, dim, bias=qkv_bias)
        self.linear_k = nn.Linear(dim, dim, bias=qkv_bias)
        self.linear_v = nn.Linear(dim, dim, bias=qkv_bias)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x_1, x_2):
        B, N, C = x_1.shape
        q = self.linear_q(x_1).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        k = self.linear_k(x_2).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        v = self.linear_v(x_1).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class CHI_Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_hidden_dim, qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm3_11 = norm_layer(dim)
        self.norm3_12 = norm_layer(dim)

        self.norm3_21 = norm_layer(dim)
        self.norm3_22 = norm_layer(dim)

        self.attn_1 = Cross_Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, \
                                      qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        self.attn_2 = Cross_Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, \
                                      qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)

        self.drop_path = DropPath(drop_path)

        self.norm2 = norm_layer(dim * 2)
        self.mlp = Mlp(in_features=dim * 2, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x_1, x_2):
        B, H, T, S, C = x_1.shape  # b h t s c
        x_1 = rearrange(x_1, 'b h t s c -> b  (t s) (c h)')
        x_2 = rearrange(x_2, 'b h t s c -> b  (t s) (c h)')
        x_1 = x_1 + self.drop_path(self.attn_1(self.norm3_11(x_2), self.norm3_12(x_1)))
        x_2 = x_2 + self.drop_path(self.attn_2(self.norm3_21(x_1), self.norm3_22(x_2)))
        x = torch.cat([x_1, x_2], dim=2)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        x_1 = x[:, :, :x.shape[2] // 2]
        x_2 = x[:, :, x.shape[2] // 2: x.shape[2]]
        x_1 = rearrange(x_1, 'b (t s) (c h) -> b h t s c', b=B, h=H, t=T, s=S, c=C)
        x_2 = rearrange(x_2, 'b (t s) (c h) -> b h t s c', b=B, h=H, t=T, s=S, c=C)
        return x_1, x_2


class STC_ATTENTION(nn.Module):
    def __init__(self, d_time, d_joint, d_coor, head=8):
        super().__init__()
        self.qkv = nn.Linear(d_coor, d_coor * 3)
        self.head = head
        self.layer_norm = nn.LayerNorm(d_coor)
        self.scale = (d_coor // 2) ** -0.5
        self.proj = nn.Linear(d_coor, d_coor)
        self.d_time = d_time
        self.d_joint = d_joint
        self.head = head
        depth = 3
        mlp_hidden_dim = 1024
        drop_rate = 0.1
        num_joints = 17
        drop_path_rate = 0.20
        attn_drop_rate = 0.
        qkv_bias = True
        qk_scale = None

        # sep1
        # print(d_coor)
        self.emb = nn.Embedding(5, d_coor // head // 2)
        self.part = torch.tensor([0, 1, 1, 1, 2, 2, 2, 0, 0, 0, 0, 3, 3, 3, 4, 4, 4]).long().cuda()
        norm_layer = partial(nn.LayerNorm, eps=1e-6)
        # sep2
        self.sep2_t = nn.Conv2d(d_coor // 2, d_coor // 2, kernel_size=3, stride=1, padding=1, groups=d_coor // 2)
        self.sep2_s = nn.Conv2d(d_coor // 2, d_coor // 2, kernel_size=3, stride=1, padding=1, groups=d_coor // 2)
        self.GCNS = GCN(d_coor // 2, d_coor // 2, num_nodes=d_joint)
        self.GCNT = TGCN(d_coor // 2, d_coor // 2, num_nodes=d_joint)

        self.CHI_Block = CHI_Block(dim=d_coor // 2, num_heads=head, mlp_hidden_dim=mlp_hidden_dim, qkv_bias=qkv_bias,
                                   qk_scale=qk_scale, drop=drop_rate, attn_drop=attn_drop_rate, drop_path=0.5,
                                   norm_layer=norm_layer)
        self.drop = DropPath(0.5)
        self.cbma = CBAM(in_channels=d_coor, reduction_ratio=16, kernel_size=7)

    def forward(self, input):
        b, t, s, c = input.shape

        h = input
        x = self.layer_norm(input)
        x = self.cbma(x.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
        qkv = self.qkv(x)  # b, t, s, c-> b, t, s, 3*c
        qkv = qkv.reshape(b, t, s, c, 3).permute(4, 0, 1, 2, 3)  # 3,b,t,s,c
        # space group and time group
        qkv_s, qkv_t = qkv.chunk(2, 4)  # [3,b,t,s,c//2],  [3,b,t,s,c//2]

        q_s, k_s, v_s = qkv_s[0], qkv_s[1], qkv_s[2]  # b,t,s,c//2
        q_t, k_t, v_t = qkv_t[0], qkv_t[1], qkv_t[2]  # b,t,s,c//2
        S = qkv_s[0, :, :, :, :]
        S1 = self.GCNT(S)
        x_s = self.GCNS(S1)
        # reshape for mat
        q_t = rearrange(q_t, 'b  t s (h c) -> (b h s) t c', h=self.head)  # b,t,s,c//2 -> b*h*s,t,c//2//h
        k_t = rearrange(k_t, 'b  t s (h c) -> (b h s) c t ', h=self.head)  # b,t,s,c//2->  b*h*s,c//2//h,t

        att_t = (q_t @ k_t) * self.scale  # b*h*s,t,t
        att_t = att_t.softmax(-1)  # b*h*s,t,t

        v_s = rearrange(v_s, 'b  t s c -> b c t s ')
        v_t = rearrange(v_t, 'b  t s c -> b c t s ')

        # sep2
        sep2_s = self.sep2_s(v_s)  # b,c//2,t,s
        sep2_t = self.sep2_t(v_t)  # b,c//2,t,s
        sep2_s = rearrange(sep2_s, 'b (h c) t s  -> (b h t) s c ', h=self.head)  # b*h*t,s,c//2//h
        sep2_t = rearrange(sep2_t, 'b (h c) t s  -> (b h s) t c ', h=self.head)  # b*h*s,t,c//2//h

        # sep1
        # v_s = rearrange(v_s, 'b c t s -> (b t ) s c')
        # v_t = rearrange(v_t, 'b c t s -> (b s ) t c')
        # print(lep_s.shape)
        sep_s = self.emb(self.part).unsqueeze(0)  # 1,s,c//2//h
        sep_t = self.emb(self.part).unsqueeze(0).unsqueeze(0).unsqueeze(0)  # 1,1,1,s,c//2//h

        # MSA

        v_t = rearrange(v_t, 'b (h c) t s  -> (b h s) t c ', h=self.head)  # b*h*s,t,c//2//h

        x_t = att_t @ v_t + sep2_t  # b*h,t,c//h                # b*h*s,t,c//2//h

        x_s = rearrange(x_s, 'b t s (c h) -> b h t s c ', h=self.head,
                        c=c // self.head // 2)  # b*h*t,s,c//h//2 -> b,h,t,s,c//h//2
        x_t = rearrange(x_t, '(b h s) t c -> b h t s c ', h=self.head, s=s)  # b*h*s,t,c//h//2 -> b,h,t,s,c//h//2
        # print(x_s.shape)
        x_t = x_t + 1e-9 * self.drop(sep_t)
        x_t, x_s = self.CHI_Block(x_t, x_s)
        x = torch.cat((x_s, x_t), -1)  # b,h,t,s,c//h
        x = rearrange(x, 'b h t s c -> b  t s (h c) ')  # b,t,s,c

        # projection and skip-connection
        x = self.proj(x)
        x = x + h
        return x


class STC_BLOCK(nn.Module):
    def __init__(self, d_time, d_joint, d_coor):
        super().__init__()

        self.layer_norm = nn.LayerNorm(d_coor)

        self.mlp = Mlp(d_coor, d_coor * 4, d_coor)

        self.stc_att = STC_ATTENTION(d_time, d_joint, d_coor)
        self.drop = DropPath(0.0)

    def forward(self, input):
        b, t, s, c = input.shape
        x = self.stc_att(input)
        x = x + self.drop(self.mlp(self.layer_norm(x)))

        return x


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.1):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features, bias=False)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features, bias=False)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class STCFormer(nn.Module):
    def __init__(self, num_block, d_time, d_joint, d_coor):
        super(STCFormer, self).__init__()

        self.num_block = num_block
        self.d_time = d_time
        self.d_joint = d_joint
        self.d_coor = d_coor

        self.stc_block = []
        for l in range(self.num_block):
            self.stc_block.append(STC_BLOCK(self.d_time, self.d_joint, self.d_coor))
        self.stc_block = nn.ModuleList(self.stc_block)

    def forward(self, input):
        # blocks layers
        for i in range(self.num_block):
            input = self.stc_block[i](input)
        # exit()
        return input



if __name__ == "__main__":
    # inputs = torch.rand(64, 351, 34)  # [btz, channel, T, H, W]
    # inputs = torch.rand(1, 64, 4, 112, 112) #[btz, channel, T, H, W]
    net = Model(layers=6, d_hid=256, frames=27)
    inputs = torch.rand([1, 27, 17, 2])
    output = net(inputs)
    print(output.size())
    from thop import profile

    # flops = 2*macs
    macs, params = profile(net, inputs=(inputs,))
    print(2 * macs)
    print(params)