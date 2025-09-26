import os
import glob
import torch
import random
import logging
import numpy as np
from tqdm import tqdm
import torch.nn as nn
import torch.utils.data
import torch.optim as optim
from common.opt import opts
from common.utils import *
from common.camera import get_uvd2xyz
from common.load_data_3dhp_mae import Fusion
from common.h36m_dataset import Human36mDataset
from model.block.refine import refine
from model.stcformer import Model
from model.stmo_pretrain import Model_MAE
import scipy.io as scio


# ---------------------------- 添加指标计算函数 ----------------------------
def compute_pck(pred, gt, threshold, limb_length=None, is_2d=False):
    """
    计算PCK（Percentage of Correct Keypoints）
    pred: 预测关节坐标 (batch, time, joints, 2/3)
    gt: 真实关节坐标 (batch, time, joints, 2/3)
    threshold: 判定正确的阈值（相对肢体长度的比例）
    limb_length: 参考肢体长度 (batch, time, 1)，用于归一化
    is_2d: 是否为2D坐标
    """
    # 计算关节距离
    dist = torch.norm(pred - gt, dim=-1)  # (batch, time, joints)
    joint_map = {
        'right_hip': 2,  # 右髋关节
        'left_hip': 3,  # 左髋关节
        'pelvis': 0,  # 骨盆
        'thorax': 8  # 胸部
    }

    # 计算参考肢体长度（3D用骨盆到胸部，2D用左右髋距离）
    if limb_length is None:
        if is_2d:
            # 2D参考：左右髋关节距离
            if joint_map['right_hip'] < pred.size(2) and joint_map['left_hip'] < pred.size(2):
                limb_length = torch.norm(
                    gt[..., joint_map['right_hip'], :] - gt[..., joint_map['left_hip'], :],
                    dim=-1, keepdim=True
                )
            else:
                limb_length = torch.ones_like(dist[..., :1])  # fallback
        else:
            # 3D参考：骨盆到胸部距离（MPI-INF-3DHP使用索引0和8）
            if joint_map['pelvis'] < pred.size(2) and joint_map['thorax'] < pred.size(2):
                limb_length = torch.norm(
                    gt[..., joint_map['pelvis'], :] - gt[..., joint_map['thorax'], :],
                    dim=-1, keepdim=True
                )
            else:
                limb_length = torch.mean(torch.norm(gt, dim=-1), dim=-1, keepdim=True)

    limb_length = torch.clamp(limb_length, min=1e-6)  # 避免除零
    normalized_dist = dist / limb_length  # 归一化距离
    correct = normalized_dist < threshold  # 判定是否正确
    return correct.float().mean().item()  # 平均正确率


def compute_auc(pred, gt, is_2d=False, thresholds=np.arange(0, 0.51, 0.01)):
    """
    计算AUC（PCK曲线下面积）
    thresholds: 阈值序列（默认0到0.5，步长0.01）
    """
    joint_map = {
        'right_hip': 2,
        'left_hip': 3,
        'pelvis': 0,
        'thorax': 8,
        'right_shoulder': 9,
        'left_shoulder': 10
    }
    # 计算参考肢体长度
    if is_2d:
        if joint_map['right_hip'] < pred.size(2) and joint_map['left_hip'] < pred.size(2):
            limb_length = torch.norm(
                gt[..., joint_map['right_hip'], :] - gt[..., joint_map['left_hip'], :],
                dim=-1, keepdim=True
            )
        else:
            limb_length = torch.ones_like(pred[..., :1, :]).norm(dim=-1)
    else:
        if (joint_map['right_shoulder'] < pred.size(2) and
                joint_map['left_hip'] < pred.size(2)):
            limb_length = torch.norm(
                gt[..., joint_map['right_shoulder'], :] - gt[..., joint_map['left_hip'], :],
                dim=-1, keepdim=True
            )
        else:
            limb_length = torch.norm(
                gt[..., joint_map['pelvis'], :] - gt[..., joint_map['thorax'], :],
                dim=-1, keepdim=True
            )

    # 计算各阈值下的PCK
    pck_values = []
    for threshold in thresholds:
        pck = compute_pck(pred, gt, threshold, limb_length, is_2d)
        pck_values.append(pck)

    return np.mean(pck_values)  # AUC为PCK均值


# ---------------------------- 主代码逻辑 ----------------------------
opt = opts().parse()
os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu


def train(opt, actions, train_loader, model, optimizer, epoch):
    return step('train', opt, actions, train_loader, model, optimizer, epoch)


def val(opt, actions, val_loader, model):
    with torch.no_grad():
        return step('test', opt, actions, val_loader, model)


def step(split, opt, actions, dataLoader, model, optimizer=None, epoch=None):
    model_trans = model['trans']
    model_refine = model['refine']
    model_MAE = model['MAE']

    if split == 'train':
        model_trans.train()
        model_refine.train()
        model_MAE.train()
    else:
        model_trans.eval()
        model_refine.eval()
        model_MAE.eval()

    # 初始化累计器
    loss_all = {'loss': AccumLoss()}
    error_sum = AccumLoss()
    error_sum_test = AccumLoss()
    pck_sum = AccumLoss()  # 新增：PCK累计
    auc_sum = AccumLoss()  # 新增：AUC累计

    action_error_sum = define_error_list(actions)
    action_error_sum_post_out = define_error_list(actions)
    action_error_sum_MAE = define_error_list(actions)

    joints_left = [5, 6, 7, 11, 12, 13]
    joints_right = [2, 3, 4, 8, 9, 10]
    data_inference = {}

    for i, data in enumerate(tqdm(dataLoader, 0)):
        if opt.MAE:
            # MAE模式（2D关节预测）
            if split == "train":
                batch_cam, input_2D, seq, subject, scale, bb_box, cam_ind = data
            else:
                batch_cam, input_2D, seq, scale, bb_box = data

            [input_2D, batch_cam, scale, bb_box] = get_varialbe(split, [input_2D, batch_cam, scale, bb_box])
            N = input_2D.size(0)
            f = opt.frames

            # 掩码生成
            mask_num = int(f * opt.temporal_mask_rate)
            mask = np.hstack([np.zeros(f - mask_num), np.ones(mask_num)]).flatten()
            np.random.shuffle(mask)
            mask = torch.from_numpy(mask).to(torch.bool).cuda()

            spatial_mask = np.zeros((f, 17), dtype=bool)
            for k in range(f):
                ran = random.sample(range(0, 16), opt.spatial_mask_num)
                spatial_mask[k, ran] = True

            # 数据增强与模型预测
            if opt.test_augmentation and split == 'test':
                input_2D, output_2D = input_augmentation_MAE(input_2D, model_MAE, joints_left, joints_right, mask,
                                                             spatial_mask)
            else:
                input_2D = input_2D.view(N, -1, opt.n_joints, opt.in_channels, 1).permute(0, 3, 1, 2, 4).type(
                    torch.cuda.FloatTensor)
                output_2D = model_MAE(input_2D, mask, spatial_mask)

            # 维度调整
            input_2D = input_2D.permute(0, 2, 3, 1, 4).view(N, -1, opt.n_joints, 2)
            output_2D = output_2D.permute(0, 2, 3, 1, 4).view(N, -1, opt.n_joints, 2)
            loss = mpjpe_cal(output_2D, torch.cat((input_2D[:, ~mask], input_2D[:, mask]), dim=1))

        else:
            # 非MAE模式（3D关节预测）
            if split == "train":
                batch_cam, gt_3D, input_2D, seq, subject, scale, bb_box, cam_ind = data
            else:
                batch_cam, gt_3D, input_2D, seq, scale, bb_box = data

            [input_2D, gt_3D, batch_cam, scale, bb_box] = get_varialbe(split,
                                                                       [input_2D, gt_3D, batch_cam, scale, bb_box])
            N = input_2D.size(0)

            out_target = gt_3D.clone().view(N, -1, opt.out_joints, opt.out_channels)
            out_target[:, :, 14] = 0  # 骨盆置为原点
            gt_3D = gt_3D.view(N, -1, opt.out_joints, opt.out_channels).type(torch.cuda.FloatTensor)

            # 处理单帧/多帧目标
            out_target_single = out_target[:, opt.pad].unsqueeze(1) if out_target.size(1) > 1 else out_target
            gt_3D_single = gt_3D[:, opt.pad].unsqueeze(1) if gt_3D.size(1) > 1 else gt_3D

            # 模型预测
            if opt.test_augmentation and split == 'test':
                input_2D, output_3D = input_augmentation(input_2D, model_trans, joints_left, joints_right)
            else:
                input_2D = input_2D.view(N, -1, opt.n_joints, opt.in_channels, 1).permute(0, 3, 1, 2, 4).type(
                    torch.cuda.FloatTensor)
                output_3D = model_trans(input_2D)

            # 尺度调整
            output_3D = output_3D * scale.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).repeat(1, output_3D.size(1),
                                                                                           opt.out_joints,
                                                                                           opt.out_channels)
            output_3D_single = output_3D[:, opt.pad].unsqueeze(1)

            # 确定预测输出
            pred_out = output_3D if split == 'train' else output_3D_single
            input_2D = input_2D.permute(0, 2, 3, 1, 4).view(N, -1, opt.n_joints, 2)

            # 精炼模型
            if opt.refine:
                pred_uv = input_2D
                uvd = torch.cat((pred_uv[:, opt.pad, :, :].unsqueeze(1), output_3D_single[:, :, :, 2].unsqueeze(-1)),
                                -1)
                xyz = get_uvd2xyz(uvd, gt_3D_single, batch_cam)
                xyz[:, :, 0, :] = 0
                post_out = model_refine(output_3D_single, xyz)
                loss = mpjpe_cal(post_out, out_target_single)
            else:
                loss = mpjpe_cal(pred_out, out_target)

        # 累计损失
        loss_all['loss'].update(loss.detach().cpu().numpy() * N, N)

        # 训练阶段更新参数
        if split == 'train':
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if not opt.MAE:
                # 计算训练集MPJPE
                if opt.refine:
                    post_out[:, :, 14, :] = 0
                    joint_error = mpjpe_cal(post_out, out_target_single).item()
                else:
                    pred_out[:, :, 14, :] = 0
                    joint_error = mpjpe_cal(pred_out, out_target).item()
                error_sum.update(joint_error * N, N)

        # 测试阶段计算指标
        elif split == 'test':
            if opt.MAE:
                # MAE模式（2D）：计算MPJPE、PCK、AUC
                gt_2d = torch.cat((input_2D[:, ~mask], input_2D[:, mask]), dim=1)
                pred_2d = output_2D
                joint_error_test = mpjpe_cal(pred_2d, gt_2d).item()

                # 计算2D PCK（阈值0.1）和AUC
                pck = compute_pck(pred_2d, gt_2d, threshold=0.1, is_2d=True)
                auc = compute_auc(pred_2d, gt_2d, is_2d=True)

            else:
                # 非MAE模式（3D）：计算MPJPE、PCK、AUC
                pred_out[:, :, 14, :] = 0
                joint_error_test = mpjpe_cal(pred_out, out_target).item()

                # 计算3D PCK（阈值0.1）和AUC
                pck = compute_pck(pred_out, out_target, threshold=0.1, is_2d=False)
                auc = compute_auc(pred_out, out_target, is_2d=False)

            # 累计测试指标
            error_sum_test.update(joint_error_test * N, N)
            pck_sum.update(pck * N, N)
            auc_sum.update(auc * N, N)

            # 保存推理结果
            if opt.train == 0:
                for seq_cnt in range(len(seq)):
                    seq_name = seq[seq_cnt]
                    if seq_name in data_inference:
                        data_inference[seq_name] = np.concatenate(
                            (data_inference[seq_name], pred_out[seq_cnt].permute(2, 1, 0).cpu().numpy()), axis=2)
                    else:
                        data_inference[seq_name] = pred_out[seq_cnt].permute(2, 1, 0).cpu().numpy()

    # 输出结果
    if split == 'train':
        return (loss_all['loss'].avg, error_sum.avg) if not opt.MAE else loss_all['loss'].avg * 1000
    elif split == 'test':
        if opt.train == 0:
            for seq_cnt in range(len(seq)):
                seq_name = seq[seq_cnt]
                if seq_name in data_inference:
                    data_inference[seq_name] = np.concatenate(
                        (data_inference[seq_name], pred_out[seq_cnt].permute(2, 1, 0).cpu().numpy()), axis=2)
                else:
                    data_inference[seq_name] = pred_out[seq_cnt].permute(2, 1, 0).cpu().numpy()

        # 返回MPJPE、PCK、AUC
        return error_sum_test.avg, pck_sum.avg, auc_sum.avg


# ---------------------------- 数据增强函数 ----------------------------
def input_augmentation_MAE(input_2D, model_trans, joints_left, joints_right, mask, spatial_mask=None):
    N, _, T, J, C = input_2D.shape
    input_2D_flip = input_2D[:, 1].view(N, T, J, C, 1).permute(0, 3, 1, 2, 4)
    input_2D_non_flip = input_2D[:, 0].view(N, T, J, C, 1).permute(0, 3, 1, 2, 4)

    output_2D_flip = model_trans(input_2D_flip, mask, spatial_mask)
    output_2D_flip[:, :, :, 0] *= -1
    output_2D_flip[:, :, :, joints_left + joints_right] = output_2D_flip[:, :, :, joints_right + joints_left]

    output_2D_non_flip = model_trans(input_2D_non_flip, mask, spatial_mask)
    output_2D = (output_2D_non_flip + output_2D_flip) / 2
    input_2D = input_2D_non_flip
    return input_2D, output_2D


def input_augmentation(input_2D, model_trans, joints_left, joints_right):
    N, _, T, J, C = input_2D.shape
    input_2D_flip = input_2D[:, 1].view(N, T, J, C, 1).permute(0, 3, 1, 2, 4)
    input_2D_non_flip = input_2D[:, 0].view(N, T, J, C, 1).permute(0, 3, 1, 2, 4)

    output_3D_flip = model_trans(input_2D_flip)
    output_3D_flip[:, :, :, 0] *= -1
    output_3D_flip[:, :, joints_left + joints_right] = output_3D_flip[:, :, joints_right + joints_left]

    output_3D_non_flip = model_trans(input_2D_non_flip)
    output_3D = (output_3D_non_flip + output_3D_flip) / 2
    input_2D = input_2D_non_flip
    return input_2D, output_3D


# ---------------------------- 主程序入口 ----------------------------
if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    opt.manualSeed = 1
    random.seed(opt.manualSeed)
    torch.manual_seed(opt.manualSeed)
    np.random.seed(opt.manualSeed)
    torch.cuda.manual_seed_all(opt.manualSeed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    if opt.train == 1:
        logging.basicConfig(format='%(asctime)s %(message)s', datefmt='%Y/%m/%d %H:%M:%S',
                            filename=os.path.join(opt.checkpoint, 'train.log'), level=logging.INFO)

    root_path = opt.root_path
    actions = define_actions(opt.actions)

    # 加载数据
    if opt.train:
        train_data = Fusion(opt=opt, train=True, root_path=root_path, MAE=opt.MAE)
        train_dataloader = torch.utils.data.DataLoader(
            train_data, batch_size=opt.batchSize, shuffle=True,
            num_workers=int(opt.workers), pin_memory=True)
    if opt.test:
        test_data = Fusion(opt=opt, train=False, root_path=root_path, MAE=opt.MAE)
        test_dataloader = torch.utils.data.DataLoader(
            test_data, batch_size=opt.batchSize, shuffle=False,
            num_workers=int(opt.workers), pin_memory=True)

    # 初始化模型
    opt.out_joints = 17
    model = {
        'trans': nn.DataParallel(Model(opt)).cuda(),
        'refine': nn.DataParallel(refine(opt)).cuda(),
        'MAE': nn.DataParallel(Model_MAE(opt)).cuda()
    }

    # 打印模型参数数量
    model_params = sum(p.numel() for p in model['trans'].parameters())
    print('INFO: Trainable parameter count:', model_params)

    # 加载预训练模型
    if opt.MAE_reload == 1:
        model_dict = model['trans'].state_dict()
        pre_dict = torch.load(opt.previous_dir)
        state_dict = {k: v for k, v in pre_dict.items() if k in model_dict.keys()}
        model_dict.update(state_dict)
        model['trans'].load_state_dict(model_dict)

    if opt.reload == 1:
        model_dict = model['trans'].state_dict()
        pre_dict = torch.load(opt.previous_dir)
        model_dict.update({k: pre_dict[k] for k in model_dict.keys()})
        model['trans'].load_state_dict(model_dict)

    if opt.refine_reload == 1:
        refine_dict = model['refine'].state_dict()
        pre_dict_refine = torch.load(opt.previous_refine_name)
        refine_dict.update({k: pre_dict_refine[k] for k in refine_dict.keys()})
        model['refine'].load_state_dict(refine_dict)

    # 优化器
    all_param = list(model['trans'].parameters()) + list(model['refine'].parameters()) + list(model['MAE'].parameters())
    optimizer_all = optim.Adam(all_param, lr=opt.lr, amsgrad=True)

    # 训练与测试循环
    for epoch in range(1, opt.nepoch):
        if opt.train == 1:
            if not opt.MAE:
                loss, mpjpe = train(opt, actions, train_dataloader, model, optimizer_all, epoch)
            else:
                loss = train(opt, actions, train_dataloader, model, optimizer_all, epoch)

        if opt.test == 1:
            # 获取测试指标：MPJPE (p1)、PCK、AUC
            p1, pck, auc = val(opt, actions, test_dataloader, model)
            data_threshold = p1

            # 保存最佳模型
            if opt.train and data_threshold < opt.previous_best_threshold:
                if opt.MAE:
                    opt.previous_name = save_model(opt.previous_name, opt.checkpoint, epoch, data_threshold,
                                                   model['MAE'], 'MAE')
                else:
                    opt.previous_name = save_model(opt.previous_name, opt.checkpoint, epoch, data_threshold,
                                                   model['trans'], 'no_refine')
                    if opt.refine:
                        opt.previous_refine_name = save_model(opt.previous_refine_name, opt.checkpoint, epoch,
                                                              data_threshold, model['refine'], 'refine')
                opt.previous_best_threshold = data_threshold

            # 输出结果
            if opt.train == 0:
                print(f'p1: {p1:.2f}, PCK: {pck:.4f}, AUC: {auc:.4f}')
                break
            else:
                # 日志与控制台输出（包含PCK和AUC）
                if opt.MAE:
                    log_str = f'epoch: {epoch}, lr: {opt.lr:.7f}, loss: {loss:.4f}, ' \
                              f'p1: {p1:.2f}, PCK: {pck:.4f}, AUC: {auc:.4f}'
                else:
                    log_str = f'epoch: {epoch}, lr: {opt.lr:.7f}, loss: {loss:.4f}, ' \
                              f'MPJPE: {mpjpe:.2f}, p1: {p1:.2f}, PCK: {pck:.4f}, AUC: {auc:.4f}'
                logging.info(log_str)
                print(log_str)

        # 学习率衰减
        if epoch % opt.large_decay_epoch == 0:
            opt.lr *= opt.lr_decay_large
        else:
            opt.lr *= opt.lr_decay
        for param_group in optimizer_all.param_groups:
            param_group['lr'] = opt.lr