import os
import sys
import glob
import json
import argparse
import numpy as np
from PIL import Image
from tqdm import tqdm
from scipy.ndimage.filters import maximum_filter
from shapely.geometry import Polygon

import torch
import torch.nn as nn
import torch.nn.functional as F

from model import HorizonNet
from dataset import visualize_a_data
from misc import post_proc, panostretch, utils


def find_N_peaks(signal, r=29, min_v=0.05, N=None):
    max_v = maximum_filter(signal, size=r, mode='wrap')
    pk_loc = np.where(max_v == signal)[0]
    pk_loc = pk_loc[signal[pk_loc] > min_v]
    if N is not None:
        order = np.argsort(-signal[pk_loc])
        pk_loc = pk_loc[order[:N]]
        pk_loc = pk_loc[np.argsort(pk_loc)]
    return pk_loc, signal[pk_loc]


def augment(x_img, flip, rotate):
    x_img = x_img.numpy()
    aug_type = ['']
    x_imgs_augmented = [x_img]
    if flip:
        aug_type.append('flip')
        x_imgs_augmented.append(np.flip(x_img, axis=-1))
    for shift_p in rotate:
        shift = int(round(shift_p * x_img.shape[-1]))
        aug_type.append('rotate %d' % shift)
        x_imgs_augmented.append(np.roll(x_img, shift, axis=-1))
    return torch.FloatTensor(np.concatenate(x_imgs_augmented, 0)), aug_type


def augment_undo(x_imgs_augmented, aug_type):
    x_imgs_augmented = x_imgs_augmented.cpu().numpy()
    sz = x_imgs_augmented.shape[0] // len(aug_type)
    x_imgs = []
    for i, aug in enumerate(aug_type):
        x_img = x_imgs_augmented[i*sz : (i+1)*sz]
        if aug == 'flip':
            x_imgs.append(np.flip(x_img, axis=-1))
        elif aug.startswith('rotate'):
            shift = int(aug.split()[-1])
            x_imgs.append(np.roll(x_img, -shift, axis=-1))
        elif aug == '':
            x_imgs.append(x_img)
        else:
            raise NotImplementedError()

    return np.array(x_imgs)


def inference(net, x, device, flip=False, rotate=[], visualize=False,
              force_cuboid=False, force_raw=False, min_v=None, r=0.05):
    '''
    net   : the trained HorizonNet
    x     : tensor in shape [1, 3, 512, 1024]
    flip  : fliping testing augmentation
    rotate: horizontal rotation testing augmentation
    '''

    H, W = tuple(x.shape[2:])

    # Network feedforward (with testing augmentation)
    x, aug_type = augment(x, flip, rotate)
    y_bon_, y_cor_ = net(x.to(device))
    y_bon_ = augment_undo(y_bon_.cpu(), aug_type).mean(0)
    y_cor_ = augment_undo(torch.sigmoid(y_cor_).cpu(), aug_type).mean(0)

    # Visualize raw model output
    if visualize:
        vis_out = visualize_a_data(x[0],
                                   torch.FloatTensor(y_bon_[0]),
                                   torch.FloatTensor(y_cor_[0]))
    else:
        vis_out = None

    # 保存原始的y_bon_用于返回


    y_bon_ = (y_bon_[0] / np.pi + 0.5) * H - 0.5
    y_bon_[0] = np.clip(y_bon_[0], 1, H/2-1)
    y_bon_[1] = np.clip(y_bon_[1], H/2+1, H-2)
    y_cor_ = y_cor_[0, 0]

    original_y_bon = y_bon_.copy()



    # Init floor/ceil plane
    z0 = 50
    _, z1 = post_proc.np_refine_by_fix_z(*y_bon_, z0)

    if force_raw:
        # Do not run post-processing, export raw polygon (1024*2 vertices) instead.
        # [TODO] Current post-processing lead to bad results on complex layout.
        cor = np.stack([np.arange(1024), y_bon_[0]], 1)

    else:
        # Detech wall-wall peaks
        if min_v is None:
            min_v = 0 if force_cuboid else 0.05
        r = int(round(W * r / 2))
        N = 4 if force_cuboid else None
        xs_ = find_N_peaks(y_cor_, r=r, min_v=min_v, N=N)[0]

        # Generate wall-walls
        cor, xy_cor = post_proc.gen_ww(xs_, y_bon_[0], z0, tol=abs(0.16 * z1 / 1.6), force_cuboid=force_cuboid)
        if not force_cuboid:
            # Check valid (for fear self-intersection)
            xy2d = np.zeros((len(xy_cor), 2), np.float32)
            for i in range(len(xy_cor)):
                xy2d[i, xy_cor[i]['type']] = xy_cor[i]['val']
                xy2d[i, xy_cor[i-1]['type']] = xy_cor[i-1]['val']
            if not Polygon(xy2d).is_valid:
                print(
                    'Fail to generate valid general layout!! '
                    'Generate cuboid as fallback.',
                    file=sys.stderr)
                xs_ = find_N_peaks(y_cor_, r=r, min_v=0, N=4)[0]
                cor, xy_cor = post_proc.gen_ww(xs_, y_bon_[0], z0, tol=abs(0.16 * z1 / 1.6), force_cuboid=True)

    # Expand with btn coory
    cor = np.hstack([cor, post_proc.infer_coory(cor[:, 1], z1 - z0, z0)[:, None]])

    # Collect corner position in equirectangular
    cor_id = np.zeros((len(cor)*2, 2), np.float32)
    for j in range(len(cor)):
        cor_id[j*2] = cor[j, 0], cor[j, 1]
        cor_id[j*2 + 1] = cor[j, 0], cor[j, 2]

    # Normalized to [0, 1]
    cor_id[:, 0] /= W
    cor_id[:, 1] /= H

    return cor_id, z0, z1, vis_out, original_y_bon


def main(pth, img_glob, output_dir, visualize=False, flip=False, rotate=[], 
         r=0.05, min_v=None, force_cuboid=False, force_raw=False, no_cuda=False):
    """
    参数说明:
    pth: str - 模型检查点路径
    img_glob: str - 输入图像路径（支持glob模式）
    output_dir: str - 输出目录路径
    visualize: bool - 是否可视化
    flip: bool - 是否进行左右翻转增强
    rotate: list[float] - 水平旋转增强的比例列表
    r: float - 后处理相关参数
    min_v: float - 后处理相关参数
    force_cuboid: bool - 是否强制立方体布局
    force_raw: bool - 是否输出原始多边形
    no_cuda: bool - 是否禁用CUDA
    """
    # 准备要处理的图像
    paths = sorted(glob.glob(img_glob))
    if len(paths) == 0:
        print('no images found')
    for path in paths:
        assert os.path.isfile(path), '%s not found' % path

    # 检查目标目录
    if not os.path.isdir(output_dir):
        print('Output directory %s not existed. Create one.' % output_dir)
        os.makedirs(output_dir)
    device = torch.device('cpu' if no_cuda else 'cuda')

    # 加载训练好的模型
    net = utils.load_trained_model(HorizonNet, pth).to(device)
    net.eval()

    # 推理
    with torch.no_grad():
        for i_path in tqdm(paths, desc='Inferencing'):
            k = os.path.split(i_path)[-1][:-4]

            # 加载图像
            img_pil = Image.open(i_path)
            if img_pil.size != (1024, 512):
                img_pil = img_pil.resize((1024, 512), Image.BICUBIC)
            img_ori = np.array(img_pil)[..., :3].transpose([2, 0, 1]).copy()
            x = torch.FloatTensor([img_ori / 255])

            # 推理角点
            cor_id, z0, z1, vis_out, y_bon = inference(net=net, x=x, device=device,
                                               flip=flip, rotate=rotate,
                                               visualize=visualize,
                                               force_cuboid=force_cuboid,
                                               force_raw=force_raw,
                                               min_v=min_v, r=r)

            # 输出结果
            with open(os.path.join(output_dir, k + '.json'), 'w') as f:
                json.dump({
                    'z0': float(z0),
                    'z1': float(z1),
                    'uv': [[float(u), float(v)] for u, v in cor_id],
                    'y_bon': y_bon.tolist(),
                }, f)

            if vis_out is not None:
                vis_path = os.path.join(output_dir, k + '.raw.png')
                vh, vw = vis_out.shape[:2]
                Image.fromarray(vis_out)\
                     .resize((vw//2, vh//2), Image.LANCZOS)\
                     .save(vis_path)

if __name__ == '__main__':
    # 示例用法
    main(
        pth='/home/jarvisai/HorizonNet/ckpt/resnet50_rnn__mp3d.pth',
        img_glob='/home/jarvisai/HorizonNet/dataset/project001/output/*.png',
        output_dir='/home/jarvisai/HorizonNet/dataset/project001/output/json',
        visualize=True
    )
