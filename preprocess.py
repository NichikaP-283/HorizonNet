'''
This script preprocess the given 360 panorama image under euqirectangular projection
and dump them to the given directory for further layout prediction and visualization.
The script will:
    - extract and dump the vanishing points
    - rotate the equirect image to align with the detected VP
    - extract the VP aligned line segments (for further layout prediction model)
The dump files:
    - `*_VP.txt` is the vanishg points
    - `*_aligned_rgb.png` is the VP aligned RGB image
    - `*_aligned_line.png` is the VP aligned line segments images

Author: Cheng Sun
Email : chengsun@gapp.nthu.edu.tw
'''

import os
import glob
import argparse
import numpy as np
from PIL import Image
from tqdm import tqdm

from misc.pano_lsd_align import panoEdgeDetection, rotatePanorama


def preprocess_panorama(img_path, output_dir, q_error=0.7, refine_iter=3, rgbonly=False):
    """
    预处理360度全景图像。
    
    参数:
        img_path (str): 输入图像路径
        output_dir (str): 输出目录路径
        q_error (float): 检测参数，默认0.7
        refine_iter (int): 优化迭代次数，默认3
        rgbonly (bool): 是否仅输出RGB图像，默认False
    """
    # 确保输出目录存在
    if not os.path.isdir(output_dir):
        print('Output directory %s not existed. Create one.')
        os.makedirs(output_dir)

    print('Processing', img_path, flush=True)

    # 加载并调整图像大小
    img_ori = np.array(Image.open(img_path).resize((1024, 512), Image.BICUBIC))[..., :3]

    # VP检测和线段提取
    _, vp, _, _, panoEdge, _, _ = panoEdgeDetection(img_ori,
                                                    qError=q_error,
                                                    refineIter=refine_iter)
    panoEdge = (panoEdge > 0)

    # 根据VP对齐图像
    i_img = rotatePanorama(img_ori / 255.0, vp[2::-1])
    l_img = rotatePanorama(panoEdge.astype(np.float32), vp[2::-1])

    # 保存结果
    basename = os.path.splitext(os.path.basename(img_path))[0]
    if rgbonly:
        path = os.path.join(output_dir, '%s.png' % basename)
        Image.fromarray((i_img * 255).astype(np.uint8)).save(path)
    else:
        path_VP = os.path.join(output_dir, '%s_VP.txt' % basename)
        path_i_img = os.path.join(output_dir, '%s_aligned_rgb.png' % basename)
        path_l_img = os.path.join(output_dir, '%s_aligned_line.png' % basename)

        with open(path_VP, 'w') as f:
            for i in range(3):
                f.write('%.6f %.6f %.6f\n' % (vp[i, 0], vp[i, 1], vp[i, 2]))
        Image.fromarray((i_img * 255).astype(np.uint8)).save(path_i_img)
        Image.fromarray((l_img * 255).astype(np.uint8)).save(path_l_img)

def main():
    # 直接指定参数
    input_dir = "/home/jarvisai/HorizonNet/dataset/project001"        # 输入文件夹路径
    output_dir = "/home/jarvisai/HorizonNet/dataset/project001/output"             # 输出目录
    q_error = 0.7                             # 检测参数
    refine_iter = 3                           # 优化迭代次数
    rgbonly = True                           # 是否仅输出RGB图像

    # 确保输入目录存在
    assert os.path.isdir(input_dir), f'{input_dir} 目录不存在'
    
    # 获取所有全景图文件
    panorama_files = []
    for ext in ['.jpg', '.jpeg', '.png']:
        panorama_files.extend(glob.glob(os.path.join(input_dir, f'*{ext}')))
    
    # 批量处理全景图
    for img_path in tqdm(panorama_files):
        try:
            preprocess_panorama(img_path, output_dir, q_error, refine_iter, rgbonly)
        except Exception as e:
            print(f'处理 {img_path} 时出错: {str(e)}')
            continue

if __name__ == '__main__':
    main()
