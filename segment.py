import json
import cv2
import numpy as np
import os
from scipy.signal import savgol_filter
from scipy.interpolate import interp1d

from PIL import Image
import py360convert


def save_segment(image, x_start, x_end, segment_index, output_subfolder):
    """
    将除[x_start, x_end]之外的部分全部涂黑，并保存处理后的图片。

    :param image: 输入的原始图像。
    :param x_start: 目标区域的起始x坐标（包含）。
    :param x_end: 目标区域的结束x坐标（不包含）。
    :param segment_index: 当前片段的索引，用于命名。
    :param output_subfolder: 输出子文件夹路径。
    """
    if x_start < x_end and 0 <= x_start < image.shape[1] and 0 < x_end <= image.shape[1]:
        # 复制原始图像以避免修改原图
        processed_image = image.copy()

        # 将目标区域外的部分涂黑
        if x_start > 0:
            processed_image[:, :x_start] = 0  # 左边部分
        if x_end < image.shape[1]:
            processed_image[:, x_end:] = 0  # 右边部分

        segment_name = f'segment_{segment_index}.jpg'
        cv2.imwrite(os.path.join(output_subfolder, segment_name), processed_image)
    else:
        print(f"Warning: Invalid segment boundaries for segment {segment_index}, skipping.")


def annotate_image(image, points, point_color=(0, 0, 255), point_radius=3, font=cv2.FONT_HERSHEY_SIMPLEX,
                   font_scale=0.5, font_thickness=1, text_color=(255, 0, 0)):
    """
    Annotate the image with points and their indices.

    :param image: Input image to annotate.
    :param points: List of (u, v) coordinates to mark on the image.
    :param point_color: Color of the points in BGR format.
    :param point_radius: Radius of the circles drawn around each point.
    :param font: Font type for text annotations.
    :param font_scale: Font scale for text annotations.
    :param font_thickness: Thickness of the text annotations.
    :param text_color: Color of the text annotations in BGR format.
    :return: Annotated image.
    """
    annotated_image = image.copy()
    for idx, (u, v) in enumerate(points):
        # Draw a circle at the point
        cv2.circle(annotated_image, (u, v), point_radius, point_color, -1)

        # Add a text label near the point
        text_size = cv2.getTextSize(str(idx), font, font_scale, font_thickness)[0]
        text_x = u + 5  # Offset from the point to avoid overlap
        text_y = v + 5
        cv2.putText(annotated_image, str(idx), (text_x, text_y), font, font_scale, text_color, font_thickness)

    return annotated_image



def interpolate_y_bon(y_bon, height, width):
    """
    对 y_bon 进行插值处理
    :param y_bon: y_bon 数据
    :param height: 图像高度
    :param width: 图像宽度
    :return: 插值后的 y_bon_upper 和 y_bon_lower
    """
    y_scale = height / 512
    y_bon_upper_scale = np.array(y_bon[0]) * y_scale  # 第一条线
    y_bon_lower_scale = np.array(y_bon[1]) * y_scale  # 第二条线
    y_smoothed_up = savgol_filter(y_bon_upper_scale, window_length=11, polyorder=3)
    y_smoothed_low = savgol_filter(y_bon_lower_scale, window_length=11, polyorder=3)

    # 使用线性插值的方法投射到原图
    x_bon_old = np.linspace(0, width, num=1024, endpoint=False)
    x_bon_new = np.arange(width)

    # 使用三次样条插值
    y_bon_upper_cubic_interp = interp1d(x_bon_old, y_smoothed_up, kind='cubic', fill_value="extrapolate")
    y_bon_upper_interp = y_bon_upper_cubic_interp(x_bon_new)

    y_bon_lower_cubic_interp = interp1d(x_bon_old, y_smoothed_low, kind='cubic', fill_value="extrapolate")
    y_bon_lower_interp = y_bon_lower_cubic_interp(x_bon_new)

    return y_bon_upper_interp, y_bon_lower_interp

def plot_and_segment(json_folder, image_folder, output_folder):
    # 遍历 JSON 文件夹中的所有 JSON 文件
    for json_file in os.listdir(json_folder):
        if json_file.endswith('.json'):
            json_path = os.path.join(json_folder, json_file)
            with open(json_path, 'r') as f:
                data = json.load(f)

            # 获取图片文件名（去掉.json后缀）
            image_name = os.path.splitext(json_file)[0] + '.jpg'
            image_path = os.path.join(image_folder, image_name)
            if not os.path.exists(image_path):
                print(f"无法找到图片: {image_path}")
                continue

            """
            part 1 分割天地墙
            """
            # 读取图片
            img = cv2.imread(image_path)
            if img is None:
                raise Exception(f"无法读取图片: {image_path}")

            # 获取图片的高度和宽度
            height, width = img.shape[:2]

            # 获取y_bon的两条线
            y_bon_upper, y_bon_lower = interpolate_y_bon(data['y_bon'], height, width)

            # 创建三个掩码图像
            mask_upper = np.zeros_like(img)
            mask_middle = np.zeros_like(img)
            mask_lower = np.zeros_like(img)

            # 根据y_bon的两条线分割区域
            for x in range(width):
                if x < len(y_bon_upper) and x < len(y_bon_lower):
                    y_upper = int(y_bon_upper[x])
                    y_lower = int(y_bon_lower[x])

                    # 上区域（y < y_upper）
                    if 0 <= y_upper < height:
                        mask_upper[:y_upper, x] = 255

                    # 中区域（y_upper <= y < y_lower）
                    if 0 <= y_upper < height and 0 <= y_lower < height:
                        mask_middle[y_upper:y_lower, x] = 255

                    # 下区域（y >= y_lower）
                    if 0 <= y_lower < height:
                        mask_lower[y_lower:, x] = 255

            # 保存分割结果
            upper_segment = cv2.bitwise_and(img, mask_upper)
            middle_segment = cv2.bitwise_and(img, mask_middle)
            lower_segment = cv2.bitwise_and(img, mask_lower)

            # 创建输出文件夹
            output_subfolder = os.path.join(output_folder, os.path.splitext(json_file)[0])
            os.makedirs(output_subfolder, exist_ok=True)

            cv2.imwrite(os.path.join(output_subfolder, 'upper_segment.jpg'), upper_segment)
            cv2.imwrite(os.path.join(output_subfolder, 'middle_segment.jpg'), middle_segment)
            cv2.imwrite(os.path.join(output_subfolder, 'lower_segment.jpg'), lower_segment)

            """
            part 2 拼接图片 分割所有的墙面
            """
            uv_points = data['uv']

            pixel_points = []
            for uv in uv_points:
                x = int(uv[0] * width)  # 将 u坐标转换为像素 x
                y = int(uv[1] * height)  # 将 v坐标转换为像素 y
                pixel_points.append((x, y))

            # 找到第一个点的 x坐标
            first_point_x = pixel_points[0][0]

            pixel_points = [(x - first_point_x, y) for x, y in pixel_points]

            # 分割图像
            left_part = middle_segment[:, :first_point_x]  # 左侧部分（小于第一个点）
            right_part = middle_segment[:, first_point_x:]  # 右侧部分（大于等于第一个点）

            # 拼接图像：将左侧部分拼接到右侧
            concatenated_image = np.hstack((right_part, left_part))

            # 保存拼接结果
            cv2.imwrite(os.path.join(output_subfolder, 'concatenated_image.jpg'), concatenated_image)

            """
            part 3 分割墙面layout图
            """
            # # 根据uv点分割concatenated_image
            # if len(pixel_points) > 0:
            #     first_x = pixel_points[0][0]
            #
            #     # 第一条分割线左侧
            #     save_segment(concatenated_image, 0, first_x, 0, output_subfolder)
            #
            #     # 第一条分割线右侧
            #     if len(pixel_points) > 1:
            #         second_x = pixel_points[2][0]
            #         save_segment(concatenated_image, first_x, second_x, 1, output_subfolder)

            # 处理中间的分割线
            for i in range(0, len(pixel_points), 2):
                if i + 2 < len(pixel_points):
                    x1 = pixel_points[i][0]
                    x2 = pixel_points[i + 2][0]
                    print(pixel_points[i][0])
                    # 中间的分割线之间的情况
                    save_segment(concatenated_image, x1, x2, (i // 2) + 1, output_subfolder)
                else:
                    x1 = pixel_points[i][0]
                    x2 = width
                    save_segment(concatenated_image, x1, x2, (i // 2) + 1, output_subfolder)


            # 处理最后一条分割线（左侧和右侧）
            # if len(pixel_points) % 2 != 0:
            #     last_x = pixel_points[-1][0]
            #
            #     # 最后一条分割线左侧
            #     save_segment(concatenated_image, last_x, width, len(pixel_points) // 2 + 1, output_subfolder)
            # else:
            #     # 如果有偶数个分割点，则处理倒数第二个分割线到图像边缘的部分
            #     second_last_x = pixel_points[-2][0]
            #     last_x = pixel_points[-1][0]
            #     save_segment(concatenated_image, second_last_x, last_x, len(pixel_points) // 2, output_subfolder)
            #
            #     # 最后一条分割线右侧
            #     save_segment(concatenated_image, last_x, width, len(pixel_points) // 2 + 1, output_subfolder)

            """
            part 4 标注分割点
            """
            # annotated_image = annotate_image(concatenated_image, [point[:2] for point in pixel_points])
            # # 保存拼接结果
            # cv2.imwrite(os.path.join(output_subfolder, 'annotated_image.jpg'), annotated_image)

            # """
            # part 5 转换cube图
            # """
            # cube_img = py360convert.e2c(e_img=middle_segment, face_w=1024, mode='bilinear', cube_format='dict')
            #
            # # 遍历每个立方体面并保存
            # for face_key, face_array in cube_img.items():
            #     # 将 numpy 数组转换为 BGR 格式的图像（OpenCV 默认使用 BGR）
            #     output_path = f'face_{face_key}.jpg'
            #
            #     cv2.imwrite(os.path.join(output_subfolder, output_path), face_array)

            """
            part 6 自定义摄像头图片
            """
            e_img = middle_segment  # 随机生成一个等矩形投影图像
            fov_deg = (150, 150)  # 水平和垂直视场角各90度
            u_deg = 90  # 水平视角为0度
            v_deg = 0  # 垂直视角为0度
            out_hw = (2048, 1024)  # 输出图像尺寸
            in_rot_deg = 0  # 输入图像不旋转
            mode = 'nearest'  # 使用双线性插值

            pers_img = py360convert.e2p(e_img, fov_deg, u_deg, v_deg, out_hw, in_rot_deg, mode)
            output_path = f'fov90.jpg'
            cv2.imwrite(os.path.join(output_subfolder, output_path), pers_img)


# 使用示例
json_folder = '/home/jarvisai/HorizonNet/dataset/project001/output/json'
image_folder = '/home/jarvisai/HorizonNet/dataset/project001'
output_folder = '/home/jarvisai/HorizonNet/dataset/project001/processed_images'
plot_and_segment(json_folder, image_folder, output_folder)