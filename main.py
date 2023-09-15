import cv2
import os
import numpy as np

import load_yml

from CalibrationImg import Calibration, CameraParameter

import cv2
import numpy as np


def create_folder_if_not_exists(path):
    if not os.path.exists(path):
        os.makedirs(path, True)


def save_camera_parameters_to_yaml(saveYMLPath, camera_parameter):
    """
    将相机参数保存到 YML 文件

    Args:
        saveYMLPath (str): 要保存的 YML 文件路径
        camera_parameter (dict): 包含相机参数的字典

    Returns:
        None
    """
    # 创建一个 FileStorage 对象以写入 YML 文件
    fs = cv2.FileStorage(saveYMLPath, cv2.FILE_STORAGE_WRITE)

    # 写入内部参数矩阵 (Camera Matrix)
    fs.write("cameraMatrix", camera_parameter.cameraMatrix)

    # 写入畸变系数 (Distortion Coefficients)
    fs.write("distCoeffs", camera_parameter.distCoeffs)
    # fs.write("text", "This is the first line")

    # 写入外参的旋转向量 (Rotation Vectors) 和平移向量 (Translation Vectors)
    for i, (rvec, tvec) in enumerate(zip(camera_parameter.external_rvecs_list, camera_parameter.external_tvecs_list)):
        fs.write(f"external_rot_index{i}", rvec)
        fs.write(f"external_tra_index{i}", tvec)
        # fs.write("text", "This is the first line")

    # 释放 FileStorage 对象
    fs.release()

    print(f"相机参数已保存到 {saveYMLPath}")


def process_images(config, img_path):
    image_files = os.listdir(img_path)
    calibration_imgs = []

    for image_filename in image_files:
        if image_filename.endswith('.jpg') or image_filename.endswith('.png'):
            calibration_img = Calibration(config, img_path, image_filename)
            if calibration_img.gray_image is not None:
                calibration_imgs.append(calibration_img)

    return calibration_imgs


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    config_file_path = 'config/cfg.yml'
    config = load_yml.read_camera_calibration_config(config_file_path)

    saveYMLPath = config['SaveYMLPath']
    inter_img_path = config['InternalImgPath']
    exterimg_path = config['ExternalImgPath']

    inter_calibration_imgs = process_images(config, inter_img_path)
    exter_calibration_imgs = process_images(config, exterimg_path)

    # 同一个相机拍出来的尺寸应该是一样的
    img_size = [inter_calibration_imgs[0].height, inter_calibration_imgs[0].width]

    object_list = []
    corners_list = []
    for cal_img in inter_calibration_imgs:
        corners_list.append(cal_img.corners)
        object_list.append(cal_img.object_points)

    camera_parameter = CameraParameter(config)
    for cal_img in exter_calibration_imgs:
        camera_parameter.calculating_internal_parameters(object_list, corners_list, img_size)
        camera_parameter.calculating_external_parameters(cal_img.object_points, cal_img.corners)

    create_folder_if_not_exists(os.path.dirname(saveYMLPath))
    save_camera_parameters_to_yaml(saveYMLPath, camera_parameter)
