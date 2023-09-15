import cv2
import os
import numpy as np

import load_yml

from CalibrationImg import Calibration, CameraParameter


def process_images(img_path, config):
    image_files = os.listdir(img_path)
    calibration_imgs = []

    for image_filename in image_files:
        if image_filename.endswith('.jpg') or image_filename.endswith('.png'):
            calibration_img = Calibration(config, image_filename)
            if calibration_img.gray_image is not None:
                calibration_imgs.append(calibration_img)

    return calibration_imgs


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    config_file_path = 'config/cfg.yml'
    config = load_yml.read_camera_calibration_config(config_file_path)

    inter_img_path = config['InternalImgPath']
    exterimg_path = config['ExternalImgPath']

    inter_calibration_imgs = process_images(inter_img_path, config)
    exter_calibration_imgs = process_images(exterimg_path, config)

    # 同一个相机拍出来的尺寸应该是一样的
    img_size = [inter_calibration_imgs[0].height, inter_calibration_imgs[0].width]

    object_list = []
    corners_list = []
    for cal_img in inter_calibration_imgs:
        corners_list.append(cal_img.corners)
        object_list.append(cal_img.object_points)

    for cal_img in exter_calibration_imgs:
        camera_parameter = CameraParameter(config)
        camera_parameter.calculating_internal_parameters(object_list, corners_list, img_size)
        camera_parameter.calculating_external_parameters(cal_img.object_points, cal_img.corners)

        print("内部参数矩阵 (Camera Matrix):")
        print(camera_parameter.cameraMatrix)
        print("\n畸变系数 (Distortion Coefficients):")
        print(camera_parameter.distCoeffs)
        print("\n 外参的旋转向量 (Rotation Vectors):")
        print(camera_parameter.external_rvecs)
        print("\n 外参的平移向量 (Translation Vectors):")
        print(camera_parameter.external_tvecs)
