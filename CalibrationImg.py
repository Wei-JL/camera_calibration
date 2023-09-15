import os
import cv2
import numpy as np

import load_yml


class Calibration:
    def __init__(self, config, image_filename):
        self.img_path = config['InternalImgPath']
        self.board_size = config['BoardSize']
        self.square_size = config['SquareSize']
        self.save_img_path = config['SaveIMGPath']

        self.original_image_name = image_filename
        self.image_path = os.path.join(self.img_path, image_filename)
        self.original_image_cv2 = cv2.imread(self.image_path)

        self.gray_image = self.load_and_convert_to_gray()
        self.width, self.height = self.gray_image.shape[:2]
        # 生成求内参用的棋盘格角点坐标
        self.object_points = self.init_checkerboard_points()
        # 生成内参矩阵
        self.camera_matrix, self.dist_coeffs = self.generate_depth_intrin()

        self.corners = []  # 新增成员变量用于记录角点
        self.is_draw = False
        # 定义一组不同的颜色，每个角点和连接线使用不同的颜色
        self.colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (0, 255, 255), (255, 0, 255)]

        # 在找到角点，并且在原始图像上绘制角点
        self.find_and_draw_corners()

    def draw_corners_and_lines(self, image):
        is_save_img, save_path = self.save_img_path
        for i in range(len(self.corners)):
            # 绘制角点
            cv2.circle(image, (int(self.corners[i][0][0]), int(self.corners[i][0][1])), 3,
                       self.colors[i % len(self.colors)], -1)
        for i in range(1, len(self.corners)):
            pt1 = (int(self.corners[i - 1][0][0]), int(self.corners[i - 1][0][1]))
            pt2 = (int(self.corners[i][0][0]), int(self.corners[i][0][1]))
            cv2.line(image, pt1, pt2, self.colors[i % len(self.colors)], 1)

        # 是否保存绘制好的图像
        if is_save_img and save_path is not None:
            if not os.path.exists(save_path):
                os.makedirs(save_path, exist_ok=True)
            cv2.imwrite(os.path.join(save_path, self.original_image_name), image)

        # 显示图像
        cv2.imshow("Corners", image)
        cv2.waitKey(800)
        cv2.destroyAllWindows()

    def find_and_draw_corners(self, ):
        flags = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE
        ret, corners = cv2.findChessboardCorners(self.gray_image, tuple(self.board_size), flags=flags)
        if ret:
            # 使用亚像素角点检测参数
            # criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.0001)
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_COUNT, 30, 0.0001)

            # 找到角点
            # corners = cv2.goodFeaturesToTrack(self.gray_image, self.board_size[0] * self.board_size[1], 0.01, 10)

            # 使用亚像素角点检测
            corners = cv2.cornerSubPix(self.gray_image, corners, (5, 5), (-1, -1), criteria)
            self.corners = corners

            if self.is_draw:
                self.draw_corners_and_lines(self.original_image_cv2)

        else:
            print("Chessboard corners not found in the image.")

    def load_and_convert_to_gray(self):
        try:
            gray_image = cv2.cvtColor(self.original_image_cv2, cv2.COLOR_BGR2GRAY)
            return gray_image
        except Exception as e:
            print(f"Error loading and converting image: {str(e)}")
            return None

    def generate_depth_intrin(self):

        # 准备棋盘格的坐标
        board_col, board_row = self.board_size
        objp = np.zeros((board_col * board_row, 3), np.float32)
        objp[:, :2] = np.mgrid[0:board_col, 0:board_row].T.reshape(-1, 2)
        return objp, 1

    def init_checkerboard_points(self, ):
        # 准备棋盘格的坐标
        board_col, board_row = self.board_size
        object_points = np.zeros((board_col * board_row, 3), np.float32)
        object_points[:, :2] = np.mgrid[0:board_col, 0:board_row].T.reshape(-1, 2)

        return object_points


class CameraParameter:
    def __init__(self, config, ):
        self.board_size = config['BoardSize']
        self.square_size = config['SquareSize']
        self.external_img_path = config['ExternalImgPath']
        self.distance_origin = config['DistanceFromOrigin']
        self.demarcate_edge = config['DemarcateEdgeDistance']

        # 初始化相机的内参与畸变
        self.cameraMatrix = []
        self.distCoeffs = []
        self.external_rvecs = []
        self.external_tvecs = []
        self.img_size = []

        # 把所有角点转为真实坐标系下的3d坐标
        self.truth_object_points = self.init_corners_3d()

    def init_corners_3d(self, ):
        object_points = []

        for j in range(self.board_size[1]):  # 高度
            for i in range(self.board_size[0]):  # 宽度
                x = round(i * self.square_size + self.distance_origin[0] + self.demarcate_edge[0], 6)
                y = self.distance_origin[1]
                z = round(j * self.square_size + self.distance_origin[2] - self.demarcate_edge[1], 6)
                object_points.append([x, y, z])

        return np.array(object_points, dtype=np.float32)

    def calculating_internal_parameters(self, object_list, corners_list, img_size):
        ret, cameraMatrix, distCoeffs, rvecs, tvecs = cv2.calibrateCamera(object_list, corners_list, img_size,
                                                                          None, None)
        self.cameraMatrix = cameraMatrix
        self.distCoeffs = distCoeffs
        self.img_size = img_size

    def calculating_external_parameters(self, object_points, corners):
        if not len(self.cameraMatrix) and len(self.distCoeffs):
            print("Please calculate internal parameters first.")
            return

        _, rvecs, tvecs = cv2.solvePnP(object_points, corners, self.cameraMatrix,
                                       self.distCoeffs)

        self.external_rvecs = rvecs
        self.external_tvecs = tvecs

