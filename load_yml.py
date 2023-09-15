import yaml


def read_camera_calibration_config(config_file_path):
    try:
        with open(config_file_path, 'r+') as yml_file:
            config_data = yaml.safe_load(yml_file)

        camera_id = config_data.get('CameraID')
        camera_name = config_data.get('CamerName')
        image_size = config_data.get('ImageSize')
        pattern = config_data.get('Pattern')
        board_size = config_data.get('BoardSize')
        frame_num = config_data.get('FrameNum')
        square_size = config_data.get('SquareSize')
        internal_img_path = config_data.get('InternalImgPath')
        external_img_path = config_data.get('ExternalImgPath')
        save_yml_path = config_data.get('SaveYMLPath')
        save_img_path = config_data.get('SaveIMGPath')
        distance_origin = config_data.get('DistanceFromOrigin')
        demarcate_edge = config_data.get('DemarcateEdgeDistance')

        return {
            'CameraID': camera_id,
            'CameraName': camera_name,
            'ImageSize': image_size,
            'Pattern': pattern,
            'BoardSize': board_size,
            'FrameNum': frame_num,
            'SquareSize': square_size,
            'InternalImgPath': internal_img_path,
            'ExternalImgPath': external_img_path,
            'SaveYMLPath': save_yml_path,
            'SaveIMGPath': save_img_path,
            'DistanceFromOrigin': distance_origin,
            'DemarcateEdgeDistance': demarcate_edge
        }
    except Exception as e:
        print(f"Error reading configuration file: {str(e)}")
        return None
