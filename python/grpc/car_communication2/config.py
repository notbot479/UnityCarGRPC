from typing import Literal
import os


# basic settings
SHOW_VIDEO = False
SHOW_DATA = True
CAMERA_MODE: Literal['video'] | Literal['image'] = 'video'
TARGET_ENCODE_TO = '.jpg'

# paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PICTURE_PATH = os.path.join(BASE_DIR,'1.jpg')
VIDEO_PATH = os.path.join(BASE_DIR,'1.mp4')

# grpc server settings
PORT = 50051
MAX_WORKERS = 24
SERVER_URL = f'localhost:{PORT}'
