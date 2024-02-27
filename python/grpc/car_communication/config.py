import os


# basic settings
SHOW_STREAM_VIDEO = False
SHOW_CLIENT_DATA = False
TARGET_ENCODE_TO = '.jpg'

# paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_PATH = os.path.join(BASE_DIR, 'src')
VIDEO_PATH = os.path.join(SRC_PATH,'1.mp4')

# grpc server settings
PORT = 50051
MAX_WORKERS = 24
SERVER_URL = f'localhost:{PORT}'
