import os


# basic settings
SHOW_STREAM_VIDEO = True
SHOW_CLIENT_DATA = True
TARGET_ENCODE_TO = '.jpg'

# paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
VIDEO_PATH = os.path.join(BASE_DIR,'1.mp4')

# grpc server settings
PORT = 50051
MAX_WORKERS = 24
SERVER_URL = f'localhost:{PORT}'
