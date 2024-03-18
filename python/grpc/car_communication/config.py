import os


# basic settings
SHOW_STREAM_VIDEO = False
SHOW_CLIENT_DATA = False
TARGET_ENCODE_TO = '.jpg'

# paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_PATH = os.path.join(BASE_DIR, 'src')
VIDEO_PATH = os.path.join(SRC_PATH,'1.mp4')

# dqn paths
AGENT_LOAD_MODEL_NAME = 'model.keras'
if AGENT_LOAD_MODEL_NAME:
    path = os.path.join(BASE_DIR, AGENT_LOAD_MODEL_NAME) 
    exists = os.path.isfile(path)
    AGENT_LOAD_MODEL_PATH = path if exists else ''
else:
    AGENT_LOAD_MODEL_PATH = ''
AGENT_MODELS_PATH = os.path.join(BASE_DIR, 'models')
AGENT_LOGS_PATH = os.path.join(BASE_DIR, 'logs')

# grpc server settings
PORT = 50051
MAX_WORKERS = 60
SERVER_URL = f'localhost:{PORT}'
