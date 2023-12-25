

DEBUG_MODE = False

# grpc server settings
PORT = 50051
MAX_WORKERS = 24
SHOW_VIDEO = False

# dqn tensorflow
VIDEO_FRAME_SIZE: tuple[int,int] = (64,64)
DISTANCE_SENSORS_COUNT = 6
MODEL_OUTPUT_COMMANDS_COUNT: int = 5

# logger settings
WRITE_LOGS_TO_CONSOLE = True
WRITE_LOGS_TO_FILE = False
LOG_FILE_NAME = "server.log"