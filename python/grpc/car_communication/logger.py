# -*- coding: utf-8 -*-
import logging

from config import (
        WRITE_LOGS_TO_CONSOLE, 
        WRITE_LOGS_TO_FILE,
        LOG_FILE_NAME,
        DEBUG_MODE, 
        )


LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
BASE_LEVEL = logging.DEBUG if DEBUG_MODE else logging.INFO

formatter = logging.Formatter(LOG_FORMAT)
logger = logging.getLogger(__name__)
logger.setLevel(BASE_LEVEL)

# write logs to console [DEBUG or INFO]
if WRITE_LOGS_TO_CONSOLE:
    ch = logging.StreamHandler()
    ch.setLevel(BASE_LEVEL)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

# write logs to file [INFO only]
if WRITE_LOGS_TO_FILE and LOG_FILE_NAME:
    fh = logging.FileHandler(LOG_FILE_NAME)
    fh.setLevel(logging.INFO)
    fh.setFormatter(formatter)
    logger.addHandler(fh)