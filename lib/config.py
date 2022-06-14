import logging
import os
import time
from importlib import reload

import torch

# Global variables
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
log_dir = None
log_name = None
writer = None


def setup_logging(name, dir=""):

    # Setup global log name and directory
    global log_name
    log_name = name

    # Setup global logging directory
    global log_dir
    log_dir = os.path.join("log", dir)

    # Create the logging folder if it does not exist already
    if not os.path.isdir(log_dir):
        os.makedirs(log_dir)

    # Need to reload logging as otherwise the logger might be captured by another library
    reload(logging)

    # Setup global logger
    logging.basicConfig(
        level=logging.INFO,
        format="[%(levelname)-5.5s %(asctime)s] %(message)s",
        datefmt='%H:%M:%S',
        handlers=[
            logging.FileHandler(os.path.join(
                log_dir, time.strftime("%Y%m%d_%H%M") + "_" + name + ".log")
            ),
            logging.StreamHandler()
        ])
