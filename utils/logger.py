import logging
import os


def setup_logger(log_dir="outputs/logs"):

    os.makedirs(log_dir, exist_ok=True)

    logger = logging.getLogger("GalaxyEye")

    logger.setLevel(logging.INFO)

    file_handler = logging.FileHandler(
        os.path.join(log_dir, "train.log")
    )

    formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s'
    )

    file_handler.setFormatter(formatter)

    logger.addHandler(file_handler)

    return logger