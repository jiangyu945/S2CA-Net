import os
import logging

# log configure
def log_args(log_file):
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter(
        '%(asctime)s ===> %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S')

    # args FileHandler to save log file
    if not os.path.exists(log_file):
        os.mknod(log_file)
    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(formatter)

    # args StreamHandler to print log to console
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    ch.setFormatter(formatter)

    # add the two Handler
    logger.addHandler(ch)
    logger.addHandler(fh)

class Log:
    def __init__(self, log_root, log_name):
        self.root = log_root
        self.name = log_name

    def init(self):
        # create log output path
        # log_root = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'log')
        if not os.path.exists(self.root):
            os.makedirs(self.root)

        log_file = os.path.join(self.root, self.name)
        log_args(log_file)

    def write(self, info):
        logging.info(info)

