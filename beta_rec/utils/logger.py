import logging
import sys
from datetime import datetime

BASIC_FORMAT = "%(asctime)s:%(levelname)s - %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def init_logger(log_file_name="log", console=True, error=True, debug=False):
    """Initialize logger."""
    logger = logging.getLogger()
    logger.setLevel("DEBUG")
    formatter = logging.Formatter(BASIC_FORMAT, DATE_FORMAT)

    file_out_info = logging.FileHandler(log_file_name + ".info")
    file_out_info.setFormatter(formatter)
    file_out_info.setLevel("INFO")
    logger.addHandler(file_out_info)

    if console:
        con_out = logging.StreamHandler()
        con_out.setFormatter(formatter)
        con_out.setLevel("INFO")
        logger.addHandler(con_out)

    if error:
        file_out_error = logging.FileHandler(log_file_name + ".erro")
        file_out_error.setFormatter(formatter)
        file_out_error.setLevel("ERROR")
        logger.addHandler(file_out_error)

    if debug:
        file_out_debug = logging.FileHandler(log_file_name + ".debug")
        file_out_debug.setFormatter(formatter)
        file_out_debug.setLevel("DEBUG")
        logger.addHandler(file_out_debug)
    print("Init logger sucussful.")
    return logger


def get_logger(filename="default", level="info"):
    """Get logger."""
    logger = logging.getLogger()

    formatter = logging.Formatter(BASIC_FORMAT, DATE_FORMAT)
    if level == "info":
        file_out_info = logging.FileHandler(filename + ".log")
        file_out_info.setFormatter(formatter)
        file_out_info.setLevel("INFO")
        logger.addHandler(file_out_info)
    elif level == "error":
        file_out_error = logging.FileHandler(filename + ".log")
        file_out_error.setFormatter(formatter)
        file_out_error.setLevel("ERROR")
        logger.addHandler(file_out_error)
    print("Init ", level, "logger sucussful.")
    return logger


class Logger(object):
    """Logger Class."""

    def __init__(self, filename="default", stdout=None, stderr=None):
        """Initialize Logger Class."""
        self.stdout = stdout
        self.stderr = stderr
        self.fileno = sys.stdout.fileno
        self.filename = filename
        self.message = ""

    def write(self, message):
        """Log out message."""
        if message == "" or message is None:
            return
        elif "\n" in message:
            message = message[:-1]
            self.message += message
            now = datetime.now()
            date_time = now.strftime("%Y-%m-%d %H:%M:%S ")
            if self.stdout is not None:
                if "\n" in self.message:
                    self.message = date_time + "[INFO]-\n" + self.message + "\n"
                else:
                    self.message = date_time + "[INFO]-" + self.message + "\n"
                self.stdout.write(self.message)
                self.stdout.flush()
            if self.stderr is not None:
                if "\n" in self.message:
                    self.message = date_time + "[ERROR]-\n" + self.message + "\n"
                else:
                    self.message = date_time + "[ERROR]-" + self.message + "\n"
                self.stderr.write(self.message)
                self.stderr.flush()
            with open(self.filename, "a") as logger:
                logger.write(self.message)
            self.message = ""
        else:
            self.message += message

    def flush(self):
        """Missing Doc."""
        pass


# capture stderr and stdout
def init_std_logger(log_file="default"):
    """Initialize std logger."""
    print("logs will save in file:", log_file, ".stdout.log", ".stderr.log")
    sys.stdout = Logger(log_file + ".stdout.log", stdout=sys.stdout)
    sys.stderr = Logger(log_file + ".stderr.log", stderr=sys.stderr)
