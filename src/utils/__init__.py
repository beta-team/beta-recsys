__title__ = "Terrier Recommenders"
__version__ = "2020.02"
__author__ = "Terrier Team at University of Glasgow"
__license__ = "MIT"
__copyright__ = "Copyright 2020-present University of Glasgow"
import os

# Version synonym
VERSION = __version__
# par_abs_dir = os.path.abspath(os.path.join(os.path.abspath("."), os.pardir))

UTILS_ROOT = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.abspath(os.path.join(UTILS_ROOT, "..", ".."))

configs = BASE_DIR + "/configs/"
datasets = BASE_DIR + "/datasets/"
checkpoints = BASE_DIR + "/checkpoints/"
results = BASE_DIR + "/results/"
logs = BASE_DIR + "/logs/"
samples = BASE_DIR + "/samples/"
runs = BASE_DIR + "/runs/"


for DIR in [configs, datasets, checkpoints, results, samples, logs, runs]:
#     print(DIR)
    if not os.path.exists(DIR):
        os.makedirs(DIR)