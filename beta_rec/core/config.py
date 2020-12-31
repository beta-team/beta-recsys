import os
import sys


def find_config(config_file):
    """Read the config file, if it exists. Using defaults otherwise."""
    if os.path.exists(config_file):
        return config_file
    for config_file in config_file_paths(config_file):
        print("Search default config file in {}".format(config_file))
        if os.path.exists(config_file):
            print("Found default config file in {}".format(config_file))
            return config_file


def config_file_paths(config_file):
    """Get a list of config file paths."""
    config_filename = config_file.replace("../configs/", "beta_rec/")
    paths = []
    paths.append(os.path.join(sys.exec_prefix, config_filename))
    paths.append(os.path.join(sys.exec_prefix + "/local", config_filename))
    return paths
