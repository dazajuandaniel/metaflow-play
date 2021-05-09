"""
Script with Utility Functions
"""
import os
import time


def script_path(filename):
    """
    A convenience function to get the absolute path to a file in this
    tutorial's directory. This allows the tutorial to be launched from any
    directory.

    """
    filepath = os.path.join(os.path.dirname(__file__))
    return os.path.join(filepath)


def get_run_logdir(path:str = '../logs/tensorflow', folder_name: str = 'tensorflow_logs'):
    """
    Function that creates a new directory with the current time

    Args:
        path(str): Path for the location of the tensorflow logs
        folder_name(str): Name to give the folder for logs

    Returns:
        (str): The path of the created folder
    """
    root_logdir = os.path.join(path, folder_name)
    run_id = time.strftime("run_%Y_%m_%d-%H_%M_%S")
    return os.path.join(root_logdir, run_id)
