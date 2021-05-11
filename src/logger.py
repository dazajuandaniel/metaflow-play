"""
Internal Module that handles Logging
"""

import logging
import sys
import os
from logging.handlers import TimedRotatingFileHandler


FORMATTER = logging.Formatter("%(asctime)s — %(name)s — %(levelname)s — %(message)s")
LOG_FILE = '../logs/python/MetaFlowTensorflow.log'

os.makedirs(os.path.dirname(LOG_FILE),exist_ok = True)

def get_console_handler():
   """
   Function to manage the console handler
   """
   console_handler = logging.StreamHandler(sys.stdout)
   console_handler.setFormatter(FORMATTER)
   return console_handler

def get_file_handler():
   """
   Function to get the handler
   """
   file_handler = logging.FileHandler(LOG_FILE, mode="a")
   file_handler.setFormatter(FORMATTER)
   return file_handler

def get_logger(logger_name):
   """
   Function to get the logger
   """
   logger = logging.getLogger(logger_name)
   logger.setLevel(logging.DEBUG)
   logger.addHandler(get_console_handler())
   logger.addHandler(get_file_handler())
   logger.propagate = False
   return logger

