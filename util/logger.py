#!/usr/bin/python
# -*- coding: utf-8 -*-
import logging
from datetime import datetime

class Logger:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super(Logger, cls).__new__(cls)
            log_filename = f"tensorboard_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
            file_handler = logging.FileHandler(log_filename)
            file_handler.setLevel(logging.INFO)
            logging.basicConfig(filename=log_filename,
                                level=logging.INFO,
                                handlers=file_handler,
                                format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            cls._instance.logger = logging.getLogger(__name__)
        return cls._instance

    def get_logger(self):
        return self.logger

