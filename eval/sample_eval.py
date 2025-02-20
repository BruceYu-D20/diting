#!/usr/bin/python
# -*- coding: utf-8 -*-
from datasets import load_dataset, load_from_disk
from tqdm import tqdm
import evaluate
from faster_whisper import WhisperModel, BatchedInferencePipeline
from multiprocessing import Pool
from util.utils import *
import time
import torch

model = WhisperModel(
        "E:/huggingface/models/faster-whisper",
        compute_type='float16',
        num_workers=4,
        device='cuda',
        local_files_only=True)