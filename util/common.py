#!/usr/bin/python
# -*- coding: utf-8 -*-

'''
DATASET_PATH：数据的地址
MODEL_PATH: 离线模型的地址
WER_PATH：wer代码的地址
LOGGING_DIR：日志目录
MODEL_OUT_DIR：model checkout目录
PEFT_MODEL_ID：被merge peft checkpoint id
MERGE_MODEL_SAVEPATH：peft merge后，model的存储地址
CT2_MERGE_MODEL_SAVEPATH：ct2转化后模型的存储地址
'''
# DATASET_PATH = "E:/huggingface/datasets/common_voice_datasets/common-17_arsub"
DATASET_PATH = "E:/huggingface/common_17_ar"
MODEL_PATH = "E:/huggingface/models/whisper-tiny"
METRICS_PATH = "E:/huggingface/metrics"
PROJECT_PATH = "E:/code/diting"