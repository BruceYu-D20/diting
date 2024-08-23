#!/usr/bin/python
# -*- coding: utf-8 -*-
#
# 代码功能：
# 合并peft checkpoint和基座模型的权重

from peft import PeftModel, PeftConfig
from transformers import WhisperForConditionalGeneration, Seq2SeqTrainer, BitsAndBytesConfig, WhisperTokenizer
from common import *
import os
import shutil
import torch

'''
拷贝['tokenizer.json', "preprocessor_config.json"]到merge目录下
'''
def _files_base_to_merge():
    wait_for_copy = ['tokenizer.json', "preprocessor_config.json"]
    sourcedir_files = [f for f in wait_for_copy if os.path.isfile(os.path.join(MODEL_PATH, f))]
    mergedir_files = [f for f in wait_for_copy if os.path.isfile(os.path.join(MERGE_MODEL_SAVEPATH, f))]
    # 找到原模型文件夹中有，但merge文件夹中没有的文件
    diff_files = [sf for sf in sourcedir_files if sf not in mergedir_files]
    print(diff_files)
    # 用于最后验证，是否全部拷完
    file_diff_num = len(diff_files)
    # 结束后被拷贝的文件数量
    copy_file_num = 0
    for file_wait_copy in diff_files:
        try:
            from_file = os.path.join(MODEL_PATH, file_wait_copy)
            to_file = os.path.join(MERGE_MODEL_SAVEPATH, file_wait_copy)
            shutil.copyfile(from_file, to_file)
            copy_file_num = copy_file_num + 1
        except Exception:
            print(f"{file_wait_copy} 拷贝失败，请检查")
    if(copy_file_num != file_diff_num):
        print("文件没有全部拷贝，请检查")

'''
合并基座模型和peft checkpoint模型
'''
def _merge_model():

    base_model = WhisperForConditionalGeneration.from_pretrained(
        MODEL_PATH,
        low_cpu_mem_usage=True,
        torch_dtype=torch.float32, # 在ct2转换时，如果指定的--quantization float16，这里必须时float16
        device_map="auto"
    )
    peft_model = PeftModel.from_pretrained(base_model, os.path.join(MODEL_OUT_DIR, os.path.join(MODEL_OUT_DIR, PEFT_MODEL_ID)))
    model = peft_model.merge_and_unload()

    tokenizer = WhisperTokenizer.from_pretrained(MODEL_PATH)
    # 保存合并后的模型
    model.save_pretrained(MERGE_MODEL_SAVEPATH)
    tokenizer.save_pretrained(MERGE_MODEL_SAVEPATH)

def merge_peft_model():
    _merge_model()
    _files_base_to_merge()

