import os
from datetime import datetime
import time
from util import common
import functools
import re

'''
获取当天时间.当前时间对当天零点的秒数
'''
@functools.lru_cache(maxsize=1)
def _dir_sufix():
    current_date = datetime.now().strftime('%Y%m%d')
    current_time = int(time.time()) % 86400
    suffix = f"{current_date}.{current_time}"
    # suffix = '20240902.2790'
    return suffix

def create_sign_begin():
    suffix = _dir_sufix()
    # 将suffix，写入文件sign.txt
    with open(os.path.join(common.PROJECT_PATH, "sign.txt"), 'w') as f:
        f.write(suffix)

def del_sign_last():
    # 删除当前目录下的sign.txt文件
    if os.path.exists(os.path.join(common.PROJECT_PATH, "sign.txt")):
        os.remove(os.path.join(common.PROJECT_PATH, "sign.txt"))

def path_with_datesuffix(model_dir: str == None) -> dict:
    # 读取sign.txt文件的内容
    if model_dir == None:
        with open(os.path.join(common.PROJECT_PATH, "sign.txt"), 'r') as f:
            time_suffix = f.read()
    else:
        time_suffix = model_dir
    logdir_suffix = os.path.join(os.path.join(common.PROJECT_PATH, "log_dir"), time_suffix)
    modleout_suffix = os.path.join(os.path.join(common.PROJECT_PATH, "model_out"), time_suffix)
    mergemodel_suffix = os.path.join(os.path.join(common.PROJECT_PATH, "merged_model"), time_suffix)
    ct2_mergemodel_suffix = os.path.join(os.path.join(common.PROJECT_PATH, "ct2_model"), time_suffix)
    path_dict = {"DATASET_PATH": common.DATASET_PATH, "MODEL_PATH": common.MODEL_PATH, "METRICS_PATH": common.METRICS_PATH,
                 "LOGGING_DIR": logdir_suffix,"MODEL_OUT_DIR": modleout_suffix,
                 "MERGE_MODEL_SAVEPATH": mergemodel_suffix, "CT2_MERGE_MODEL_SAVEPATH": ct2_mergemodel_suffix}
    print(path_dict)
    return path_dict

def find_last_n_checkpoint(n: int):
    checkpoint_dirs = os.listdir(path_with_datesuffix()["CT2_MERGE_MODEL_SAVEPATH"])
    checkpoint_ids = [int(checkpoint_dir.split('-')[0]) for checkpoint_dir in checkpoint_dirs]
    # 保留checkpoint中最大的n个数
    checkpoint_ids = sorted(checkpoint_ids, reverse=True)[:n]
    last_n_checkpoint = [f'checkpoint-{checkpoint_id}' for checkpoint_id in checkpoint_ids]
    return last_n_checkpoint

# 去除阿拉伯文的标符
def remove_arabic_diacritics(text):
    # 匹配阿拉伯语中的标符（元音符号等）
    arabic_diacritics = re.compile(r'[\u0610-\u061A\u064B-\u065F\u06D6-\u06DC\u06DF-\u06E8\u06EA-\u06ED]')
    # 使用正则表达式去除标符
    return re.sub(arabic_diacritics, '', text)
