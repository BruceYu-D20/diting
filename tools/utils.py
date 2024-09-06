import os
from datetime import datetime
import time
from tools import common
import functools

'''
获取当天时间.当前时间对当天零点的秒数
'''
@functools.lru_cache(maxsize=1)
def _dir_sufix():
    current_date = datetime.now().strftime('%Y%m%d')
    current_time = int(time.time()) % 86400
    suffix = f"{current_date}.{current_time}"
    suffix = '20240902.2790'
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

def path_with_datesuffix():
    # 读取sign.txt文件的内容
    with open(os.path.join(common.PROJECT_PATH, "sign.txt"), 'r') as f:
        time_suffix = f.read()
    logdir_suffix = os.path.join(os.path.join(common.PROJECT_PATH, "log_dir"), time_suffix)
    modleout_suffix = os.path.join(os.path.join(common.PROJECT_PATH, "model_out"), time_suffix)
    mergemodel_suffix = os.path.join(os.path.join(common.PROJECT_PATH, "merged_model"), time_suffix)
    ct2_mergemodel_suffix = os.path.join(os.path.join(common.PROJECT_PATH, "ct2_model"), time_suffix)
    path_dict = {"DATASET_PATH": common.DATASET_PATH, "MODEL_PATH": common.MODEL_PATH, "METRICS_PATH": common.METRICS_PATH,
                 "LOGGING_DIR": logdir_suffix,"MODEL_OUT_DIR": modleout_suffix,
                 "MERGE_MODEL_SAVEPATH": mergemodel_suffix, "CT2_MERGE_MODEL_SAVEPATH": ct2_mergemodel_suffix}
    print(path_dict)
    return path_dict