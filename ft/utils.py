import os
from datetime import datetime
import time
import common
import functools

'''
获取当天时间.当前时间对当天零点的秒数
'''
@functools.lru_cache(maxsize=1)
def _dir_sufix():
    current_date = datetime.now().strftime('%Y%m%d')
    current_time = int(time.time()) % 86400
    suffix = f"{current_date}.{current_time}"
    return suffix

def path_with_datesuffix():
    time_suffix = _dir_sufix()
    logdir_suffix = os.path.join(os.path.join(common.PROJECT_PATH, "log_dir"), time_suffix)
    modleout_suffix = os.path.join(os.path.join(common.PROJECT_PATH, "model_out"), time_suffix)
    mergemodel_suffix = os.path.join(os.path.join(common.PROJECT_PATH, "merged_model"), time_suffix)
    ct2_mergemodel_suffix = os.path.join(os.path.join(common.PROJECT_PATH, "ct2_model"), time_suffix)
    path_dict = {"DATASET_PATH": common.DATASET_PATH, "MODEL_PATH": common.MODEL_PATH,"METRICS_PATH": common.METRICS_PATH,
                 "LOGGING_DIR": logdir_suffix,"MODEL_OUT_DIR": modleout_suffix, "PEFT_MODEL_ID": common.PEFT_MODEL_ID,
                 "MERGE_MODEL_SAVEPATH": mergemodel_suffix, "CT2_MERGE_MODEL_SAVEPATH": ct2_mergemodel_suffix}
    print(path_dict)
    return path_dict

path_with_datesuffix()
from time import sleep
sleep(2)
path_with_datesuffix()
