import os
from datetime import datetime
import time
import functools
import re
import yaml
from pathlib import Path

'''
支持的语言列表
再change_audio2array.py处理数据时，会验证配置
'''
SUPPORT_LAUNGUAGES = ['ar', 'en', 'he', 'zh']

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
    config = parse_core_config()
    suffix = _dir_sufix()
    # 将suffix，写入文件sign.txt
    with open(os.path.join(config['project_path'], "sign.txt"), 'w') as f:
        f.write(suffix)

def del_sign_last():
    config = parse_core_config()
    # 删除当前目录下的sign.txt文件
    if os.path.exists(os.path.join(config['project_path'], "sign.txt")):
        os.remove(os.path.join(config['project_path'], "sign.txt"))

def path_with_datesuffix(model_dir: str=None) -> dict:
    # 解析core.yaml
    config = parse_core_config()
    # 读取sign.txt文件的内容
    if model_dir == None:
        with open(os.path.join(config['project_path'], "sign.txt"), 'r') as f:
            task_id = f.read()
    else:
        task_id = model_dir

    # 一级目录
    logdir_suffix = os.path.join(os.path.join(config['project_path'], "log_dir"), task_id)
    modleout_suffix = os.path.join(os.path.join(config['project_path'], "model_out"), task_id)
    mergemodel_suffix = os.path.join(os.path.join(config['project_path'], "merged_model"), task_id)
    ct2_mergemodel_suffix = os.path.join(os.path.join(config['project_path'], "ct2_model"), task_id)
    # model_id下的目录
    tensorboard_logdir = os.path.join(logdir_suffix, "tensor_log")
    eval_logdir = os.path.join(logdir_suffix, "eval_log")
    path_dict = {
         "MODEL_PATH": config['model_path'],
         "METRICS_PATH": config['metrics_path'],
         "LOGGING_DIR": logdir_suffix,
         "MODEL_OUT_DIR": modleout_suffix,
         "MERGE_MODEL_SAVEPATH": mergemodel_suffix,
         "CT2_MERGE_MODEL_SAVEPATH": ct2_mergemodel_suffix,
         "TENSORBOARD_LOGDIR": tensorboard_logdir,
         "EVAL_LOGDIR": eval_logdir
    }
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

def read_yaml(config_path):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def parse_core_config():
    '''
    解析core/config.yaml文件，返回一个字典，包含以下键值对：
    {
        'dataset_path': [(dataset_name1, split1), (dataset_name2, split2), ...],
        'model_path': model_path,
        'metrics_path': metrics_path,
        'project_path': project_path
    }
    '''
    current_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.abspath(os.path.join(current_dir, '../core/core.yaml'))
    config_path = Path(config_path).as_posix()

    config = read_yaml(config_path)

    core_yaml_dict = {}
    dataset_path = []

    dataset_paths = config['dataset_paths']
    name_index = [i for i in range(0, len(dataset_paths)) if i % 2 == 0]
    for i in name_index:
        dataset_path.append((dataset_paths[i]['name'], dataset_paths[i+1]['split']))

    core_yaml_dict['dataset_paths'] = dataset_path
    core_yaml_dict['model_path'] = config['model_path']
    core_yaml_dict['metrics_path'] = config['metrics_path']
    core_yaml_dict['project_path'] = config['metrics_path']
    return core_yaml_dict

def valid_tool_config(config_path):
    '''
    检查tools/tool.yaml的配置项是否合法
    1. 文件必须存在
    2. 配置项必须包含change_audio2array
    3. 配置项change_audio2array必须包含data_rootpath
    4. 配置项change_audio2array中包含的语种必须在SUPPORT_LAUNGUAGE之中
    '''
    # 检查文件是否存在
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"{config_path} 不存在，请检查环境")
    # 检查值是否合法
    config = read_yaml(config_path)
    if not 'change_audio2array' in config.keys():
        raise ValueError("tools/tool.yaml的配置项必须包含change_audio2array")
    # 检查是否包含data_rootpath
    if 'data_rootpath' not in config['change_audio2array'].keys():
        raise ValueError("tools/tool.yaml的配置项change_audio2array必须包含data_rootpath")
    # 检查是否是支持的语种
    config_laungs = [laung for laung in config['change_audio2array'].keys() if laung != 'data_rootpath' and laung in SUPPORT_LAUNGUAGES]
    if len(config_laungs) == 0:
        raise ValueError(f"tools/tool.yaml的配置项change_audio2array中必须包含语种文件夹 {SUPPORT_LAUNGUAGES}")

    for laung in config['change_audio2array'].keys():
        if laung == 'data_rootpath':
            continue
        if laung not in SUPPORT_LAUNGUAGES:
            raise ValueError(f"tools/tool.yaml的配置项change_audio2array中包含不支持的语种文件夹: {laung}")








