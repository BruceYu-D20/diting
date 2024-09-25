from datasets import load_from_disk, concatenate_datasets, DatasetDict
from util.utils import parse_core_config
import os
from pathlib import Path


def ft_fetch_data():
    '''
    获取util.common.py中的DATASET_PATH
    并加载train到数据
    '''
    config = parse_core_config()
    dataset_paths = config['dataset_paths']

    datasets_list = []
    for dataset_path in dataset_paths:
        path = dataset_path[0]
        splits = dataset_path[1]
        for split in splits:
            datasets_list.append(load_from_disk(path)[split])
    print(datasets_list)

    common_voice = DatasetDict()
    common_voice['train'] = concatenate_datasets(datasets_list)
    print(common_voice)
    return common_voice

if __name__ == '__main__':
    ft_fetch_data()