#!/usr/bin/python
# -*- coding: utf-8 -*-
from datasets import load_from_disk, concatenate_datasets, DatasetDict, Audio, Dataset
from api.data_process import data_process
from util.utils import parse_core_config

class fetch_data_process(data_process):
    def fetch_data(self) -> DatasetDict:
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
                ds = (load_from_disk(path)[split]
                      .cast_column("audio", Audio(sampling_rate=16000))
                      .select_columns(['audio', 'sentence']))
                datasets_list.append(ds)
        print(datasets_list)

        common_voice = DatasetDict()
        common_voice['train'] = concatenate_datasets(datasets_list)
        print(common_voice)
        return common_voice

# def ft_fetch_data():
#     '''
#     获取util.common.py中的DATASET_PATH
#     并加载train到数据
#     '''
#     config = parse_core_config()
#     dataset_paths = config['dataset_paths']
#
#     datasets_list = []
#     for dataset_path in dataset_paths:
#         path = dataset_path[0]
#         splits = dataset_path[1]
#         for split in splits:
#             ds = (load_from_disk(path)[split]
#                   .cast_column("audio", Audio(sampling_rate=16000))
#                   .select_columns(['audio', 'sentence']))
#             datasets_list.append(ds)
#     print(datasets_list)
#
#     common_voice = DatasetDict()
#     common_voice['train'] = concatenate_datasets(datasets_list)
#     print(common_voice)
#     return common_voice

if __name__ == '__main__':
    fetch_data_process.fetch_data()