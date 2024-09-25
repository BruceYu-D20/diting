from datasets import load_from_disk, concatenate_datasets, DatasetDict
import data_process
from util.common import DATASET_PATH

class ft_data_process(data_process):

    def ft_fetch_data(self, data_paths: list):
        '''
        获取util.common.py中的DATASET_PATH
        并加载train到数据
        '''
        user_datasets = [load_from_disk(data_path)['train'] for data_path in data_paths]
        for user_ds in user_datasets:
            print(user_ds)
        ft_voice_data = DatasetDict()
        ft_voice_data['train'] = concatenate_datasets(user_datasets)
        print(ft_voice_data)
        return ft_voice_data

if __name__ == '__main__':
    pass