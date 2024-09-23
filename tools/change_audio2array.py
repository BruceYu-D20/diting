#!/usr/bin/python
# -*- coding: utf-8 -*-
import yaml
import os
from util.utils import *
from datasets import load_dataset, DownloadMode
import datacsv_check

'''
将音频文件处理为datasets.features.Audio
规则看README.md的2.3和2.4章节
'''

# def parse_toolconfig(config_path):
#     with open(config_path, 'r') as file:
#         config = yaml.safe_load(file)
#     return config

def load_data(config, load_script_path):
    data_rootpath = config['change_audio2array']['data_rootpath']
    for locale, locale_config in config['change_audio2array'].items():
        # 保证locale是支持的语言
        if locale not in SUPPORT_LAUNGUAGES:
            continue
        if isinstance(locale_config, dict) and locale_config.get('enable'):
            # 拼接数据路径
            data_path = os.path.join(data_rootpath, locale)
            save_path = locale_config['save_path']
            print(data_path, save_path)
            user_ds = load_dataset(f'{load_script_path}',
                                   trust_remote_code=True,
                                   data_rootpath=data_path,
                                   lauguage=locale,
                                   download_mode=DownloadMode.FORCE_REDOWNLOAD,
                                   )
            user_ds.save_to_disk(save_path)

def main():
    # 获取配置文件路径
    current_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(current_dir, 'tool.yaml')
    # 获取load_data.py的路径
    load_data_relative_path = os.path.join(current_dir, "../data_process/load_data.py")
    load_script_path = os.path.abspath(load_data_relative_path)
    print(f"load脚本：{load_script_path}")
    # 加载tool.yaml
    config = read_yaml(config_path)
    # 验证tool.yaml文件合法性
    valid_toolyaml(config_path)
    # 验证csv文件的合法性
    datacsv_check.main(False, config)
    load_data(config, load_script_path)



if __name__ == '__main__':
    main()
