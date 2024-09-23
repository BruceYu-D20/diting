#!/usr/bin/python
# -*- coding: utf-8 -*-
import yaml
import os
from util.utils import *
from datasets import load_dataset, DownloadMode
import datacsv_check

'''
业务上，微调的数据要求提供：语音文件和语音文件对应的文本文件

业务上，语音文件存储在
/data/audio/{languege}/
|-- batch_1.xlxs
|-- batch_1/
    |-- audio_1.wav
    |-- audio_2.wav
    |-- audio_3.wav
    |-- ......
|-- batch_2.xlsx
|-- batch_2/
    |-- audio_4.wav
    |-- audio_5.wav
    |-- audio_6.wav
    |-- ......

eg.
在/data/audio/en 目录下，有batch_1.xlsx文件
要求填写: 字段1：语音文件的路径 字段2：语音文件对应的文字 字段3：采样率
在/data/audio/en/batch_1/ 目录下，有audio_1.wav、audio_2.wav、audio_3.wav等语音文件
/data/audio/en/batch_1/audio_1.wav  'this is a script process original audio' 16000
/data/audio/en/batch_1/audio_2.wav  'jingle bell jingle bell jingle all the way' 16000
/data/audio/en/batch_1/audio_3.wav  'this is peppa pig' 16000

说明：
1. 每个batch文件夹，在存储完成后，运行转化脚本，会将audio的array存储到新的文件夹中。
所以，batch一旦执行脚本，请不要在向文件夹中添加新的语音文件。
如果有新的语音文件，请新建一个目录，将新的语音文件存到里面
2. 根据excel去找文件夹，所以excel和文件夹要有相同的名称，且在同一目录下
3. 文件夹下所有excel对应的文件夹都会被处理，所以excel修改好之后，再放到对应目录下

执行流程：
1. 获取到当前目录下的excel文件（xlxs格式）

输出的音频：
path Audio sentence locale
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
