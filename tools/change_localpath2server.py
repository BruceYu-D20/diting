#!/usr/bin/python
# -*- coding: utf-8 -*-
import argparse
from datasets import load_dataset, load_from_disk
from pathlib import Path
import os

'''
流程：
1. 从huggingface上下载数据
2. 将数据save_to_disk到硬盘上，此时，path是本机的绝对路径
3. 将下载的数据和2存储的arrow一起放到服务器上
4. 服务器读取2的arrow，将path变成服务器上的downloads文件夹地址

说明：
为什么不在本机下载之后就直接map改path，再存储下来？
服务器端磁盘空间大，防止本地磁盘资源不足的情况发生
'''

# DATA_CACHE_DIR 在服务器端不要改动
DATA_CACHE_DIR = 'E:/huggingface/common_17_ar'
# 本机存储arrow的地址
# SAVED_ARROW_PATH = os.path.join(DATA_CACHE_DIR, "save_arrow")
SAVED_ARROW_PATH = 'F:/common_17/save_arrow/en'
# 要修改成的服务器地址，下一层就是downloads文件夹
DATA_LINUX_PATH = '/data/huggingface/test_modify/17_dataset'
# 服务器新生成arrow的地址
SAVED_ARROW_PATH_NEW = os.path.join(DATA_LINUX_PATH, "save_arrow_changed")

def parse_args():
    parser = argparse.ArgumentParser(description='接收一个 --env 参数')
    parser.add_argument('--env', action='store_true', help='local 或者 server，代表是本机下载步骤和服务器处理path步骤')
    args = parser.parse_args()
    env_arg = args.env
    print(f'传入参数是： {env_arg}')
    if not env_arg in ['local', 'server']:
        raise ValueError("--env 的值必须是local 或 server")
    return args

def _modify_path_with_server_sample(sample):
    path = Path(sample['path'])
    path = path.as_posix()
    linux_path = path.replace('\\', '/')
    linux_path = linux_path.replace(DATA_CACHE_DIR, DATA_LINUX_PATH)
    sample['path'] = linux_path
    return sample

def download_and_save_arrow():
    '''
    如果保存的文件夹存在，不会重复下载
    先核实
    '''
    if not os.path.exists(SAVED_ARROW_PATH):
        en_datasets = load_dataset(
            path='mozilla-foundation/common_voice_17_0',
            name='en',
            cache_dir=DATA_CACHE_DIR,
            trust_remote_code=True,
            token='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',
        )
        en_datasets.save_to_disk(SAVED_ARROW_PATH, num_proc=8)
    else:
        print(f'{SAVED_ARROW_PATH}存在，读取缓存数据')

def process_path():
    '''
    如果转换后的arrow文件夹存在，不会执行
    先核实
    '''
    if not os.path.exists(SAVED_ARROW_PATH_NEW):
        en_datasets = load_from_disk(SAVED_ARROW_PATH)
        print(en_datasets['test'][:1])
        modified_ds = en_datasets.map(_modify_path_with_server_sample)
        print(modified_ds['test'][:1])
        modified_ds.save_to_disk(SAVED_ARROW_PATH_NEW)
    else:
        print(f'{SAVED_ARROW_PATH_NEW}存在，先检查')

def main():
    args = parse_args()
    if args == 'local':
        download_and_save_arrow()
    elif args == 'server':
        process_path()
    else:
        raise ValueError('--env的值，只能传入local或server')

def local_test():
    download_and_save_arrow()

if __name__ == '__main__':
    local_test()
