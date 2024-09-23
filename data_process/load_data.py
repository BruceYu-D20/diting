#!/usr/bin/python
# -*- coding: utf-8 -*-
import os.path
import pandas as pd
import datasets
import glob
import uuid

class UserAudioConfig(datasets.BuilderConfig):
    '''
    load_datasets不能给_split_generators传参数，
    只能将参数写入到BuilderConfig中，让GeneratorBasedBuilder从config中获取参数
    '''
    def __init__(self, data_rootpath, lauguage, **kwargs):
        super(UserAudioConfig, self).__init__(**kwargs)
        self.data_rootpath = data_rootpath
        self.lauguage = lauguage

class UserAudio(datasets.GeneratorBasedBuilder):

    BUILDER_CONFIG_CLASS = UserAudioConfig

    def _info(self):
        """
        定义数据结构
        """
        return datasets.DatasetInfo(
            description="加载微调语音数据",
            features=datasets.Features(
                {
                    "id": datasets.Value("string"),
                    "path": datasets.Value("string"),
                    "audio": datasets.features.Audio(sampling_rate=16_000),
                    "sentence": datasets.Value("string"),
                    "locale": datasets.Value("string"),
                }
            ),
        )

    def _split_generators(self, dl_manager):
        """
       根据传入的 language 参数选择相应的数据目录。
       """
        print(self.config)
        data_rootpath = self.config.data_rootpath
        lauguage = self.config.lauguage
        print(f'data_rootpath: {data_rootpath}')
        print(f'lauguage: {lauguage}')
        # 返回训练集 split，并传入数据路径和语言环境
        return [datasets.SplitGenerator(name=datasets.Split.TRAIN,
                                        gen_kwargs={
                                            "path": data_rootpath, # 传入 CSV 文件所在目录
                                            "locale": lauguage, # 传入语言环境
                                        })
                ]
    def _generate_examples(self, path, locale):
        """
        遍历指定路径下的所有 CSV 文件，并逐行生成样本数据。
        假设 CSV 文件中包含 'path' 和 'text' 字段。
        """
        csv_files = glob.glob(os.path.join(path, "*.csv"))
        print(f'csv_files: {csv_files}')
        # 逐个文件进行解析
        for csv_file in csv_files:
            df = pd.read_csv(csv_file, header=None)
            # 遍历每一行
            for _, row in df.iterrows():
                unique_id = str(uuid.uuid4()).replace('-', '')
                audio_path = row[0]
                sentence = row[1]
                # 返回数据集的每一条记录
                yield unique_id, {
                    "id": str(unique_id),  # 这里将行号作为 id
                    "path": audio_path,  # 原始的音频路径
                    "audio": audio_path,  # datasets.features.Audio 会自动加载此路径的音频
                    "sentence": sentence,
                    "locale": locale  # 使用传入的 locale 参数
                }
