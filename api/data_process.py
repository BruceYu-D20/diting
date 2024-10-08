#!/usr/bin/python
# -*- coding: utf-8 -*-
from typing import Union
from datasets import DatasetDict, Dataset

class data_process:
    # 获取数据的接口类，其他获取数据的接口都继承自该类
    def fetch_data(self) -> Union[DatasetDict, Dataset]:
        pass

def fetch_ftdata(dp: data_process):
    dp.fetch_data()

