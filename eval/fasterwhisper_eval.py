import time

from faster_whisper import WhisperModel, BatchedInferencePipeline
from util.utils import *
import os
from datasets import load_from_disk, load_dataset, DatasetDict
from tqdm import tqdm
import evaluate
from pathlib import Path
import re
from multiprocessing import Pool
import numpy as np
import torch

'''
faster-whisper的基座模型eval
用于比较和checkpoint模型的错误率。

用audio.array字段进行训练
不用在处理audio字段的path；
需要提前把语音文件变成datasets.features.Audio格式；
'''

# 获取所有的数据读写路径
paths = path_with_datesuffix()
CT2_MERGE_MODEL_SAVEPATH = paths['CT2_MERGE_MODEL_SAVEPATH']
print(paths['DATASET_PATH'])

test_ds = load_from_disk(paths['DATASET_PATH'])['test']

'''
将数据集分成n份，用于多进程处理
'''
def split_ds_to_pices(n: int, ds_len: int) -> list:
    '''
    将数据集分成n份
    :param n: 份数
    :param ds_len: 数据集长度
    :return: 数据集在各split点的index
    '''
    # 计算每份的长度
    length_of_each_part = ds_len // n
    # 创建一个空列表来存储每份的最后一个元素的下标
    pice_idx = [0]
    # 遍历 al 来找到每份的最后一个元素的下标
    for i in range(n):
        # 计算每份的最后一个元素的下标
        index = (i + 1) * length_of_each_part - 1
        pice_idx.append(index)
    # 如果 al 的长度不能被 n 整除，处理剩余的元素
    if ds_len % n != 0:
        pice_idx[-1] = ds_len - 1  # 最后一份的最后一个元素是 al 的最后一个元素
    return pice_idx
# 把数据集分成4份
ds_split_idx = split_ds_to_pices(4, len(test_ds))
# 把切分的数据集放到DatasetDict中，key是split_0, split_1, split_2, split_3
splited_datasets = DatasetDict()
# zip用于组合每一份数据的起始位置和结束位置
for step, start_and_end in enumerate(zip(ds_split_idx[:-1], ds_split_idx[1:])):
    '''
    例如列表的值是[0,100,200,300,400]，会zip([0,100,200,300], [100,200,300,400])
    zip的结果是[(0,100), (100,200), (200,300), (300,400)]
    ds.select(range(0,100))就会抽取原数据集中下标0-100的数据
    '''
    splited_datasets[f'split_{step}'] = test_ds.select(range(start_and_end[0], start_and_end[1]))

# 基座模型的eval方法
def asr_eval(datasets_key: str):
    # 存储预测结果和实际结果，用于cer计算
    # 存储预测结果(去掉标符)
    predictions = []
    # 存储预测结果(保留标符)
    predictions_with_ad = []
    # 存储实际结果
    references = []
    references_with_ad = []

    # 加载模型
    model = WhisperModel(
        "Systran/faster-whisper-large-v3",
        # cache_dir='/data/huggingface/hub',
        compute_type='float16',
        num_workers=4,
        device='cuda',
        local_files_only=True,
        )

    batched_model = BatchedInferencePipeline(model=model, use_vad_model=True, chunk_length=20)

    for sample in tqdm(splited_datasets[datasets_key]):
        audio = sample['audio']['array']
        try:
            if 'locale' in sample.keys():
                language = sample['locale'] if sample['locale'] is not None else None
            else:
                language = None
            segments, info = batched_model.transcribe(
                audio = torch.tensor(audio, dtype=torch.float32),  # np.ndarray
                language=language,
                task='transcribe',
                beam_size=5,
                batch_size=20,
                initial_prompt=None,
            )
            prediction_with_ad = "".join(segment.text for segment in segments)
            reference_with_ad = sample['sentence']
            prediction = remove_arabic_diacritics(prediction_with_ad)
            reference = remove_arabic_diacritics(reference_with_ad)

            predictions_with_ad.append(prediction_with_ad)
            references_with_ad.append(reference_with_ad)
            predictions.append(prediction)
            references.append(reference)
        except Exception as e:
            print(e)
            continue

    # 测评
    metric_wer = evaluate.load(os.path.join(paths['METRICS_PATH'], "wer"))
    metric_cer = evaluate.load(os.path.join(paths['METRICS_PATH'], "cer"))
    wer = metric_wer.compute(predictions=predictions, references=references)
    cer = metric_cer.compute(predictions=predictions, references=references)
    wer_ad = metric_wer.compute(predictions=predictions_with_ad, references=references_with_ad)
    cer_ad = metric_cer.compute(predictions=predictions_with_ad, references=references_with_ad)
    print(f'faster-whisper base -- wer: {wer} -- cer: {cer}\n -- wer_ad: {wer_ad} -- cer_ad: {cer_ad}\n')
    # 将wer cer以追加的形式写道当前目录下的eval_er.txt中
    with open(os.path.join(paths['LOGGING_DIR'], "eval_er.txt"), "a") as f:
        f.write(f'faster-whisper base -- wer: {wer} -- cer: {cer}\n -- wer_ad: {wer_ad} -- cer_ad: {cer_ad}\n')
    return wer, cer, wer_ad, cer_ad

if __name__ == '__main__':
    start_time = time.time()
    data_pices = [key for key in splited_datasets.keys()]
    with Pool(processes=4) as pool:
        results = pool.map(asr_eval, data_pices)
    pool.close()
    pool.join()

    # 计算所有线程返回结果的平均值
    total_wer = sum(result[0] for result in results) / len(results)
    total_cer = sum(result[1] for result in results) / len(results)
    total_wer_ad = sum(result[2] for result in results) / len(results)
    total_cer_ad = sum(result[3] for result in results) / len(results)

    print(f"平均的 WER去标符: {total_wer} 平均的 CER去标符: {total_cer}\n")
    print(f"平均的 WER带标符: {total_wer_ad} 平均的 CER带标符: {total_cer_ad}\n")
    # 结束执行时间
    end_time = time.time()
    print(f"执行时间: {end_time - start_time}秒")

