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

# 基座模型的eval方法
def asr_eval(ds, paths, data_type, model_path):
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
        model_path,
        compute_type='float16',
        num_workers=4,
        device='cuda',
        local_files_only=True,
        )

    batched_model = BatchedInferencePipeline(model=model, use_vad_model=True, chunk_length=20)

    for sample in tqdm(ds):
        if data_type == 'array':
            audio = torch.tensor(sample['audio']['array'], dtype=torch.float32)
        else:
            audio = sample['path']
        try:
            if 'locale' in sample.keys():
                language = sample['locale'] if sample['locale'] is not None else None
            else:
                language = None
            segments, info = batched_model.transcribe(
                audio = audio,  # np.ndarray
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
    return wer, cer, wer_ad, cer_ad

def main(data_type='array'):
    start_time = time.time()
    # 查看eval.yaml文件是否存在
    current_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(current_dir, 'eval.yaml')
    if not os.path.exists(config_path):
        # 检查配置文件是否存在
        raise FileNotFoundError(f"eval.yaml不存在：{config_path}")
    eval_config = read_yaml(config_path)
    if data_type == 'array':
        data_path = eval_config.get("array_data_path")
    else:
        data_path = eval_config.get("audio_data_path")
    num_process = eval_config.get("num_process")
    model_path = eval_config.get("faster_whisper")

    # 获取所有的数据读写路径
    paths = path_with_datesuffix("0")

    split = eval_config.get("split")
    # 加载数据集，把数据分成多分
    test_ds = load_from_disk(data_path)[split]
    ds_split_idx = split_ds_to_pices(num_process, len(test_ds))
    splited_datasets = DatasetDict()
    for step, start_and_end in enumerate(zip(ds_split_idx[:-1], ds_split_idx[1:])):
        '''
        例如列表的值是[0,100,200,300,400]，会zip([0,100,200,300], [100,200,300,400])
        zip的结果是[(0,100), (100,200), (200,300), (300,400)]
        ds.select(range(0,100))就会抽取原数据集中下标0-100的数据
        '''
        splited_datasets[f'split_{step}'] = test_ds.select(range(start_and_end[0], start_and_end[1]))
    data_pices = [(splited, paths, data_type, model_path) for splited in splited_datasets.values()]

    with Pool(processes=num_process) as pool:
        results = pool.starmap(asr_eval, data_pices)
    pool.close()
    pool.join()

    # 计算所有线程返回结果的平均值
    total_wer = sum(result[0] for result in results) / len(results)
    total_cer = sum(result[1] for result in results) / len(results)
    total_wer_ad = sum(result[2] for result in results) / len(results)
    total_cer_ad = sum(result[3] for result in results) / len(results)

    print(f"平均的 WER去标符: {total_wer} 平均的 CER去标符: {total_cer} 平均的 WER带标符: {total_wer_ad} 平均的 CER带标符: {total_cer_ad}\n")
    if not os.path.exists(paths['LOGGING_DIR']):
        os.makedirs(paths['LOGGING_DIR'])
    log_file = os.path.join(paths['LOGGING_DIR'], f"eval_er_{start_time}.txt")
    with open(log_file, "a") as f:
        f.write(f"平均的 WER去标符: {total_wer} 平均的 CER去标符: {total_cer} 平均的 WER带标符: {total_wer_ad} 平均的 CER带标符: {total_cer_ad}\n")
    print(f"{__file__}: 日志已保存到: {log_file}")
    # 结束执行时间
    end_time = time.time()
    print(f"执行时间: {end_time - start_time}秒")
    pass

if __name__ == '__main__':
    data_type = 'array'
    main(data_type)
