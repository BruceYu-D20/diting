import os
from datasets import load_dataset, load_from_disk
from tqdm import tqdm
import evaluate
from faster_whisper import WhisperModel, BatchedInferencePipeline
from multiprocessing import Pool
from util.utils import *
import time
from pathlib import Path
import re
import torch
import yaml

'''
微调后生成的faster-whisper模型
用于评估训练后模型的错误率
'''
def parse_evalconfig(data_type):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(current_dir, 'eval.yaml')
    config = read_yaml(config_path)
    # with open(config_path, 'r') as file:
    #     config = yaml.safe_load(file)
    print(config)
    print(type(config))
    if data_type == 'array':
        data_path = config.get("array_data_path")
    else:
        data_path = config.get("audio_data_path")
    return data_path

def process_checkpoint(checkpoint_dir, paths: dict, model_id: str, data_type: str):
    """每个进程执行的函数，处理一个 checkpoint_dir"""
    CT2_MERGE_MODEL_SAVEPATH = paths['CT2_MERGE_MODEL_SAVEPATH']
    model_path = os.path.join(CT2_MERGE_MODEL_SAVEPATH, checkpoint_dir)
    print(f"Processing model: {model_path}")

    #获取日志文件夹路径
    eval_logdir = paths['EVAL_LOGDIR']

    # 获取eval.yaml中array_data_path的值
    array_data_path = parse_evalconfig(data_type)
    print(f"array_data_path：{array_data_path}" )
    test_ds = load_from_disk(array_data_path)['test']

    # 存储预测结果和实际结果，用于cer计算
    predictions = []
    references = []
    predictions_with_ad = []
    references_with_ad = []

    # 加载模型
    model = WhisperModel(
        model_path,
        compute_type='float16',
        num_workers=4,
        device='cuda',
        local_files_only=True)

    batched_model = BatchedInferencePipeline(model=model, use_vad_model=True, chunk_length=20)

    # 逐个推理
    for step, sample in enumerate(tqdm(test_ds)):
        audio = sample['audio']['array']
        try:
            if 'locale' in sample.keys():
                language = sample['locale'] if sample['locale'] is not None else None
            else:
                language = None
            segments, info = batched_model.transcribe(
                audio = torch.tensor(audio, dtype=torch.float32),
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

            predictions.append(prediction)
            references.append(reference)
            predictions_with_ad.append(prediction_with_ad)
            references_with_ad.append(reference_with_ad)
        except Exception as e:
            print(e)

    # 测评
    metric_wer = evaluate.load(os.path.join(paths['METRICS_PATH'], "wer"))
    metric_cer = evaluate.load(os.path.join(paths['METRICS_PATH'], "cer"))
    wer = metric_wer.compute(predictions=predictions, references=references)
    cer = metric_cer.compute(predictions=predictions, references=references)
    wer_ad = metric_wer.compute(predictions=predictions_with_ad, references=references)
    cer_ad = metric_cer.compute(predictions=references_with_ad, references=references)

    print(f'Model {checkpoint_dir} -- WER: {wer}, CER: {cer} -- WER带标符: {wer_ad}, CER去标符: {cer_ad}')
    with open(os.path.join(eval_logdir, f"asr_cp_{model_id}_array.txt"), "a") as f:
        f.write(f'Model {checkpoint_dir} -- WER: {wer}, CER: {cer} -- WER_AD: {wer_ad}, CER_AD: {cer_ad}')

    return wer, cer, wer_ad, cer_ad

def main(model_id=None, data_type='array'):
    # 开始执行时间
    start_time = time.time()
    # 查看eval.yaml文件是否存在
    current_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(current_dir, 'eval.yaml')
    print(f"查看{config_path}是否存在")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"eval.yaml不存在：{config_path}")
    # 取转换后的模型地址
    paths: dict = path_with_datesuffix(model_id)
    CT2_MERGE_MODEL_SAVEPATH = paths['CT2_MERGE_MODEL_SAVEPATH']
    # 获取所有的 checkpoint_dir
    tasks = [(checkpoint_dir, paths, model_id, data_type) for checkpoint_dir in os.listdir(CT2_MERGE_MODEL_SAVEPATH)]

    # 创建进程池，启动多个进程
    with Pool(processes=4) as pool:  # 这里的 processes 参数可以根据你的 CPU 核心数进行调整
        # 向进程池分发任务
        results = pool.starmap(process_checkpoint, tasks)
        # results = pool.map(process_checkpoint, checkpoint_dirs)
    pool.close()
    pool.join()

    # 计算所有线程返回结果的平均值
    total_wer = sum(result[0] for result in results) / len(results)
    total_cer = sum(result[1] for result in results) / len(results)
    total_wer_ad = sum(result[2] for result in results) / len(results)
    total_cer_ad = sum(result[3] for result in results) / len(results)

    print(f"平均的 WER: {total_wer} 平均的 CER: {total_cer} 平均的 WER带标符: {total_wer_ad} 平均的 CER去标符: {total_cer_ad}")
    # 结束执行时间
    end_time = time.time()
    print(f"执行时间: {end_time - start_time}秒")

if __name__ == "__main__":
    model_id = '20240918.29775'
    data_type = 'array'
    main(model_id, data_type)
