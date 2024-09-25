import os
from datasets import load_dataset
from tqdm import tqdm
import evaluate
from faster_whisper import WhisperModel, BatchedInferencePipeline
from multiprocessing import Pool
from util.utils import *
import time
from pathlib import Path
import re
import yaml

'''
微调后生成的faster-whisper模型
用于评估训练后模型的错误率

@deprecated 合并进fasterwhisper_checkpoint_eval.py 在--data_type audio时会用语音文件进行微调

'''
# 获取所有的数据读写路径

def parse_evalconfig():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(current_dir, 'eval.yaml')
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    print(config)
    print(type(config))
    audio_data_path = config.get("audio_data_path")
    return audio_data_path

# 加载数据集
def _prepare_data(sample):
    path = Path(sample['path'])
    path = path.as_posix()
    linux_path = path.replace('\\', '/')
    linux_path = linux_path.replace('E:', '/data')
    sample['linux_path'] = linux_path
    return sample


def _prepare_data(sample):
    path = Path(sample['path'])
    path = path.as_posix()
    linux_path = path.replace('\\', '/')
    linux_path = linux_path.replace('E:', '/data')
    sample['linux_path'] = linux_path
    return sample

def process_checkpoint(checkpoint_dir, paths: dict, model_id: str):
    """每个进程执行的函数，处理一个 checkpoint_dir"""
    CT2_MERGE_MODEL_SAVEPATH = paths['CT2_MERGE_MODEL_SAVEPATH']
    model_path = os.path.join(CT2_MERGE_MODEL_SAVEPATH, checkpoint_dir)
    print(f"Processing model: {model_path}")

    # 获取eval.yaml中array_data_path的值
    audio_data_path = parse_evalconfig()
    print(f"audio_data_path：{audio_data_path}")
    test_ds = load_dataset('mozilla-foundation/common_voice_17_0',
                           'ar',
                           cache_dir=audio_data_path,
                           )['test']
    test_ds = test_ds.remove_columns(
        ['client_id', 'audio', 'up_votes', 'down_votes', 'age', 'gender', 'accent', 'segment', 'variant'])
    test_ds = test_ds.map(_prepare_data)
    print(test_ds)

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
        try:
            if 'locale' in sample.keys():
                language = sample['locale'] if sample['locale'] is not None else None
            else:
                language = None
            segments, info = batched_model.transcribe(
                sample['linux_path'],
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

    print(f'Model {checkpoint_dir} -- WER: {wer}, CER: {cer} --wer_ad: {wer_ad}, cer_ad: {cer_ad}')
    with open(os.path.join(paths['LOGGING_DIR'], f"eval/asr_cp_{model_id}_audio.txt"), "a") as f:
        f.write(f'{checkpoint_dir} -- wer: {wer} -- cer: {cer} --wer_ad: {wer_ad}, cer_ad: {cer_ad}\n')

    return wer, cer, wer_ad, cer_ad

def main(model_id=None):
    # 开始执行时间
    start_time = time.time()
    # 查看eval.yaml文件是否存在
    current_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(current_dir, 'eval.yaml')
    print(f"查看{config_path}是否存在")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"eval.yaml不存在：{config_path}")
    # 获取所有的 checkpoint_dir
    paths = path_with_datesuffix(model_id)
    CT2_MERGE_MODEL_SAVEPATH = paths['CT2_MERGE_MODEL_SAVEPATH']
    tasks = [(checkpoint_dir, paths, model_id) for checkpoint_dir in os.listdir(CT2_MERGE_MODEL_SAVEPATH)]

    # 创建进程池，启动多个进程
    with Pool(processes=2) as pool:  # 这里的 processes 参数可以根据你的 CPU 核心数进行调整
        # 向进程池分发任务
        results = pool.map(process_checkpoint, tasks)
    pool.close()
    pool.join()

    # 计算所有线程返回结果的平均值
    total_wer = sum(result[0] for result in results) / len(results)
    total_cer = sum(result[1] for result in results) / len(results)
    total_wer_ad = sum(result[2] for result in results) / len(results)
    total_cer_ad = sum(result[3] for result in results) / len(results)

    print(f"平均的 WER: {total_wer} 平均的 CER: {total_cer} 平均的 wer_ad: {total_wer_ad} 平均的 cer_ad: {total_cer_ad}")
    # 结束执行时间
    end_time = time.time()
    print(f"执行时间: {end_time - start_time}秒")

if __name__ == "__main__":
    model_id=None
    main(model_id)
