import os
from datasets import load_dataset
from tqdm import tqdm
import evaluate
from faster_whisper import WhisperModel, BatchedInferencePipeline
from multiprocessing import Pool
from tools.utils import *
import time
from pathlib import Path
import re

'''
微调后生成的faster-whisper模型
用于评估训练后模型的错误率
'''
# 获取所有的数据读写路径
paths = path_with_datesuffix()
CT2_MERGE_MODEL_SAVEPATH = paths['CT2_MERGE_MODEL_SAVEPATH']
print(paths['DATASET_PATH'])

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

print(paths['DATASET_PATH'])
test_ds = load_dataset('mozilla-foundation/common_voice_17_0',
                       'ar',
                       cache_dir=paths['DATASET_PATH'],
                       )['test']
test_ds = test_ds.remove_columns(['client_id', 'audio', 'up_votes', 'down_votes', 'age', 'gender', 'accent', 'segment', 'variant'])
test_ds = test_ds.map(_prepare_data)
print(test_ds)
print(test_ds[:1])

def process_checkpoint(checkpoint_dir):
    """每个进程执行的函数，处理一个 checkpoint_dir"""
    model_path = os.path.join(CT2_MERGE_MODEL_SAVEPATH, checkpoint_dir)
    print(f"Processing model: {model_path}")

    # 存储预测结果和实际结果，用于cer计算
    predictions = []
    references = []

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
            language = sample['locale'] if sample['locale'] is not None else None
            segments, info = batched_model.transcribe(
                sample['linux_path'],
                language=language,
                task='transcribe',
                beam_size=5,
                batch_size=20,
                initial_prompt=None,
            )
            prediction = remove_arabic_diacritics("".join(segment.text for segment in segments))
            reference = remove_arabic_diacritics(sample['sentence'])

            predictions.append(prediction)
            references.append(reference)
        except Exception as e:
            print(e)

    # 测评
    metric_wer = evaluate.load(os.path.join(paths['METRICS_PATH'], "wer"))
    metric_cer = evaluate.load(os.path.join(paths['METRICS_PATH'], "cer"))
    wer = metric_wer.compute(predictions=predictions, references=references)
    cer = metric_cer.compute(predictions=predictions, references=references)

    print(f'Model {checkpoint_dir} -- WER: {wer}, CER: {cer}')
    with open(os.path.join(paths['LOGGING_DIR'], "eval_er_2.txt"), "a") as f:
        f.write(f'{checkpoint_dir} -- wer: {wer} -- cer: {cer}\n')

    return wer, cer

if __name__ == "__main__":
    #开始执行时间
    start_time = time.time()
    # 获取所有的 checkpoint_dir
    checkpoint_dirs = os.listdir(CT2_MERGE_MODEL_SAVEPATH)

    # 创建进程池，启动多个进程
    with Pool(processes=2) as pool:  # 这里的 processes 参数可以根据你的 CPU 核心数进行调整
        # 向进程池分发任务
        results = pool.map(process_checkpoint, checkpoint_dirs)
    pool.close()
    pool.join()

    # 计算所有线程返回结果的平均值
    total_wer = sum(result[0] for result in results) / len(results)
    total_cer = sum(result[1] for result in results) / len(results)

    print(f"平均的 WER: {total_wer} 平均的 CER: {total_cer}")
    # 结束执行时间
    end_time = time.time()
    print(f"执行时间: {end_time - start_time}秒")
