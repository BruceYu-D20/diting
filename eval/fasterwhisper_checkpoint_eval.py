import os
from datasets import load_dataset, load_from_disk
from tqdm import tqdm
import evaluate
from faster_whisper import WhisperModel, BatchedInferencePipeline
from multiprocessing import Pool
from util.utils import *
import time
import torch

'''
微调后生成的faster-whisper模型
用于评估训练后模型的错误率
'''

def process_checkpoint(checkpoint_dir, paths: dict, log_file: str, eval_config, data_type: str):
    """
    每个进程执行的函数，处理一个 checkpoint_dir
    """
    CT2_MERGE_MODEL_SAVEPATH = paths['CT2_MERGE_MODEL_SAVEPATH']
    model_path = os.path.join(CT2_MERGE_MODEL_SAVEPATH, checkpoint_dir)
    print(f"Processing model: {model_path}")

    if data_type == 'array':
        data_path = eval_config.get("array_data_path")
    else:
        data_path = eval_config.get("audio_data_path")

    # 数据分片
    data_split = eval_config.get("split")

    # 获取eval.yaml中array_data_path的值
    test_ds = load_from_disk(data_path)[data_split]

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
    wer_ad = metric_wer.compute(predictions=predictions_with_ad, references=references_with_ad)
    cer_ad = metric_cer.compute(predictions=predictions_with_ad, references=references_with_ad)

    print(f'Model {checkpoint_dir} -- WER去标符: {wer}, CER去标符: {cer} -- WER带标符: {wer_ad}, CER带标符: {cer_ad}')
    with open(log_file, "a") as f:
        f.write(f'Model {checkpoint_dir} -- WER去标符: {wer}, CER去标符: {cer} -- WER带标符: {wer_ad}, CER带标符: {cer_ad}\n')

    return wer, cer, wer_ad, cer_ad

def main(model_id=None, data_type='array'):
    # 开始执行时间
    start_time = time.time()
    # 查看eval.yaml文件是否存在
    current_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(current_dir, 'eval.yaml')
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"eval.yaml不存在：{config_path}")

    # 读取eval.yaml文件
    eval_config = read_yaml(config_path)
    num_process = eval_config.get("num_process")

    # 取转换后的模型地址
    paths: dict = path_with_datesuffix(model_id)
    CT2_MERGE_MODEL_SAVEPATH = paths['CT2_MERGE_MODEL_SAVEPATH']
    # 获取日志文件夹，如果不存在就创建一个
    eval_logdir = paths['EVAL_LOGDIR']
    if not os.path.exists(eval_logdir):
        os.makedirs(eval_logdir)
    # 获取日志文件，如果存在就删除
    log_file = os.path.join(eval_logdir, f"asr_cp_{model_id}_array.txt")
    if os.path.exists(log_file) and os.path.isfile(log_file):
        print(f'{__file__}：日志存在，已删除 {log_file}')
        os.remove(log_file)
    # 获取所有的 checkpoint_dir
    tasks = [(checkpoint_dir, paths, log_file, eval_config, data_type) for checkpoint_dir in os.listdir(CT2_MERGE_MODEL_SAVEPATH)]

    # 创建进程池，启动多个进程
    with Pool(processes=num_process) as pool:  # 这里的 processes 参数可以根据你的 CPU 核心数进行调整
        # 向进程池分发任务
        results = pool.starmap(process_checkpoint, tasks)
    pool.close()
    pool.join()

    # 计算所有线程返回结果的平均值
    total_wer = sum(result[0] for result in results) / len(results)
    total_cer = sum(result[1] for result in results) / len(results)
    total_wer_ad = sum(result[2] for result in results) / len(results)
    total_cer_ad = sum(result[3] for result in results) / len(results)

    print(f"平均的 WER去标符: {total_wer} 平均的 CER去标符: {total_cer} 平均的 WER带标符: {total_wer_ad} 平均的 CER带标符: {total_cer_ad}")
    with open(log_file, "a") as f:
        f.write(f"平均的 WER去标符: {total_wer} 平均的 CER去标符: {total_cer} 平均的 WER带标符: {total_wer_ad} 平均的 CER带标符: {total_cer_ad}\n")
    print(f'日志已保存到: {log_file}')
    # 结束执行时间
    end_time = time.time()
    print(f"执行时间: {end_time - start_time}秒")

if __name__ == "__main__":
    model_id = '20240918.29775'
    data_type = 'array'
    main(model_id, data_type)
