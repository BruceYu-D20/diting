from faster_whisper import WhisperModel, BatchedInferencePipeline
from tools.utils import *
import os
from datasets import load_from_disk, load_dataset
from tqdm import tqdm
import evaluate
from pathlib import Path
import re

def remove_arabic_diacritics(text):
    # 匹配阿拉伯语中的标符（元音符号等）
    arabic_diacritics = re.compile(r'[\u0610-\u061A\u064B-\u065F\u06D6-\u06DC\u06DF-\u06E8\u06EA-\u06ED]')
    # 使用正则表达式去除标符
    return re.sub(arabic_diacritics, '', text)

# 获取所有的数据读写路径
paths = path_with_datesuffix()
CT2_MERGE_MODEL_SAVEPATH = paths['CT2_MERGE_MODEL_SAVEPATH']

# 数据集加载
# ds = load_from_disk(paths['DATASET_PATH'])

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
# test_ds = ds['test'].select(range(2))
test_ds = test_ds.remove_columns(['client_id', 'audio', 'up_votes', 'down_votes', 'age', 'gender', 'accent', 'segment'])
test_ds = test_ds.map(_prepare_data)

for step, checkpoint_dir in enumerate(os.listdir(CT2_MERGE_MODEL_SAVEPATH)):
    model_path = os.path.join(CT2_MERGE_MODEL_SAVEPATH, checkpoint_dir)

    # 存储预测结果和实际结果，用于cer计算
    # 存储预测结果
    predictions = []
    # 存储实际结果
    references = []

    # 加载模型
    model = WhisperModel(
        model_path,
        compute_type='float16',
        num_workers=4,
        device='cuda',
        local_files_only=True)

    batched_model = BatchedInferencePipeline(model=model, use_vad_model=True, chunk_length=20)

    for sample in tqdm(test_ds):
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
            continue

    # 测评
    metric_wer = evaluate.load(os.path.join(paths['METRICS_PATH'], "wer"))
    metric_cer = evaluate.load(os.path.join(paths['METRICS_PATH'], "cer"))
    wer = metric_wer.compute(predictions=predictions, references=references)
    cer = metric_cer.compute(predictions=predictions, references=references)
    # 将wer cer以追加的形式写道当前目录下的eval_er.txt中
    with open(os.path.join(paths['LOGGING_DIR'], "eval_er.txt"), "a") as f:
        f.write(f'{checkpoint_dir} -- wer: {wer} -- cer: {cer}\n')
