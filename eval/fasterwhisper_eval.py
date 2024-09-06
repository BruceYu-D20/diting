from faster_whisper import WhisperModel, BatchedInferencePipeline
from tools.utils import path_with_datesuffix
import os
from datasets import load_from_disk
from tqdm import tqdm
import evaluate

# 获取所有的数据读写路径
paths = path_with_datesuffix()
CT2_MERGE_MODEL_SAVEPATH = paths['CT2_MERGE_MODEL_SAVEPATH']

# 数据集加载
ds = load_from_disk(paths['DATASET_PATH'])
test_ds = ds['test'].select(range(2))
print(test_ds)

for step, checkpoint_dir in enumerate(os.listdir(CT2_MERGE_MODEL_SAVEPATH)):
    # test
    if step > 1:
        break

    print(f'{step} -- {checkpoint_dir}')
    model_path = os.path.join(CT2_MERGE_MODEL_SAVEPATH, checkpoint_dir)

    # 存储预测结果和实际结果，用于cer计算
    # 存储预测结果
    predictions = []
    # 存储实际结果
    references = []

    # 加载模型
    # model = WhisperModel(model_path, device="cuda")
    model = WhisperModel(
        model_path,
        compute_type='float16',
        num_workers=4,
        local_files_only=True)

    batched_model = BatchedInferencePipeline(model=model, use_vad_model=True, chunk_length=20)

    for sample in tqdm(test_ds):
        language = sample['locale'] if sample['locale'] is not None else None
        segments, info = batched_model.transcribe(
            sample['path'],
            language=language,
            task='transcribe',
            beam_size=5,
            batch_size=20,
            initial_prompt=None,
        )
        prediction = "".join(segment.text for segment in segments)
        reference = sample['sentence']

        predictions.append(prediction)
        references.append(reference)

    # test
    for pre,ref in zip(predictions, references):
        print(f'pre: {pre} -- ref: {ref}')
    # 测评
    '''
    metric_wer = evaluate.load(os.path.join(paths['METRICS_PATH'], "wer"))
    metric_cer = evaluate.load(os.path.join(paths['METRICS_PATH'], "cer"))
    wer = metric_wer.compute(predictions=predictions, references=references)
    cer = metric_cer.compute(predictions=predictions, references=references)
    print(f'wer: {wer} -- cer: {cer}')
    '''


