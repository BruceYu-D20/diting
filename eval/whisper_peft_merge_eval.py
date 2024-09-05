import os

import evaluate
from peft import PeftModel
from transformers import WhisperForConditionalGeneration, WhisperProcessor, WhisperTokenizer
from datasets import load_from_disk, concatenate_datasets, DatasetDict, Audio
from tools.utils import *
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import gc
from tools.DataCollatorSpeechSeq2SeqWithPadding import DataCollatorSpeechSeq2SeqWithPadding
import pandas as pd

# 获取所有的数据读写路径
paths = path_with_datesuffix()
print(paths)

processor = WhisperProcessor.from_pretrained(paths['MODEL_PATH'], language='ar')
print(processor)
feature_extractor = processor.feature_extractor
tokenizer = processor.tokenizer

# 数据集
dataset = load_from_disk(paths['DATASET_PATH'])['test']
dataset = dataset.rename_column('sentence', 'text')
print(dataset)
# 获取所有的checkpoint dir
checkpoint_dirs = [d for d in os.listdir(paths['MODEL_OUT_DIR']) if os.path.isdir(os.path.join(paths['MODEL_OUT_DIR'], d))]

# 如果当前目录下cr.txt存在，就删除
if os.path.exists('er.txt'):
    os.remove('er.txt')

for checkpoint_dir in checkpoint_dirs:
    # 加载wer metrics
    metric_wer = evaluate.load(os.path.join(paths['METRICS_PATH'], "wer"))
    metric_cer = evaluate.load(os.path.join(paths['METRICS_PATH'], "cer"))
    # 加载本地checkpoint，此dir是peft的参数
    base_model = WhisperForConditionalGeneration.from_pretrained(
        paths['MODEL_PATH'],
        low_cpu_mem_usage=True,
        torch_dtype=torch.float32,  # 在ct2转换时，如果指定的--quantization float16，这里必须时float16
        device_map="auto"
    )
    peft_model = PeftModel.from_pretrained(base_model, os.path.join(paths['MODEL_OUT_DIR'], checkpoint_dir))
    model = peft_model.merge_and_unload()
    #
    predictions = []
    references = []
    # 模型评估
    model.eval()
    for sample in tqdm(dataset):
        audio = sample['audio']
        inputs = processor(
            audio["array"],
            sampling_rate=audio["sampling_rate"],
            return_tensors="pt")
        inputs.to(device='cuda')

        output = base_model.generate(**inputs)
        predictions.append(processor.batch_decode(output, skip_special_tokens=True, normalize=True)[0])
        references.append(processor.tokenizer._normalize(sample["text"]))

    result = pd.DataFrame({"predictions": predictions, "references": references})

    metric_wer = evaluate.load(os.path.join(paths['METRICS_PATH'], "wer"))
    wer = metric_wer.compute(predictions=predictions, references=references)
    cer = metric_cer.compute(predictions=predictions, references=references)
    print(f'{checkpoint_dir} -- {wer} -- {cer}')
    
