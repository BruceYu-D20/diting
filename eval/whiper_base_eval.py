from evaluate import evaluator
from datasets import Audio, load_from_disk,DatasetDict,concatenate_datasets
from tools.utils import *
from transformers import WhisperForConditionalGeneration, WhisperProcessor, AutoModelForSpeechSeq2Seq
import torch
from tqdm import tqdm
import pandas as pd
from datasets import load_dataset
import evaluate

'''
基座模型

wer: 0.276422
'''

# 获取所有的数据读写路径
paths = path_with_datesuffix()
print(paths)

'''
dataset = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
print(dataset)
print(dataset[:1])
'''

dataset = load_from_disk(paths['DATASET_PATH'])['test']
dataset = dataset.rename_column('sentence', 'text')
print(dataset)


base_model = AutoModelForSpeechSeq2Seq.from_pretrained(
        paths['MODEL_PATH'],
        low_cpu_mem_usage=True,
        torch_dtype=torch.float32, # 在ct2转换时，如果指定的--quantization float16，这里必须时float16
        attn_implementation="sdpa",
        device_map='cuda'
    )
print(base_model)

processor = WhisperProcessor.from_pretrained(paths['MODEL_PATH'], language='ar', task='transcribe')
predictions = []
references = []

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

# print(result)

metric_wer = evaluate.load(os.path.join(paths['METRICS_PATH'], "wer"))
metric_cer = evaluate.load(os.path.join(paths['METRICS_PATH'], "cer"))
wer = metric_wer.compute(predictions=predictions, references=references)
cer = metric_cer.compute(predictions=predictions, references=references)
print(f'wer: {wer} -- cer: {cer}')
