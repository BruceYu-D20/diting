import evaluate
from peft import PeftModel
from transformers import WhisperForConditionalGeneration, WhisperProcessor
from datasets import load_from_disk, concatenate_datasets, DatasetDict, Audio
from tools.utils import *
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import gc
from tools.DataCollatorSpeechSeq2SeqWithPadding import DataCollatorSpeechSeq2SeqWithPadding


# 获取所有的数据读写路径
paths = path_with_datesuffix()

base_model = WhisperForConditionalGeneration.from_pretrained(
        paths['MODEL_PATH'],
        low_cpu_mem_usage=True,
        torch_dtype=torch.float32, # 在ct2转换时，如果指定的--quantization float16，这里必须时float16
        device_map="auto"
    )
# model = PeftModel.from_pretrained(base_model, os.path.join(paths['MODEL_OUT_DIR'], paths['PEFT_MODEL_ID']))
model = PeftModel.from_pretrained(base_model, "E:/code/whisper_ft/model_out/20240828.29311/checkpoint-2")
processor = WhisperProcessor.from_pretrained(paths['MODEL_PATH'], language='ar', task='transcribe')
feature_extractor = processor.feature_extractor
tokenizer = processor.tokenizer

# 数据集
def _prepare_dataset(batch):
    audio = batch["audio"]
    # compute log-Mel input features from input audio array
    batch["input_features"] = feature_extractor(audio["array"], sampling_rate=audio["sampling_rate"]).input_features[0]
    # encode target text to label ids
    batch["labels"] = tokenizer(batch["sentence"]).input_ids
    return batch

ds = load_from_disk(paths['DATASET_PATH'])
common_voice = DatasetDict()
common_voice['train'] = concatenate_datasets([ds['train'], ds['validation']])
common_voice['test'] = ds['test']
common_voice = common_voice.cast_column("audio", Audio(sampling_rate=16000))
common_voice = common_voice.map(_prepare_dataset, remove_columns=common_voice.column_names["train"])

# 拼装DataLoader
data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)
eval_dataloader = DataLoader(common_voice["test"], batch_size=8, collate_fn=data_collator)

# 加载wer metrics
metric_wer = evaluate.load(os.path.join(paths['METRICS_PATH'], "wer"))

# 模型评估
model.eval()
for step, batch in enumerate(tqdm(eval_dataloader)):
    with torch.cuda.amp.autocast():
        with torch.no_grad():
            generated_tokens = (
                model.generate(
                    input_features=batch["input_features"].to('cuda'),
                    decoder_input_ids=batch["labels"][:, :4].to('cuda'),
                    max_new_tokens=255,
                )
                .cpu()
                .numpy()
            )
            labels = batch["labels"].cpu().numpy()
            labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
            decoded_preds = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
            decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
            metric_wer.add_batch(
                predictions=decoded_preds,
                references=decoded_labels,
            )
    del generated_tokens, labels, batch
    gc.collect()
wer = 100 * metric_wer.compute()
print(f"{wer=}")