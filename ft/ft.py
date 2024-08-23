#!/usr/bin/python
# -*- coding: utf-8 -*-

import os
import time
from common import *
from datasets import load_from_disk, DatasetDict,concatenate_datasets
from datasets import Audio
import evaluate
from transformers import BitsAndBytesConfig, WhisperProcessor, WhisperForConditionalGeneration, Seq2SeqTrainer
from peft import prepare_model_for_kbit_training
from peft import LoraConfig, get_peft_model
from features import *
from torch.utils.tensorboard import SummaryWriter

# 创建模型+peft lora
model = WhisperForConditionalGeneration.from_pretrained(MODEL_PATH, quantization_config=BitsAndBytesConfig(load_in_8bit=True))
model.config.forced_decoder_ids = None
model.config.suppress_tokens = []
model = prepare_model_for_kbit_training(model)
# Lora配置
config = LoraConfig(r=32, lora_alpha=64, target_modules=["q_proj", "v_proj"], lora_dropout=0.05, bias="none")
model = get_peft_model(model, config)
model.print_trainable_parameters()

# 分词器等
# feature_extractor = WhisperFeatureExtractor.from_pretrained(MODEL_PATH, language='ar', task="transcribe")
# tokenizer = WhisperTokenizer.from_pretrained(MODEL_PATH, language="ar", task="transcribe")
processor = WhisperProcessor.from_pretrained(MODEL_PATH, language="ar", task="transcribe")
feature_extractor = processor.feature_extractor
tokenizer = processor.tokenizer

def _prepare_dataset(batch):
    # trainer.state.global_step
    # load and resample audio data from 48 to 16kHz
    audio = batch["audio"]
    # compute log-Mel input features from input audio array
    batch["input_features"] = feature_extractor(audio["array"], sampling_rate=audio["sampling_rate"]).input_features[0]
    # encode target text to label ids
    batch["labels"] = tokenizer(batch["sentence"]).input_ids
    return batch

# 数据集
ds = load_from_disk(DATASET_PATH)
common_voice = DatasetDict()
common_voice['train'] = concatenate_datasets([ds['train'], ds['validation']])
common_voice['test'] = ds['test']
common_voice = common_voice.cast_column("audio", Audio(sampling_rate=16000))
common_voice = common_voice.map(_prepare_dataset, remove_columns=common_voice.column_names["train"])

data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)

# metrics 函数
metric = evaluate.load(WER_PATH)

def _compute_metrics(pred):
    pred_ids = pred.predictions
    label_ids = pred.label_ids
    # replace -100 with the pad_token_id
    label_ids[label_ids == -100] = tokenizer.pad_token_id
    # we do not want to group tokens when computing the metrics
    pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = tokenizer.batch_decode(label_ids, skip_special_tokens=True)
    wer = 100 * metric.compute(predictions=pred_str, references=label_str)
    return {"wer": wer}

from transformers import Seq2SeqTrainingArguments

# log日志文件夹按时间存
sd = str(int(time.time()))
training_args = Seq2SeqTrainingArguments(
    output_dir=MODEL_OUT_DIR, # change to a repo name of your choice
    logging_dir=LOGGING_DIR,
    logging_steps=1,
    num_train_epochs=10, # ecpoch 10 batch_size=128 情况下总step 1930
    per_device_train_batch_size=128,
    gradient_accumulation_steps=1, # increase by 2x for every 2x decrease in batch size
    per_device_eval_batch_size=128,
    learning_rate=3e-4, # 经过几次训练发现设置5e-4，loss会趋于收敛，lr会减小到3e-4左右
    warmup_ratio=0.05, # warm up占总step的比例
    # warmup_steps=300,
    # max_steps=1000,
    gradient_checkpointing=True,
    fp16=True,
    eval_strategy="steps",
    save_strategy="steps",
    save_steps=100, # 因为设置了load_best_model_at_end，所以要根据总step来调整，或save_steps=eval_steps=小数。基本设置到1个epoch一次eval
    eval_steps=1,
    predict_with_generate=True,
    generation_max_length=512,
    report_to=["tensorboard"],
    # 下面四个参数配合使用
    # 逻辑：当load_best_model_at_end=True，save_total_limit会存储最优的3个checkpoint，评价标准是metric_for_best_model="wer"，且
    # 根据greater_is_better=False，wer的值越小越好
    # 注意：load_best_model_at_end=True时，eval_strategy和save_strategy必须同时是epoch或steps。且save_strategy必须是eval_strategy的倍数
    #      但我并不希望每个epoch才存一次模型的checkpoint，所以设置为steps。
    #      我希望在loss收敛后可以手动停止训练，所以必须要把save_strategy设置为steps
    #      但eval耗时，如何平衡eval和save mode的平衡？
    load_best_model_at_end=True,
    metric_for_best_model="wer", # metric_for_best_model默认会认为metrics越大越好，wer是越小越好，所以wer要把greater_is_better设置为false
    greater_is_better=False,
    save_total_limit=3,
    label_names=['labels'],
    weight_decay=0.01,
    max_grad_norm=1, # 限制grad_norm的最大值
    remove_unused_columns=False,  # required as the PeftModel forward doesn't have the signature of the wrapped model's forward
)

# 将wer值写入tensorboard，创建writer
tb_writer = SummaryWriter(training_args.logging_dir)

trainer = Seq2SeqTrainer(
    args=training_args,
    model=model,
    train_dataset=common_voice["train"],
    eval_dataset=common_voice["test"],
    data_collator=data_collator,
    compute_metrics=_compute_metrics,
    tokenizer=processor.feature_extractor,
    callbacks=[SavePeftModelCallback, TensorBoardWerCallback(tb_writer)] # callbacks函数允许在一定阶段被回调
)

trainer.train()