#!/usr/bin/python
# -*- coding: utf-8 -*-
import os.path
from datasets import load_from_disk, DatasetDict,concatenate_datasets
from datasets import Audio
import evaluate
from transformers import BitsAndBytesConfig, WhisperProcessor, WhisperForConditionalGeneration, Seq2SeqTrainer
from peft import prepare_model_for_kbit_training
from peft import LoraConfig, get_peft_model
from util.features import *
from torch.utils.tensorboard import SummaryWriter
from util.utils import path_with_datesuffix
from transformers import Seq2SeqTrainingArguments

def create_model(paths):
    # 创建模型+peft lora
    model = WhisperForConditionalGeneration.from_pretrained(
        paths['MODEL_PATH'],
        quantization_config=BitsAndBytesConfig(load_in_8bit=True),
    )
    model.config.forced_decoder_ids = None
    model.config.suppress_tokens = []
    model = prepare_model_for_kbit_training(model)
    # Lora配置
    config = LoraConfig(r=32, lora_alpha=64, target_modules=["q_proj", "v_proj", "fc1", "fc2"], lora_dropout=0.05,
                        bias="none")
    model = get_peft_model(model, config)
    model.print_trainable_parameters()

    # 分词器等
    processor = WhisperProcessor.from_pretrained(paths['MODEL_PATH'], task="transcribe",)
    return model, processor


def _prepare_dataset(batch, processor):
    audio = batch["audio"]
    # compute log-Mel input features from input audio array
    batch["input_features"] = processor.feature_extractor(audio["array"], sampling_rate=audio["sampling_rate"]).input_features[0]
    # encode target text to label ids
    batch["labels"] = processor.tokenizer(batch["sentence"]).input_ids
    return batch

def prepare_data(paths, processor):
    # 数据集
    ds = load_from_disk(paths['DATASET_PATH'])
    common_voice = DatasetDict()
    common_voice['train'] = concatenate_datasets([ds['train'], ds['validation']])
    # common_voice['test'] = ds['test']
    common_voice = common_voice.cast_column("audio", Audio(sampling_rate=16000))
    common_voice = common_voice.map(
        lambda batch: _prepare_dataset(batch, processor),
        remove_columns=common_voice.column_names["train"],
        # num_proc=4,
    )
    return common_voice

def create_metrics_methods(paths):
    # 加载wer metrics
    metric_wer = evaluate.load(os.path.join(paths['METRICS_PATH'], "wer"))
    # 加载cer metrics
    metric_cer = evaluate.load(os.path.join(paths['METRICS_PATH'], "cer"))
    return metric_wer, metric_cer

def _compute_metrics(pred, processor, metric_wer, metric_cer):
    # 预测值和真实值
    pred_ids = pred.predictions
    label_ids = pred.label_ids
    # replace -100 with the pad_token_id
    label_ids[label_ids == -100] = processor.tokenizer.pad_token_id
    # 解码预测值和真实值
    pred_str = processor.tokenizer.batch_decode(pred_ids, skip_special_tokens=True, normalize=True)
    label_str = processor.tokenizer.batch_decode(label_ids, skip_special_tokens=True, normalize=True)
    # wer
    wer = 100 * metric_wer.compute(predictions=pred_str, references=label_str)
    # cer
    cer = 100 * metric_cer.compute(predictions=pred_str, references=label_str)
    return {"wer": wer, "cer": cer}


def create_trainer(model, processor, common_voice, paths, data_collator):
    # log日志文件夹按时间存
    # 定义Seq2Seq训练参数
    training_args = Seq2SeqTrainingArguments(
        output_dir=paths['MODEL_OUT_DIR'],  # 设置模型输出目录，可以根据需要更改
        logging_dir=paths['TENSORBOARD_LOGDIR'],  # 设置日志目录
        logging_steps=1,  # 每一步记录一次日志
        num_train_epochs=5,  # 训练10个epoch
        per_device_train_batch_size=150,  # 每个设备的训练批次大小为128
        gradient_accumulation_steps=1,  # 每次减少2倍的batch size，增加2倍
        # per_device_eval_batch_size=64, # 每个设备的评估批次大小为64
        # eval_accumulation_steps=2, # 每两个step评估一次
        learning_rate=3e-4,  # 学习率为3e-4
        warmup_ratio=0.05,  # warm up占总step的比例
        # eval_delay=1, # 第一轮有warn up，不eval
        # warmup_steps=300, # warm up的step数为300
        # max_steps=1000, # 最大step数为1000
        gradient_checkpointing=True,  # 开启梯度检查点
        fp16=True,  # 开启半精度训练
        eval_strategy="no",  # 每个epoch评估一次
        save_strategy="epoch",  # 每个epoch保存一次
        # save_steps=200, # 每个step保存一次
        # eval_steps=1, # 每个step评估一次
        # batch_eval_metrics=,
        predict_with_generate=True,  # 使用生成进行预测
        generation_max_length=512,  # 生成最大长度为512
        report_to=["tensorboard"],  # 使用tensorboard进行报告
        # 下面四个参数配合使用
        # 逻辑：当load_best_model_at_end=True，save_total_limit会存储最优的3个checkpoint，评价标准是metric_for_best_model="wer"，且
        # 根据greater_is_better=False，wer的值越小越好
        # 注意：load_best_model_at_end=True时，eval_strategy和save_strategy必须同时是epoch或steps。且save_strategy必须是eval_strategy的倍数
        #      但我并不希望每个epoch才存一次模型的checkpoint，所以设置为steps。此时启用max_steps？
        #      我希望在loss收敛后可以手动停止训练，所以必须要把save_strategy设置为steps
        #      但eval耗时，如何平衡eval和save mode的平衡？
        # load_best_model_at_end=True, # 加载最优模型
        # metric_for_best_model="wer", # 评价最优模型的指标为wer
        # greater_is_better=False,  # metric_for_best_model默认会认为metrics越大越好，wer是越小越好，所以wer要把greater_is_better设置为false
        # 下面两个参数不开是因为：认为loss最小，eval结果最优
        # save_total_limit=3, # 保存最优的3个checkpoint
        label_names=['labels'],  # 标签名称
        # weight_decay=0.01, # 权重衰减为0.01
        max_grad_norm=1,  # 限制grad_norm的最大值为1
        remove_unused_columns=False,
        # required as the PeftModel forward doesn't have the signature of the wrapped model's forward
    )

    # 将wer值写入tensorboard，创建writer
    tb_writer = SummaryWriter(training_args.logging_dir)

    trainer = Seq2SeqTrainer(
        args=training_args,
        model=model,
        train_dataset=common_voice["train"],
        # eval_dataset=common_voice["test"],
        data_collator=data_collator,
        # compute_metrics=_compute_metrics,
        tokenizer=processor.feature_extractor,
        callbacks=[SavePeftModelCallback, TensorBoardWerCallback(tb_writer)],  # callbacks函数允许在一定阶段被回调
    )
    return trainer

def main(paths: dict):
    model, processor = create_model(paths)
    train_datasets = prepare_data(paths, processor)
    data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)
    metric_wer, metric_cer = create_metrics_methods(paths)
    trainer = create_trainer(model, processor, train_datasets, paths, data_collator)
    trainer.train()

if __name__ == '__main__':
    paths = path_with_datesuffix()
    main(paths)
