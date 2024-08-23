#!/usr/bin/python
# -*- coding: utf-8 -*-
#
# 工具类：包含了data_collator和call_back

import torch
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Union

from torch.utils.tensorboard import SummaryWriter
from transformers import TrainingArguments, TrainerState, TrainerControl, TrainerCallback
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR

@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lengths and need different padding methods
        # first treat the audio inputs by simply returning torch tensors
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        # get the tokenized label sequences
        label_features = [{"input_ids": feature["labels"]} for feature in features]
        # pad the labels to max length
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        # replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        # if bos token is appended in previous tokenization step,
        # cut bos token here as it's append later anyways
        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels

        return batch

'''
callback有多种状态，在不同的时间调用。查看callback代码
'''
class SavePeftModelCallback(TrainerCallback):
    def on_save(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        checkpoint_folder = os.path.join(args.output_dir, f"{PREFIX_CHECKPOINT_DIR}-{state.global_step}")

        peft_model_path = os.path.join(checkpoint_folder, "adapter_model")
        kwargs["model"].save_pretrained(peft_model_path)

        pytorch_model_path = os.path.join(checkpoint_folder, "pytorch_model.bin")
        if os.path.exists(pytorch_model_path):
            os.remove(pytorch_model_path)
        return control

'''
在每次eval后，输出wer到tensorboard
用于观察loss和wer的关系
'''
class TensorBoardWerCallback(TrainerCallback):
    def __init__(self, tb_writer: SummaryWriter):
        self.tb_writer = tb_writer

    def on_evaluate(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        # 获取wer值
        metrics = kwargs['metrics']
        print(metrics)
        eval_wer = metrics['eval_wer']
        print(eval_wer)
        # 写道tensorboard上
        self.tb_writer.add_scalar('train/wer', eval_wer, state.global_step)
