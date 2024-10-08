#!/usr/bin/python
# -*- coding: utf-8 -*-
from util.utils import *
from tqdm import tqdm

# 获取所有的数据读写路径
def ctranslate_whisper(paths):
    MERGE_MODEL_SAVEPATH = paths['MERGE_MODEL_SAVEPATH']
    CT2_MERGE_MODEL_SAVEPATH = paths['CT2_MERGE_MODEL_SAVEPATH']

    for step, checkpoint_dir in enumerate(tqdm(os.listdir(MERGE_MODEL_SAVEPATH))):
        merge_model_savepath = os.path.join(MERGE_MODEL_SAVEPATH, checkpoint_dir)
        ct2_merge_model_savepath = os.path.join(CT2_MERGE_MODEL_SAVEPATH, checkpoint_dir)
        print(f'CTRANSFORM2 {step}: FROM | {merge_model_savepath}| ---> |{ct2_merge_model_savepath}')
        #--low_cpu_mem_usage   Enable the flag low_cpu_mem_usage when loading the model with from_pretrained. (default: False)
        # cover的时候指定--quantization float16 , peft加载的时候也要torch.floatx格式
        ct_cmd = f"ct2-transformers-converter --model {merge_model_savepath} --output_dir {ct2_merge_model_savepath} --quantization float16 --force --copy_files tokenizer.json preprocessor_config.json"
        os.system(ct_cmd)


def main(paths: dict):
    ctranslate_whisper(paths)

if __name__ == '__main__':
    paths = path_with_datesuffix()
    main(paths)