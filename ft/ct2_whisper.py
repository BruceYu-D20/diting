import os
from tools.utils import path_with_datesuffix

# 获取所有的数据读写路径
paths = path_with_datesuffix()
MERGE_MODEL_SAVEPATH = paths['MERGE_MODEL_SAVEPATH']
CT2_MERGE_MODEL_SAVEPATH = paths['CT2_MERGE_MODEL_SAVEPATH']

#--low_cpu_mem_usage   Enable the flag low_cpu_mem_usage when loading the model with from_pretrained. (default: False)
# cover的时候指定--quantization float16 , peft加载的时候也要torch.floatx格式
ct_cmd = f"ct2-transformers-converter --model {MERGE_MODEL_SAVEPATH} --output_dir {CT2_MERGE_MODEL_SAVEPATH} --quantization float16 --force --copy_files tokenizer.json preprocessor_config.json"
os.system(ct_cmd)