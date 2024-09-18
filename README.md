Whisper 微调的代码

执行流程：
1. 微调 ft.ft.py
2. peft合并模型 ft.merge_model.py
3. ct2转换 ft.ct2_whisper.py
4. 评估 eval.fasterwhisper_eval.py(语音文件)
   或eval.fasterwhisper_eval_ndarray.py(datasets.features.Audio)

sign.txt说明：
如果有sign.txt存在，说明有微调任务正在执行或有任务执行失败，先看日志。
https://github.com/BruceYu-D20/diting

目录：
api：获取数据的接口
core：微调、合并、ct2的代码
data_process：数据处理脚本
docs：文档
eval：eval过程代码
test：测试代码
tools：工具脚本
util：公共类脚本
diting.py：主入口