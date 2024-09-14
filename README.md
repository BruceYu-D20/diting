Whisper 微调的代码

执行流程：
1. 微调 ft.ft.py
2. peft合并模型 ft.merge_model.py
3. ct2转换 ft.ct2_whisper.py
4. 评估 eval.whisper_eval.py

sign.txt说明：
如果有sign.txt存在，说明有微调任务正在执行或有任务执行失败，先看日志。
https://github.com/BruceYu-D20/diting