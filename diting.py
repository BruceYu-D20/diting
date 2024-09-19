import core.ft
import core.merge_model
import core.ct2_whisper
import eval.fasterwhisper_checkpoint_eval
from util.utils import *
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='接受--run和--model_id参数')
    parser.add_argument(
        '--run',
        choices=['ft', 'sign', 'eval'],
        required=True,
        help='参数值只能为ft或sign。ft表示执行整个微调步骤；sign表示只创建sign.txt；eval标识执行验证过程。'
             '当run的值是ft或sign时，不允许传入--model_id参数；当run的值是eval时，必须传入--model_id参数。')
    parser.add_argument('--model_id', help="指定ctranslate后模型的id，依赖--run eval参数")
    args = parser.parse_args()

    # 如果传入--run eval 但没传入model_id，不执行
    if args.run == 'eval' and not args.model_id:
        raise ValueError("--run eval 需要传入--model_id参数")
    # 如果传入--run是ft或sign，同时传入了model_id，报错。传入--run in [ft,sign]时，不允许传入--model_id
    if args.run in ['ft', 'sign'] and args.model_id:
        raise ValueError("--run in [ft,sign]时，不允许传入--model_id参数")

    run_arg = args.run
    if args.run in ['ft', 'sign']:
        return (args.run, None)
    elif args.run == 'eval':
        return (args.run, args.model_id)
    else:
        raise ValueError("--run 的值必须是ft/sign/eval，--model_id依赖--run eval")
    print(f'传入参数是： {run_arg}')

def main(run_arg, model_id):
    if run_arg == 'ft':
        # 创建训练的标志，当sign.txt存在时，说明当前有训练在执行
        create_sign_begin()
        # 获取所有文件目录
        paths = path_with_datesuffix()
        # 训练
        core.ft.main(paths)
        # 合并
        core.merge_model.main(paths)
        # CTranslate2转换模型
        core.ct2_whisper.main(paths)
    elif run_arg == 'sign':
        # 创建训练的标志，当sign.txt存在时，说明当前有训练在执行
        create_sign_begin()
    elif run_arg == 'eval' and model_id is not None:
        eval.fasterwhisper_checkpoint_eval(model_id)

    else:
        raise ValueError("传值错误，查看python diting.py --help")

if __name__ == '__main__':
    run_arg, model_id = parse_args()
    main(run_arg, model_id)