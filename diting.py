import core.ft
import core.merge_model
import core.ct2_whisper
from tools.utils import *

if __name__ == '__main__':

    try:
        # 创建训练的标志，当sign.txt存在时，说明当前有训练在执行
        create_sign_begin()
        # 训练
        core.ft.main()
        # 合并
        core.merge_model.main()
        # CTranslate2转换模型
        core.ct2_whisper.main()
    except Exception as e:
        print(e)
    finally:
        # 删除训练的标志
        del_sign_last()