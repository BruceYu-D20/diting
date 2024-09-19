

## 1. 说明

本文档旨在说明微调流程、流程规范和操作命令。

微调流程分为三个阶段：数据准备阶段、微调阶段、验证阶段。

文档将对三个阶段的规范、操作方法、命令逐一解释。

## 2. 数据准备

### 2.1 数据格式要求

微调中，要求数据是wav格式，且数据采样率是16_000。

尽量保证文件是单声道，之后在媒体中心里面也是会区分声道来翻译。

### 2.2 数据格式转换及采样率转换

pcm格式转wav：ffmpeg -ar 原始采样率 -f s16le -i 原始文件 -ar 16000 -y 输出文件

其他格式转wav：ffmpeg -i 原始文件 -ar 16000 -y 输出文件

验证脚本待补充，脚本功能是验证是否所有语音文件都转成了wav，16_000，转换的条数是否和原始语音文件个数匹配

### 2.3 数据存放

数据以 **语种** 作为区别，存放在不同的语种文件夹下。

在存放时，需确认音频文件已经做完格式和采样率的转换。

#### 2.3.1 语种标志和语种对应关系

| 语种标志 | 语种     |
| -------- | -------- |
| ar       | 阿拉伯语 |
| he       | 希伯来语 |
| en       | 英语     |
| zh       | 中文     |

#### 2.3.2 存放目录说明

| 目录                          | 作用                     |
| ----------------------------- | ------------------------ |
| /data/audio/origin/{languege} | 存放待处理的原始语音文件 |
| /data/audio/ft_audio/{date}   | 存放微调读取的语音格式   |

eg.

**待处理的文件：**

英语语音文件存放在 /data/audio/origin/en目录下；阿拉伯语存放在/data/audio/origin/ar目录下。

**微调的输入文件：**

2025/01/01日处理的文件存放在/data/audio/ft_audio/20250101目录下

#### 2.3.3 语音文件存放规范

- 每一批数据以一个文件夹存放。存放完成后，放入一个和文件夹同名的**数据excel**文件，excel和数据文件夹在同一文件目录层级下。
- 脚本根据excel的名称去处理同名的数据文件夹，数据处理完后不可变，所以在执行脚本之前，请确认所有数据已经全部放入。
- 脚本可以同时处理多个excel

eg.

```
/data/audio/origin
|--  ar
     |--  batch_1.xlxs
     |--  batch_1
          |--  ar_audio_1_1.wav
          |--  ar_audio_1_2.wav
          |--  ......
     |--  batch_2.xlxs
     |--  batch_2
          |--  ar_audio_2_1.wav
          |--  ar_audio_2_2.wav
          |--  ......
|--  en
     |--  batch_3.xlxs
     |--  batch_3
          |--  ar_audio_3_1.wav
          |--  ar_audio_3_2.wav
          |--  ......
     |--  batch_4.xlxs
     |--  batch_4
          |--  ar_audio_4_1.wav
          |--  ar_audio_4_2.wav
          |--  ......
```

  #### 2.3.4 数据excel填写规范

数据excel中有2个字段，分别为：语音文件的路径、语音对应的文本

eg.

/data/audio/origin/en目录下，batch_3目录中存放语音文件，batch_3.xlxs存放内容如下：

path：语音文件在服务器上的绝对路径

text： 语音对应的文本内容

文件：batch_3.xlxs

| path                                        | text                                       |
| ------------------------------------------- | ------------------------------------------ |
| /data/audio/origin/en/batch_3/audio_3_1.wav | this is a script process original audio    |
| /data/audio/origin/en/batch_3/audio_3_2.wav | jingle bell jingle bell jingle all the way |
| /data/audio/origin/en/batch_3/audio_3_3.wav | this is peppa pig                          |

### 2.4 数据处理

数据处理的目的是将2.2步骤处理的语音文件（wav格式、采样率=16_000），处理成微调时输入需要的datasets.features.Audio格式。

数据处理脚本位置位于：diting/tools/change_audio2array.py

数据处理验证脚本位置位于：diting/tools/eval_audio2array.py

***重要：请查验下面两点事项后，再进行2.4.1的转换操作***

- ***数据已经全部放入/data/audio/{languege}下***
- ***数据已经转换成wav格式，采样率=16_000***

#### 2.4.1 执行数据处理操作

1. 创建docker容器（不用重复创建）

```shell
docker run -itd --gpus all -v /data:/data --name trans_audio --ipc=host ft_env
```

说明：

**-v**：指定挂载到容器的磁盘，-v /data:/data代表将本机的/data挂载到容器的/data目录

2. 进入容器

```shell
docker exec -it ft_env /bin/bash
```

3. 选择执行环境

```shell
conda activate ftenv
cd /data/diting
export PYTHONPATH=/data/diting
```

4. 执行数据处理

```shell
python tools/change_audio2array.py
```

5. 验证

```
python tools/eval_audio2array.py
```

查看日志输出条数是否和原始文件数据条数相同

## 3. 微调

### 3.1 微调流程及对应功能脚本

1. 微调 core.ft.py
2. 模型参数合并 core.merge_model.py
3. ct2转换 core.ct2_whisper.py

***重要：4个步骤从前到后依赖，不可跳步骤执行***

### 3.2 执行方法

1. 创建docker容器（不可重复创建）

```shell
docker run -itd --gpus all -v /data:/data --name trans_audio --ipc=host ft_env
```

2. 进入容器

```shell
docker exec -it ft_env /bin/bash
```

3. 选择执行环境

```shell
conda activate ftenv
cd /data/diting
export PYTHONPATH=/data/diting
```

4. 执行微调过程

```shell
python diting.py --run ft
```

参数解释：

--run ft：执行微调过程

**说明：步骤4.执行微调过程的分解执行步骤如下。此处的3步，在前序步骤执行完成的情况下，希望单独执行后续步骤，可通过以下命令实现**

1. 创建sign.txt文件

```shell
python diting.py --run sign
```

参数解释：

--run sign：仅生成sign.txt文件

2. 执行微调步骤

```shell
python core/ft.py
```

3. 执行模型参数合并步骤

```
python core/merge_model.py
```

4. 执行CTranslate2转换模型

```shell
python core/ct2_whisper.py
```

### 3.3 配置文件

 配置文件是diting/util/common.py，配置参数意义如下

| 名称         | 含义                  | 示例                                            |
| ------------ | --------------------- | ----------------------------------------------- |
| DATASET_PATH | 数据位置              | /data/huggingface/common_voice_17/ar            |
| MODEL_PATH   | 模型位置              | /data/huggingface/model/openai-whisper-large-v3 |
| METRICS_PATH | metrics函数文件夹位置 | /data/huggingface/metrics                       |
| PROJECT_PATH | 项目位置              | /data/diting                                    |

### 3.4 注意事项

为保证微调过程同时只有一个程序在执行，diting.py在执行后，会在diting目录下创建sign.txt文件，微调流程结束后，sign.txt文件会被脚本删除。

sign.txt内容为当前程序编号，形如20240902.2790。

如果此文件存在，微调过程的三个脚本，会分别从配置的 模型存储目录、参数合并存储目录、ct2转换模型存储目录下的20240902.2790读取模型信息。

正常情况下，如果在流程执行完成后，sign.txt文件存在，则流程存在异常，请查看日志。

## 4. 验证

验证阶段即微调模型的效果验证，会对结果模型进行WER和CER的计算。

### 4.1 WER和CER

WER（Word Error Rate）和CER（Character Error Rate）通常用于评估ASR（自动语音识别）的识别准确率。

两者区别在于WER基于word（单词）计算；CER基于Character （字符）计算

- WER计算方法

WER=（ASR多识别单词数+ASR漏识别单词数+ASR识别的单词和参考文本不同的单词数）/参考文本总单词数

- CER计算方法

CER=（ASR多识别字符数+ASR漏识别字符数+ASR识别的字符和参考文本不同的字符数）/参考文本总字符数

### 4.2 阿拉伯语的标符

在阿拉伯语中，省略标符可影响单词意义，但基本不对母语者有影响。但在ASR中，对CER和WER影响明显。

eg.

单词 "كتب" 可以有多种含义：

- **كَتَبَ (kataba)**：写了
- **كُتُب (kutub)**：书籍
- **كِتَاب (kitāb)**：一本书

### 4.3 验证指标

diting在验证脚本中，输出4项指标，分别为 **去标符的WER，去标符的CER，带标符的WER，带标符的CER**

### 4.4 执行验证脚本

1. 创建docker容器（不用重复创建）

```shell
docker run -itd --gpus all -v /data:/data --name trans_audio --ipc=host ft_env
```

2. 进入容器

```
docker exec -it ft_env /bin/bash
```

3. 选择执行环境

```shell
conda activate ftenv
cd /data/diting
export PYTHONPATH=/data/diting
```

4. 执行验证脚本

```shell
python diting.py --run eval --model_id {model_id}
```

参数解释：

--run eval：执行验证过程

--model_id：要执行验证的model id。查看model_id的值，在CTranslate2输出目录下。

### 4.5查看日志

在验证流程结束后，日志在 {PROJECT_PATH}/log_dir/eval/asr_{model_id}_{checkpoint_dir}.txt。

日志中可以查看 **WER带标符、CER带标符、WER去标符、CER去标符** 四个评估值

### 4.6 验证的数据集

获取验证数据的两种方式：

- 将所有的音频数据做分割，将8成分给微调，2成分给验证
- **固定的验证数据集，验证仅做准确率计算，为微调结果提供参考，所以几乎不用变动。且不变动容易和之前的模型做比较。**

## 5. 项目结构及使用方法

### 5.1 项目目录说明

项目名：diting

接口目录：diting/api

**核心脚本：diting/core**

数据预处理：diting/data_process

文档：diting/docs

测试代码：diting/test

**工具类：tools**

utils：核心代码的公共类

日志：diting/log_dir

微调输出模型参数：diting/model_out

合并后的模型存储：diting/merged_model

ctranslate后的模型：diting/ct2_model

### 5.2 主要类的使用

#### 5.2.1 diting.py

- 作用：主类，入口类
- 参数

--run：指定运行模式，必须传入，取值sign ft eval。

​	sign指只生成sign.txt文件，用于手动执行微调中每个小步骤

​	ft指执行整个微调流程，包括参数微调、模型参数合并、CTranslate2转换

​	eval指执行验证过程，调用。必须和--model_id一同指定

--model_id：指定在--run eval 的验证过程中，要验证的model_id

- 样例

```shell
python diting.py --run sign
python diting.py --run ft
python diting.py --run eval --model_id 20240918.29775
```

#### 5.2.2 core/ft.py

- 作用：微调参数主类，微调openai-whisper-large-v3的参数，并输出微调后的模型参数
- 说明

执行之前，先保证项目根目录下sign.txt已经存在，文件中记录的model_id是当此微调参数的任务编号。如果文件不存在，使用

`python diting.py --run sign`创建文件。

- 执行方法

```shell
python core/ft.py 
```

- 存储目录

diting/model_out/{model_id}

#### 5.2.3 core/merge_model.py

- 作用：用于合并微调参数和openai-whisper-large-v3的参数，并输出合并之后的模型
- 说明

执行之前，先保证sign.txt存在，且sign.txt中记录model_id的5.2.2步骤结果已经存在

- 执行方法

```shell
python core/merge_model.py
```

- 存储目录

diting/merged_model/{model_id}

#### 5.2.4 core/ct2_whisper.py

- 作用：将合并参数后的模型，做CTranslate2的转换，输出转换后的模型
- 说明：

执行之前，先保证sign.txt存在，且sign.txt中记录的model_id的5.2.3步骤结果已经存在

- CTranslate2转换模型的作用

CTranslate2 是一个高效的推理引擎，专门用于加速和优化机器翻译、文本生成等任务中的模型推理。主要作用是：

推理优化：减少计算开销

内存优化：转换后的模型通常占用更少的内存

- 执行方法

```shell
python core/ct2_whisper.py
```

- 存储目录

diting/ct2_model/{model_id}

微调数据

微调轮数：5

总微调时长：12h11m42s

合并参数时长

## 6. 测试结果

### 6.1运行时长 

GPU：NVIDIA A40 46G x1

微调数据条数：24654

验证数据条数：10480

微调轮数：5

合并模型个数：5

转换模型个数：5

验证模型个数：5

| 执行过程               | 耗时                      |
| ---------------------- | ------------------------- |
| 微调时长               | 12.195h（平均每轮2.439h） |
| 合并模型时长           | 4分21秒                   |
| 转换模型时长           | 2分01秒                   |
| 验证过程时长（单进程） | 5.45h                     |
| 验证过程时长（2进程）  | 4.05h                     |
| 验证过程时长（3进程）  | 3.48h                     |
| 验证过程时长（4进程）  | 3.28h                     |
| 验证过程时长（5进程）  | 3.01h                     |

### 6.2 错误率

基座模型为 Systran/faster-whisper-large-v3

m_165、m_330、m_495、m_460、m_825是5轮微调后的结果模型

|           | 基座   | m_165 | m_330 | m_495 | m_460 | m_825 |
| --------- | ------ | ----- | ----- | ----- | ----- | ----- |
| WER       | 0.2667 | 0.208 | 0.201 | 0.202 | 0.205 | 0.375 |
| CER       | 0.0751 | 0.060 | 0.057 | 0.057 | 0.058 | 0.157 |
| WER带标符 | 0.4227 | 0.369 | 0.349 | 0.357 | 0.374 | 0.375 |
| CER带标符 | 0.1654 | 0.155 | 0.144 | 0.144 | 0.157 | 0.157 |

