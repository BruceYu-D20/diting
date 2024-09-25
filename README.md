

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

上面两步会封装成脚本，后续开发。

验证功能会开发成脚本，脚本功能是验证是否所有语音文件都转成了wav，16_000，转换的条数是否和原始语音文件个数匹配。待开发

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
| /data/audio/origin/{language} | 存放待处理的原始语音文件 |
| /data/audio/ft_audio/{date}   | 存放微调读取的语音格式   |

eg.

**待处理的文件：**

英语语音文件存放在 /data/audio/origin/en目录下；阿拉伯语存放在/data/audio/origin/ar目录下。

**微调的输入文件：**

2025/01/01日处理的文件存放在/data/audio/ft_audio/20250101目录下

#### 2.3.3 语音文件存放样例

- 每一批语音数据以一个文件夹存放，文件名为日期，格式为yyyyMMdd。
- 语音数据文件夹的同层会有一个或多个csv文件，每个csv代表处理一个或几个语音数据文件夹
- 每个csv的内容包含语音文件的绝对路径和文本内容。
- 脚本可以同时处理多个csv文件

eg.

```
/data/audio/origin
|--  ar
     |--  audio_20240101-20240103.csv
     |--  20240101
          |--  ar_audio_1_1.wav
          |--  ar_audio_1_2.wav
          |--  ......
     |--  20240102
          |--  ar_audio_2_1.wav
          |--  ar_audio_2_2.wav
          |--  ......
     |--  20240108
          |--  ar_audio_3_1.wav
          |--  ar_audio_3_2.wav
          |--  ......
|--  en
     |--  audio_20240101-20240103.csv
     |--  20240102
          |--  ar_audio_3_1.wav
          |--  ar_audio_3_2.wav
          |--  ......
     |--  20240103
          |--  ar_audio_4_1.wav
          |--  ar_audio_4_2.wav
          |--  ......
```

对例子的说明：

- /data/audio/origin/ar下20240101、20240102、20240108分别为3个批次的语音数据文件

- audio_20240101-20240103.csv会根据日期范围，处理20240101和20240102中的语音文件

  #### 2.3.4 数据CSV填写样例

数据excel中有2个字段，分别为：语音文件的路径、语音对应的文本

eg.

/data/audio/origin/en目录下，20240102、20240103目录中存放语音文件，audio_20240101-20240103.csv存放内容如下：

path：语音文件在服务器上的绝对路径

text： 语音对应的文本内容

文件：audio_20240101-20240103.csv

| path                                            | text                                       |
| ----------------------------------------------- | ------------------------------------------ |
| /data/audio/origin/en/20240102/ar_audio_3_1.wav | this is a script process original audio    |
| /data/audio/origin/en/20240102/ar_audio_3_2.wav | jingle bell jingle bell jingle all the way |
| /data/audio/origin/en/20240103/ar_audio_4_1.wav | this is peppa pig                          |
| ...                                             | ...                                        |

**注意：CSV文件不加表头，默认第一列为path，第二列为text**

#### 2.3.5 语音文件存放和数据csv填写规则约束

脚本会对以下项目进行规则验证：

- 数据csv文件必须以 **audio_** 开头。例如audio_20240101_20240103.csv
- 数据csv文件必须包含两个日期，分别代表开始时间和结束时间，格式为yyyyMMdd，两个日期中间用_分隔，开始结束日期可以相同
- 数据csv文件的第一个日期必须小于等于第二个日期。例如audio_20240101_20240101.csv和audio_20240101_20240103.csv都是合法文件名，但audio_20240103_20240101.csv不合法
- 数据csv文件的两个日期区间，必须有可以匹配的日期数据文件夹。例如audio_20240101_20240103.csv可以匹配出20240101、20240102、20240103的语音文件夹
- 数据csv文件的sentence字段，不能为空；不能为多个空格；不能为tab
- 数据csv文件的path字段，path在服务器上必须存在且是一个文件
- 数据csv文件的path字段，path必须在可以匹配的日期数据文件夹内
- 可以匹配的日期数据文件夹内所有语音文件，必须在数据csv文件的path字段中

### 2.4 数据处理

数据处理的目的是将2.2步骤处理的语音文件（wav格式、采样率=16_000），处理成微调时输入需要的datasets.features.Audio格式。

数据处理脚本位置位于：diting/tools/change_audio2array.py

数据处理验证脚本位置位于：diting/tools/eval_audio2array.py

***重要：请查验下面两点事项后，再进行2.4.1的转换操作***

- ***数据已经全部放入/data/audio/{language}下***
- ***数据已经转换成wav格式，采样率=16_000***
- **不转换采样率的数据会影响模型的效果**

#### 2.4.1 配置文件

diting/tools/tool.yaml

```yaml
change_audio2array:
  data_rootpath: D:/data/audio/origin
  en:
    enable: no
    save_path: D:/data/audio/ft_audio/en
  ar:
    enable: no
    save_path: D:/data/audio/ft_audio/ar
  zh:
    enable: yes
    save_path: D:/data/audio/ft_audio/zh
  he:
    enable: no
    save_path: D:/data/audio/ft_audio/he
```

change_audio2array：配置是change_audio2array.py类的配置项

data_rootpath：数据根目录

en ar zh he：语种文件夹，会和data_rootpath拼接，例如  D:/data/audio/origin/en

enable：本次处理中，是否有该语言的数据需要预处理。取值：[no, yes]。

save_path：该语种处理完的文件存储路径

#### 2.4.2 执行数据处理操作

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

4. 检查csv文件

```shell
python tools/datacsv_check.py
```

若有报错，请按照报错内容修改csv填充字段

5. 执行数据处理

```shell
python tools/change_audio2array.py
```

6. 验证

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

 配置文件是diting/core/core.yaml，配置参数意义如下

```yaml
dataset_paths:
  - name: "E:/huggingface/datasets/common_voice_datasets/mgb2_ar"
  - split: ['train']
  - name: "E:/huggingface/datasets/common_voice_datasets/common-17_ar"
  - split: ['train', 'validation']
model_path: "E:/huggingface/models/whisper-large"
metrics_path: "E:/huggingface/metrics"
project_path: "E:/code/diting"
```

dataset_paths: 配置数据源的路径，必须有成对的name和split。一对name、split代表一个数据源的名称和其对应的数据分片。

model_path：openai-whisper的路径

metrics_path：验证函数的路径

project_path：项目路径

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

### 4.4 执行验证模型脚本

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

5. 【按需执行】执行验证基座模型脚本

```shell
python diting.py --run eval --model_id 0
```

**参数解释：**

--run eval：执行验证过程

--model_id：要执行验证的model id或0。查看model_id的值，在CTranslate2输出目录下；0代表基座模型

**说明：**

每一份数据基座模型验证只需要执行1次，不需要多次执行

### 4.5 执行验证基座模型脚本

### 4.5查看日志

在验证流程结束后，日志在 {PROJECT_PATH}/log_dir/{model_id}/eval_log/asr_cp _ {model_id} _ array.txt。

日志中可以查看 **WER带标符、CER带标符、WER去标符、CER去标符** 四个评估值

### 4.6 验证的数据集

获取验证数据的两种方式：

- 将所有的音频数据做分割，将8成分给微调，2成分给验证
- **固定的验证数据集，验证仅做准确率计算，为微调结果提供参考，所以几乎不用变动。且不变动容易和之前的模型做比较。**

### 4.7 配置文件 

- 位置

eval/eval.yaml

- 解释

```yaml
num_process: 4
faster_whisper: /data/models/faster-whisper
array_data_path: /data/..
audio_data_path: /data/
```

num_process：多进程执行验证，进程数

faster_whisper：基座模型的位置，在验证基座模型时生效

array_data_path：验证数据的位置，必须包含Audio.array字段

audio_data_path：验证数据的位置，必须包含path字段

## 5. 项目结构及使用方法

### 5.1 项目目录说明

```
diting/
├── api
│   └── data_process.py  					# 获取数据接口类
├── core				 					# 2. 核心步骤文件夹：微调+合并+转换
│   ├── ct2_whisper.py   					# - 步骤3：Ctranslate2转换模型主类
│   ├── ft.py            					# - 步骤1：微调主类
│   ├── merge_model.py   					# - 步骤2：合并模型参数主类
│   ├── start_tensorboard.sh
│   └── tensorboard_ui.service
├── data_process
│   ├── load_data.py     					# 加载数据，将语音文件变成datasets.features.Audio的数据处理方法
├── diting.py			 					# 1. 项目主入口
├── eval				 					# 3. 验证代码文件夹
│   ├── eval.yaml		 					# eval配置文件
│   ├── fasterwhisper_checkpoint_eval_audio.py 
│   ├── fasterwhisper_checkpoint_eval.py	# 验证微调后的模型
│   ├── fasterwhisper_eval_audio.py
│   └── fasterwhisper_eval.py				# 验证基座模型
├── README.md								# 6. 帮助文档
├── tools									# 4. 工具类
│   ├── change_audio2array.py				# 将语音文件转成微调需要的数据类型
│   ├── change_localpath2server.py		
│   ├── datacsv_check.py					# 校验方法：检查数据准备的csv文件是否符合规范
│   └── tool.yaml							# 工具类的配置文件
└── util									# 5. 公共类
    ├── common.py							# 微调代码的配置文件
    ├── features.py
    ├── logger.py
    └── utils.py
```

### 5.2 主要类的使用

#### 5.2.1 diting.py

- 作用：主类，入口类
- 参数

--run：指定运行模式，必须传入，取值sign ft eval。

​	sign指只生成sign.txt文件，用于手动执行微调中每个小步骤

​	ft指执行整个微调流程，包括参数微调、模型参数合并、CTranslate2转换

​	eval指执行验证过程，调用。必须和--model_id一同指定

--model_id：指定在--run eval 的验证过程中，要验证的model_id；当model_id=0时，验证基座模型

--data_type：--run eval时，可选指定。取值[array, audio]。array代表处理Audio，audio代表处理音频数据。默认是array

- 样例

```shell
# 单步执行时生成sign
python diting.py --run sign
# 整体执行微调-合并模型-ctranslate模型
python diting.py --run ft
# 验证微调模型，数据源是audio.array
python diting.py --run eval --model_id 20240918.29775
# 验证微调模型，数据源是path
python diting.py --run eval --model_id 20240918.29775 --data_type audio
# 验证基座模型
python diting.py --run eval --model_id 0
```

#### 5.2.2 core/ft.py

- 作用：希望单步执行。微调参数主类，微调openai-whisper-large-v3的参数，并输出微调后的模型参数
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

- 作用：希望单步执行。用于合并微调参数和openai-whisper-large-v3的参数，并输出合并之后的模型
- 说明

执行之前，先保证sign.txt存在，且sign.txt中记录model_id的5.2.2步骤结果已经存在

- 执行方法

```shell
python core/merge_model.py
```

- 存储目录

diting/merged_model/{model_id}

#### 5.2.4 core/ct2_whisper.py

- 作用：希望单步执行。将合并参数后的模型，做CTranslate2的转换，输出转换后的模型
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

### 5.3 工具类的使用

#### 5.3.1 tools/datacsv_check.py

- 作用：校验加载数据配置的csv文件的合法性
- 参数：

参数从tools/tool.yaml中取，详细查看2.4.1章节

- 样例

```shell
python tools/datacsv_check.py
```

#### 5.3.2 tools/chage_audio2array.py

- 作用：将现场提供的音频文件变成datasets.features.Audio格式
- 参数：

参数从tools/tool.yaml中取，详细查看2.4.1章节

- 样例

```
python tools/change_audio2array.py
```

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

m_165、m_330、m_495、m_660、m_825是5轮微调后的结果模型

错误率，模型效果越好

|           | 基座   | m_165  | m_330  | m_495  | m_660  | m_825  |
| --------- | ------ | ------ | ------ | ------ | ------ | ------ |
| WER       | 0.2667 | 0.2088 | 0.2011 | 0.2026 | 0.2059 | 0.2087 |
| CER       | 0.0751 | 0.0601 | 0.0578 | 0.0573 | 0.0585 | 0.0596 |
| WER带标符 | 0.4227 | 0.3691 | 0.3498 | 0.3572 | 0.3742 | 0.3758 |
| CER带标符 | 0.1654 | 0.1554 | 0.1440 | 0.1443 | 0.1579 | 0.1576 |

### 6.3 每次启动微调的数据量建议

- 总语音时长：建议每次微调增加100小时以上的语音训练数据
- 单条语音时长：建议每个语音文件的总时长在10s-20s左右，且是有效时长。太短的片段无法提供足够的信息，太长的片段会增加训练复杂度。合理的时长可以确保模型在处理长段落时的稳定性，同时避免内存和计算资源的过多消耗。
- 语音完整性：保证每个语音文件中的语音是完整的句子，最大程度保证语义完整
- 数据多样性：尽量覆盖 **不同的语言场景的语音**，**不同性别、不同年龄的人的语音**。数据量越多、场景越多的数据，越有利于模型的泛化能力。

## 7. 版本日志

V1.0：

基础的微调、合并、转换功能；

基础的验证脚本

基础的工具脚本

V1.1：

要添加微调代码支持多个路径的配置