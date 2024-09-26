#!/usr/bin/python
# -*- coding: utf-8 -*-
import os.path
from pathlib import Path
from util.utils import *
import glob
import pandas as pd

'''
本脚本用于校验csv文件，规则如下：
- 数据csv文件必须以 **audio_** 开头。例如audio_20240101_20240103.csv
- 数据csv文件必须包含两个日期，分别代表开始时间和结束时间，格式为yyyyMMdd，两个日期中间用_分隔
- 数据csv文件的第一个日期必须小于等于第二个日期。例如audio_20240101_20240101.csv和audio_20240101_20240103.csv都是合法文件名
- 数据csv文件的两个日期区间，必须有可以匹配的日期数据文件夹。例如audio_20240101_20240103.csv可以匹配出20240101、20240102、20240103的语音文件夹
- 数据csv文件的path字段，path必须存在且是一个文件
- 数据csv文件的path字段，path必须在可以匹配的日期数据文件夹内
- 可以匹配的日期数据文件夹内所有语音文件，必须在数据csv文件的path字段中
'''

# loaddata csv文件名校验
def check_audiocsv(config):
    # 正则匹配日期规则
    date_pattern = re.compile(r'\d{8}')  # yyyyMMdd 日期格式

    data_rootpath = config['change_audio2array']['data_rootpath']
    for locale, locale_config in config['change_audio2array'].items():
        # 保证locale是支持的语言
        if locale not in SUPPORT_LAUNGUAGES:
            continue
        if isinstance(locale_config, dict) and locale_config.get('enable'):
            # 用于存放每个csv上的开始时间和结束时间
            all_dates = []
            # 文件夹下的路径
            data_path = os.path.join(data_rootpath, locale)
            # 获取data_path下的所有.csv文件
            csv_files = glob.glob(os.path.join(data_path, "*.csv"))

            #验证csv文件名和内容合法性
            for csv_file in csv_files:
                csv_file = Path(csv_file).as_posix()
                filename = os.path.basename(csv_file)
                '''
                对所有的csv文件进行校验
                1. 检查文件名是否以'audio_'开头
                2. 检查是否包含两个 yyyyMMdd 格式的日期
                3. 检查第一个日期是否小于第二个日期
                4. csv指定的两个日期，必须有能匹配的文件夹
                '''
                # 1. 检查文件名是否以'audio_'开头
                name_without_extension = filename.rstrip(".csv")
                print(f"csv文件名称：{filename}")
                if not filename.startswith("audio_"):
                    raise Exception(f"csv文件名必须以'audio_'开头: {filename}")

                # 2. 检查是否包含两个 yyyyMMdd 格式的日期
                dates = date_pattern.findall(name_without_extension)
                if len(dates) != 2:
                    raise ValueError(f"文件名 '{filename}' 没有包含正确的两个日期（yyyyMMdd格式）。")

                # 3. 检查第一个日期是否小于第二个日期
                date1_str, date2_str = dates
                try:
                    date1 = datetime.strptime(date1_str, '%Y%m%d')
                except:
                    raise ValueError(f"文件名 '{filename}' 中第一个日期格式不正确。正确格式yyyyMMDD")
                try:
                    date2 = datetime.strptime(date2_str, '%Y%m%d')
                except:
                    raise ValueError(f"文件名 '{filename}' 中第二个日期格式不正确。正确格式yyyyMMDD")
                if date1 > date2:
                    raise ValueError(f"文件名 '{filename}' 中第一个日期大于第二个日期。")

                # 保存每个文件的日期范围及其对应文件名
                all_dates.append((date1, date2, filename))

                # 4. 检查所有文件日期之间是否有交集
                for i in range(len(all_dates)):
                    date1_start, date1_end, file1 = all_dates[i]
                    for j in range(i + 1, len(all_dates)):
                        date2_start, date2_end, file2 = all_dates[j]

                        # 检查日期范围是否有交集
                        if (date1_start <= date2_end) and (date2_start <= date1_end):
                            raise ValueError(f"文件 '{file1}' 和文件 '{file2}' 的日期范围有交集。")

                '''
                开始检查文件的内容合法性
                '''
                # 查找当前文件夹下，所有的文件夹
                folders = [f for f in os.listdir(data_path) if os.path.isdir(os.path.join(data_path, f))]
                # 所有在csv date1到date2 的日期文件夹
                folder_in_date1_date2 = []
                for folder in folders:
                    try:
                        folder_date = datetime.strptime(folder, '%Y%m%d')
                    except ValueError:
                        print(f"文件夹名 '{folder}' 不是有效的日期格式。")
                        continue
                    if folder_date >= date1 and folder_date <= date2:
                        folder_in_date1_date2.append(folder)

                if not folder_in_date1_date2:
                    raise ValueError(f"csv文件 {csv_file} 中日期范围没有对应的文件夹。")
                match_folder_str = ",".join(folder_in_date1_date2)
                print(f'匹配的语音数据文件夹 {csv_file}： {match_folder_str}')

                # 读取有效日期范围的文件夹目录下所有的文件路径
                valid_paths = []
                for folder in folder_in_date1_date2:
                    folder_path = os.path.join(data_path, folder)
                    for root, dirs, files in os.walk(folder_path):
                        for file in files:
                            file_path = os.path.join(root, file)
                            # 加Path为了容错，让路径是linux路径的格式
                            valid_paths.append(Path(file_path).as_posix())

                '''
                验证csv内容的合法性
                1. sentence不能为空
                2. audio_path必须存在
                3. audio_path必须是文件
                4. csv中第一列的文件路径，必须在日期范围文件夹内
                5. 范围内文件夹的所有文件必须在csv中第一列的文件路径中
                '''
                audio_df = pd.read_csv(csv_file, header=None).fillna('')
                for _, row in audio_df.iterrows():
                    audio_path = row[0]
                    sentence = row[1]
                    # 5. 检查csv中文件是否存在
                    if not os.path.exists(audio_path):
                        raise FileNotFoundError(f"csv文件 '{csv_file}',{audio_path} 路径不存在")
                    # 6. sentence不能为空
                    if sentence == '' or sentence == None or sentence == 'nan' or sentence.strip() == '':
                        raise ValueError(f"csv文件 '{csv_file}': {audio_path}中sentence不能为空")
                    # 7. audio_path必须是文件
                    if not os.path.isfile(audio_path):
                        raise ValueError(f"csv文件 '{csv_file}': {audio_path}不是文件")
                    # 8. csv中的文件路径，必须在日期范围文件夹内
                    if not audio_path in valid_paths:
                        raise ValueError(f"csv文件 '{csv_file}': {audio_path}不在匹配的文件夹中")
                csv_paths = set(audio_df[0].tolist())
                for valid_path in valid_paths:
                    if valid_path not in csv_paths:
                        raise ValueError(f"csv文件 '{csv_file}': {valid_path}不在csv中或查看是否是匹配的语音数据文件夹中的语音")

def main(check_yaml=True, config=None):
    # 获取配置文件路径
    current_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(current_dir, 'tool.yaml')
    # 读取配置文件
    if config == None:
        config = read_yaml(config_path)
    # 验证配置文件合法性
    if check_yaml:
        valid_tool_config(config_path)
    # 检验csv文件名的合法性
    check_audiocsv(config)
    print(f'语音数据csv文件校验合法')

if __name__ == '__main__':
    main()
