import os
import dl_translate as dlt
import pandas as pd
import json
import re
from typing import List

# 如果需要 dl-translate 作为备用，可以保留此初始化
mt = dlt.TranslationModel()

# 以下部分为你现有的 Gemini 配置与函数
try:
    import google.generativeai as genai
    from dotenv import load_dotenv
    
    # 加载环境变量
    load_dotenv()
    
    # 从环境变量获取API密钥
    api_key = os.getenv("GOOGLE_API_KEY")
    if api_key:
        genai.configure(api_key=api_key)
        GEMINI_AVAILABLE = True
    else:
        print("警告: 未找到GOOGLE_API_KEY环境变量，请确保已设置正确的API密钥")
        GEMINI_AVAILABLE = False
except ImportError:
    print("警告: 未安装google-generativeai或dotenv包，请使用 pip install google-generativeai python-dotenv 安装")
    GEMINI_AVAILABLE = False
except Exception as e:
    print(f"警告: 配置Gemini API时出错: {e}")
    GEMINI_AVAILABLE = False


def gemini_translate(text_batch: List[str], max_retries: int = 3) -> List[str]:
    """
    使用Google Gemini API翻译一批医疗短语，使用特殊标记确保准确分割
    
    Args:
        text_batch: 需要翻译的中文短语列表
        max_retries: 最大重试次数
    
    Returns:
        翻译后的英文短语列表
    """
    if not text_batch:
        return []
    
    if not GEMINI_AVAILABLE:
        print("错误: Gemini API不可用，无法进行翻译，已返回原文")
        return text_batch
    
    START_TAG = "<<PHRASE_"
    END_TAG = ">>"
    
    # 将每个短语用独特标记包裹
    tagged_texts = [
        f"{START_TAG}{i}{END_TAG}{phrase}{START_TAG}/{i}{END_TAG}"
        for i, phrase in enumerate(text_batch)
    ]
    combined_text = " ".join(tagged_texts)
    batch_size = len(text_batch)
    
    prompt = f"""以下是{batch_size}个中文医疗短语，每个短语包含在特殊XML样式标记中。
每个短语的格式为：<<PHRASE_数字>>短语<<PHRASE_/数字>>，其中数字是短语的索引。

请将每个中文短语翻译成英文，保持原始的标记格式不变。
请按照原始顺序返回结果，确保每个翻译后的短语都包含在相同的标记之间。
不要翻译标记本身，它们仅用于标识不同的短语。
确保返回所有{batch_size}个短语的翻译。
保留英文、数字和特殊符号不变，只翻译中文部分。

例如输入：
<<PHRASE_0>>感冒<<PHRASE_/0>> <<PHRASE_1>>高血压(原发性)<<PHRASE_/1>> <<PHRASE_2>>头痛，严重<<PHRASE_/2>>

应该返回：
<<PHRASE_0>>common cold<<PHRASE_/0>> <<PHRASE_1>>hypertension (primary)<<PHRASE_/1>> <<PHRASE_2>>headache, severe<<PHRASE_/2>>

以下是需要翻译的短语：
{combined_text}
"""
    
    # 创建 Gemini 模型
    try:
        model = genai.GenerativeModel("gemini-2.0-flash")
    except Exception as e:
        print(f"创建Gemini模型实例失败: {e}")
        return text_batch
    
    response_text = ""
    for attempt in range(max_retries):
        try:
            # 根据实际SDK确认 generate_text() 的返回结构；此处假设直接返回翻译后的字符串
            response = model.generate_content(prompt)
            response_text = response.text
            break
        except Exception as e:
            if attempt < max_retries - 1:
                print(f"调用Gemini翻译失败，正在重试... (第{attempt+1}次)")
            else:
                print(f"翻译请求多次失败，返回原文: {e}")
                return text_batch
    
    # 利用正则匹配提取翻译结果
    translated_phrases = []
    for i in range(batch_size):
        pattern = re.compile(
            re.escape(f"{START_TAG}{i}{END_TAG}")
            + r"(.*?)"
            + re.escape(f"{START_TAG}/{i}{END_TAG}"),
            re.DOTALL
        )
        match = pattern.search(response_text)
        if match:
            translated_phrases.append(match.group(1).strip())
        else:
            translated_phrases.append(text_batch[i])
    
    return translated_phrases


def translate_colname(file_path):
    """
    使用批量翻译的方式，翻译 CSV 文件中的所有列名。
    """
    # 读取CSV文件
    df = pd.read_csv(file_path)
    colnames = df.columns.tolist()

    #分成100个一组
    colnames_groups = [colnames[i:i+100] for i in range(0, len(colnames), 100)] 
    translated_colnames_groups = []
    for colnames_group in colnames_groups:
        # 一次性批量翻译所有列名
    # 一次性批量翻译所有列名
        translated_colnames_sub = gemini_translate(colnames_group)
        translated_colnames_groups.append(translated_colnames_sub)

    # 合并所有翻译结果
    translated_colnames = [item for sublist in translated_colnames_groups for item in sublist]


    # 逐个显示进度及结果

    for i, (original, translated) in enumerate(zip(colnames, translated_colnames), start=1):
        print(f"进度：已翻译 {i}/{len(colnames)} 列 -> 原列名: {original} | 英文: {translated}")

    # 构建「原列名 -> 英文翻译」的映射
    colname_mapping = dict(zip(colnames, translated_colnames))
    
    # 保存映射结果到 JSON 文件
    with open('../../data/hospital_A/tmp_preprocessed_data/colname_translation.json', 'w', encoding='utf-8') as f:
        json.dump(colname_mapping, f, ensure_ascii=False, indent=4)


def translate_colname(files_path):
    """
    遍历指定目录下的所有 CSV 文件，并根据翻译映射替换列名。
    
    Args:
        files_path: 包含 CSV 文件的目录路径
    """
    # 读取翻译映射json
    with open('../../data/hospital_A/tmp_preprocessed_data/colname_translation.json', 'r', encoding='utf-8') as f:
        colname_mapping = json.load(f)

    # 遍历目录下所有 CSV 文件
    for filename in os.listdir(files_path):
        if filename.endswith('.csv'):
            file_path = os.path.join(files_path, filename)
            df = pd.read_csv(file_path)

            # 替换列名
            df.rename(columns=colname_mapping, inplace=True)

            # 保存修改后的文件
            df.to_csv(file_path, index=False)
            print(f"已更新文件: {filename}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Translate column names in CSV files')
    parser.add_argument('--files_path', type=str, required=True, help='Path to the directory containing CSV files')
    args = parser.parse_args()

    translate_colname(args.files_path)

  




