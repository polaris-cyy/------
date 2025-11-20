#import "template.typ": *

#let code = (

data-process: ```python
import csv

train_data = './datasets/68f201a04e0f8ad44a62069b-momodel/train_data.csv'
symptoms_data, ZX, ZF = [], [], []

with open(train_data, 'r', encoding='utf-8') as f:
    reader = csv.reader(f)
    header = next(reader)
    for row in reader:
        symptoms_data.append(row[1])  # 症状列
        ZX.append(row[2])             # 证型列
        ZF.append(row[3])             # 治法列

```,

system-prompt: ```python
system_prompt = """
你是一位经验丰富的中医专家，有三十年的问诊经验，可以根据患者症状准确描述判断证型和治法。
请用JSON格式输出：{"证型":"", "治法":""}
"""
system_prompt += str(symptoms_data)
system_prompt += str(ZX)
system_prompt += str(ZF)

```,

llm-inference: ```python
from openai import OpenAI

client = OpenAI(api_key="YOUR_API_KEY", base_url="https://aistudio.baidu.com/llm/lmapi/v3")

messages = [
    {"role": "system", "content": system_prompt},
    {"role": "user", "content": f"患者症状：{symptoms}"}
]

response = client.chat.completions.create(
    model="ernie-4.5-0.3b",
    messages=messages,
    max_completion_tokens=512,
    temperature=0.3,
    top_p=0.7
)

```,

response-parse: ````python
import json, re

def parse_response(content):
    code_block_pattern = re.compile(r"```json\s*(.*?)\s*```", re.DOTALL)
    match = code_block_pattern.search(content)
    if match:
        json_str = match.group(1).strip()
        return json.loads(json_str)

def predict(symptom):
    model_result = get_result(symptom)
    zx_predict, zf_predict = model_result["证型"], model_result["治法"]
    return zx_predict, zf_predict

````,

response-time: ```python
response = client.chat.completions.create(
    model="ernie-4.5-0.3b",
    messages=messages,
    max_completion_tokens=512,
    temperature=0.3,
    top_p=0.7
)

```

) // End of namespace code