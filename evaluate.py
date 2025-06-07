import json, time
from tqdm import tqdm
from PIL import Image
import io, os
import random
import argparse
import base64
import re
from datasets import load_dataset
import collections
from zhipuai import ZhipuAI
import torch
import clip
import sys

# ========= clip 模型加载函数（使用默认路径） =========
def load_clip_model(device):
    try:
        model, preprocess = clip.load("ViT-L/14", device=device)
        print("CLIP 模型已加载（默认路径）")
        return model, preprocess
    except Exception as e:
        print("加载 CLIP 模型失败：", e)
        sys.exit(1)

# 修改后的query_zhipuai函数（替换原query_gpt4v函数）
def query_zhipuai(model_name, image_path, prompt, retry=10):
    """使用智谱清言GLM-4V模型进行查询"""
    # 编码图片为base64
    with open(image_path, "rb") as img_file:
        img_base = base64.b64encode(img_file.read()).decode('utf-8')

    # 初始化客户端（注意环境变量名称改为ZHIPUAI_API_KEY）
    client = ZhipuAI(api_key="YOUR_API_KEY")

    for r in range(retry):
        try:
            response = client.chat.completions.create(
                model="glm-4v-plus-0111",  # GLM-4V模型固定名称
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": img_base}},
                        {"type": "text", "text": prompt}
                    ]
                }],
                max_tokens=10,
                temperature=0.0,
            )
            # 提取响应内容（根据智谱API的实际响应结构调整）
            print(f"Response from ZhipuAI: {response.choices[0].message.content}")  # 调试输出
            return response.choices[0].message.content
        except Exception as e:
            print(f"请求失败: {str(e)}")
            time.sleep(2)
    return 'Failed: Query GLM-4V Error'

# Function to encode the image
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')


def eval_mllm(data, t2i_model, evaluator, image_folder, eval_output_path):
    print(f"数据集包含 {len(data)} 条数据")

    if os.path.exists(eval_output_path):
        outputs = json.load(open(eval_output_path, 'r'))
    else:
        outputs = {}

    for i, d in tqdm(enumerate(data), desc="Evaluating images", total=len(data)):
        index = i + 1
        if str(index) in outputs:
            continue  # 如果已评估，跳过
        print(f"Processing {index}/{len(data)}")

        gpt_answers = {}
        extracted_answers = []

        prompt1_image_paths = [
            f'{image_folder}/prompt1/original/{str(index).zfill(4)}-{j + 1}.jpg' for j in range(4)
        ]
        prompt2_image_paths = [
            f'{image_folder}/prompt2/original/{str(index).zfill(4)}-{j + 1}.jpg' for j in range(4)
        ]

        description1 = d["description1"].replace("\n", ", ")
        description2 = d["description2"].replace("\n", ", ")
        degree = 'generally'

        prompt_eval1 = f'Can you tell me if the image {degree} fits the descriptions "{description1}"? If it {degree} fits the descriptions, then return 1, otherwise, return 0. Give me number 1 or 0 only.'
        prompt_eval2 = f'Can you tell me if the image {degree} fits the descriptions "{description2}"? If it {degree} fits the descriptions, then return 1, otherwise, return 0. Give me number 1 or 0 only.'

        for prompt_number, prompt_images in zip(['prompt1', 'prompt2'], [prompt1_image_paths, prompt2_image_paths]):
            for j in range(4):
                image_path = prompt_images[j]
                for eval_number, prompt_eval in zip(['description1', 'description2'], [prompt_eval1, prompt_eval2]):
                    key = f'{prompt_number}_image{j + 1}_{eval_number}'
                    try:
                        retry_limit = 3
                        for attempt in range(retry_limit):
                            glm_answer = query_zhipuai(evaluator, image_path, prompt_eval)
                            gpt_answers[key] = glm_answer

                            # 使用正则提取0或1
                            match = re.search(r'\b[01]\b', glm_answer)
                            if match:
                                extracted = int(match.group())
                                break  # 成功提取，跳出循环
                            else:
                                print(f"[警告] 回答不含有效数字 (0/1)，尝试第 {attempt + 1} 次：{glm_answer}")
                                time.sleep(1)

                        else:
                            print(f"[失败] 回答始终无法提取有效数字：{glm_answer}")
                            extracted = 0

                    except Exception as e:
                        print(f"[异常] 处理失败: {key}, 错误: {e}")
                        gpt_answers[key] = 'Error'
                        extracted = 0

                    extracted_answers.append(extracted)

        d['prompt1_image_paths'] = prompt1_image_paths
        d['prompt2_image_paths'] = prompt2_image_paths
        d['prompt_eval1'] = prompt_eval1
        d['prompt_eval2'] = prompt_eval2
        d['predictions'] = gpt_answers
        d['prediction_extracted'] = extracted_answers
        d['idx'] = index
        outputs[str(index)] = d

        json.dump(outputs, open(eval_output_path, 'w'), indent=4)

    # === 自动跳过 prediction_extracted 长度不足的样本 ===
    scores = {}
    for index, d in outputs.items():
        pred = d.get('prediction_extracted', [])
        if len(pred) != 16:
            print(f"[跳过] 样本 {index} prediction_extracted 数量为 {len(pred)}，应为 16")
            continue
        scores[int(index)] = pred

    score = get_score(scores)
    print(f'{evaluator} eval scores for task {t2i_model} is', round(score / len(scores) * 100, 2))


def eval_clip(data, t2i_model, evaluator, image_folder, eval_output_path, model, preprocess):
    def get_clip_score(image_path, evals, model, preprocess, device='cuda'):
        image_input = preprocess(Image.open(image_path)).unsqueeze(0).to(device)
        text_inputs = torch.cat([clip.tokenize(evals)]).to(device)
        with torch.no_grad():
            image_features = model.encode_image(image_input)
            text_features = model.encode_text(text_inputs)
        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)
        similarity = (100.0 * image_features @ text_features.T)
        similarity = similarity.softmax(dim=-1)
        return similarity.cpu().numpy().tolist()[0]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    outputs = {}
    for i, d in tqdm(enumerate(data), desc="Evaluating images", total=len(data)):
        index = i + 1
        extracted_answers = []
        clip_answers = {}
        prompt1_image_paths = [f"{image_folder}/prompt1/original/{str(index).zfill(4)}-{j+1}.jpg" for j in range(4)]
        prompt2_image_paths = [f"{image_folder}/prompt2/original/{str(index).zfill(4)}-{j+1}.jpg" for j in range(4)]
        description1 = d["description1"].replace("\n", ", ")
        description2 = d["description2"].replace("\n", ", ")

        for prompt_number, prompt_images in zip(['prompt1', 'prompt2'], [prompt1_image_paths, prompt2_image_paths]):
            for j in range(4):
                image_path = prompt_images[j].replace("\\", "/")  # 修复路径分隔符
                clip_similarities = get_clip_score(image_path, [description1, description2], model, preprocess, device)
                clip_answers[f'{prompt_number}_image{j+1}_eval1'] = clip_similarities[0]
                clip_answers[f'{prompt_number}_image{j+1}_eval2'] = clip_similarities[1]
                if clip_similarities[0] > clip_similarities[1]:
                    extracted_answers += [1, 0]
                elif clip_similarities[0] < clip_similarities[1]:
                    extracted_answers += [0, 1]
                else:
                    extracted_answers += [0, 0]

        d['prompt1_image_paths'] = prompt1_image_paths
        d['prompt2_image_paths'] = prompt2_image_paths
        d['prompt_eval1'] = description1
        d['prompt_eval2'] = description2
        d['predictions'] = clip_answers
        d['prediction_extracted'] = extracted_answers
        d['idx'] = index
        outputs[str(index)] = d

    with open(eval_output_path, 'w') as f:
        json.dump(outputs, f, indent=4)

    scores = {int(index): d['prediction_extracted'] for index, d in outputs.items() if len(d['prediction_extracted']) == 16}
    score = get_score(scores)
    print(f'{evaluator} eval scores for task {t2i_model} is', round(score / len(scores) * 100, 2))

# ========= 统一主控函数：同时运行 MLLM + CLIP 评估 =========
def eval_mllm_and_clip(data, t2i_model, evaluator, image_folder, eval_output_path, eval_output_path_clip):
    print("\n================ MLLM 模型评估开始 ================")
    eval_mllm(data, t2i_model, evaluator, image_folder, eval_output_path)

    print("\n================ CLIP 模型评估开始 ================")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = load_clip_model(device)

    eval_clip(data, t2i_model, 'clip', image_folder, eval_output_path_clip, model, preprocess)

def get_score(scores):
    score = 0
    for index, orig_scores in scores.items():
        # Rule: must be pairwise correct to be correct.
        support_eval = [tell_supporting_description(orig_scores[i*2], orig_scores[i*2+1]) for i in range(8)]
        s = sum([1 for i in range(4) if support_eval[i] == 1 and support_eval[i+4] == 2])/4
        score += s
    return score


def tell_supporting_description(score1, score2):
    if score1 == 1 and score2 == 0:
        return 1  # image supports description1
    elif score1 == 0 and score2 == 1:
        return 2  # image supports description2
    elif score1 == 1 and score2 == 1:
        return random.choice([1, 2])  # Randomly pick 1 or 2 if both descriptions are supported
    else:
        return 0  # Image fails to support either description

# 参数解析部分
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--evaluator", type=str, default='ZhipuAI', help="GPT4VO, ZhipuAI, clip")
    parser.add_argument("--t2i_model", type=str, default='flux_schnell', help="dalle-3, flux_schenel, sd_3, sd_xl, openjourneyv4")
    parser.add_argument("--use_negative_prompt", type=int, default=0, help="whether to use negative prompt")
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    # ==== 参数设置 ====
    t2i_model = 'flux_schnell' # 选择评估的模型 ← 此处请用户自行设置
    evaluator = 'ZhipuAI'
    use_negative_prompt = 0

    # ==== 路径配置 ====
    generated_image_root = r'RLCG\generated_images'

    # 定义 generated_image_dir
    generated_image_dir = f'{generated_image_root}/{t2i_model}{"neg" if use_negative_prompt else ""}_images'

    eval_output_path = f'{generated_image_root}/evals/ZhipuAI_eval/{t2i_model}{"neg" if use_negative_prompt else ""}.json'
    eval_output_path_clip = f'{generated_image_root}/evals/clip_eval/{t2i_model}{"neg" if use_negative_prompt else ""}.json'

    # 创建新路径的文件夹（如果文件夹不存在）
    os.makedirs(f'{generated_image_root}/evals/ZhipuAI_eval', exist_ok=True)
    os.makedirs(f'{generated_image_root}/evals/clip_eval', exist_ok=True)

    # ==== 加载数据集 ====
    try:
        dataset_path = r'RLCG\data.json'
        with open(dataset_path, 'r', encoding='utf-8') as f:
            raw_data = json.load(f)
        data = [{
            'prompt1': d['prompt1'],
            'prompt2': d['prompt2'],
            'description1': d['description1'],
            'description2': d['description2']
        } for d in raw_data]
        print(f"成功加载数据集，共 {len(data)} 条数据")
    except Exception as e:
        raise RuntimeError(f"加载数据失败: {e}")

    # ==== 同时评估 ZhipuAI 和 CLIP ====
    eval_mllm_and_clip(
        data=data,
        t2i_model=t2i_model,
        evaluator=evaluator,
        image_folder=generated_image_dir,
        eval_output_path=eval_output_path,
        eval_output_path_clip=eval_output_path_clip
    )
