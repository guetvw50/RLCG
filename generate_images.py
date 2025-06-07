import os
import time
import random
import json
from PIL import Image
from huggingface_hub import InferenceClient
from zhipuai import ZhipuAI
from sentence_transformers import SentenceTransformer, util
import re
import openai
import requests

# === 模型选择配置 ===
MODEL_FOLDER = "openjourneyv4"  # 选择使用的模型 ← 此处请用户自行设置

# === 初始化句子嵌入模型，用于计算奖励 ===
similarity_model = SentenceTransformer('all-MiniLM-L6-v2')

# === 加载 JSON 数据集 ===
data_path = r"RLCG\data.json"
with open(data_path, "r") as f:
    dataset = json.load(f)

# === 初始化智谱清言 GPT ===
zhipu_client = ZhipuAI(api_key="YOUR_API_KEY")
# === 加载图像生成模型，包括 HuggingFace 和 OpenAI DALL·E 3 ===
def load_model_by_name(model_name):
    model_map = {
        "flux_schnell": "black-forest-labs/FLUX.1-schnell",
        "sd3": "stabilityai/stable-diffusion-3-medium-diffusers",
        "openjourneyv4": "prompthero/openjourney-v4",
        "sd_xl": "stabilityai/stable-diffusion-xl-base-1.0",
        "dalle-3": "openai/dall-e-3"
    }

    if model_name == "dalle-3":
        openai.api_key = "your-openai-api-key"  # ← 此处请用户自行设置
        print("DALL·E 3 via OpenAI API initialized.")
        return "openai"
    else:
        try:
            repo_id = model_map.get(model_name)
            if not repo_id:
                raise ValueError(f"模型名 '{model_name}' 未在模型映射中定义")
            client = InferenceClient(repo_id, token="your-huggingface-token")  # ← 用户自行配置
            print(f"{model_name} successfully initialized!")
            return client
        except Exception as e:
            print(f"Error initializing the client for {model_name}: {e}")
            return None

# === 无限重试生成反事实问题 ===
def generate_counterfactual_prompt_with_retry(original_prompt, delay=10):
    """无限重试生成反事实推理问题，直到成功"""
    while True:
        try:
            counterfactual_questions = generate_counterfactual_prompt(original_prompt)  # 使用原函数生成反事实问题
            if not counterfactual_questions:
                print("Error: Generated counterfactual questions are not valid. Retrying...")
                time.sleep(delay)
                continue  # 如果问题无效，重新尝试生成
            return counterfactual_questions
        except Exception as e:
            print(f"Failed to generate counterfactual questions: {e}")
            print(f"Retrying in {delay} seconds...")
            time.sleep(delay)

# === 无限重试生成图像 ===
def robust_text_to_image_with_retry(client, prompt, model_name=None, delay=60):
    while True:
        try:
            if model_name == "dalle-3":
                response = openai.Image.create(
                    model="dall-e-3",
                    prompt=prompt,
                    size="1024x1024",
                    n=1
                )
                image_url = response["data"][0]["url"]
                image = Image.open(requests.get(image_url, stream=True).raw)
                return image
            else:
                response = client.text_to_image(prompt)
                return response
        except Exception as e:
            print(f"Failed to generate image for prompt: {prompt}. Error: {e}")
            print(f"Retrying in {delay} seconds...")
            time.sleep(delay)


# === 拼接图片为 2x2 网格 ===
def create_grid(images):
    grid_size = (2, 2)
    img_width, img_height = images[0].size
    grid_width = grid_size[0] * img_width
    grid_height = grid_size[1] * img_height
    grid_image = Image.new('RGB', (grid_width, grid_height))
    for idx, img in enumerate(images):
        x_offset = (idx % 2) * img_width
        y_offset = (idx // 2) * img_height
        grid_image.paste(img, (x_offset, y_offset))
    return grid_image

# === 反事实问题生成 ===
def generate_counterfactual_prompt(original_prompt, retries=5, delay=20):
    """重试机制生成反事实推理问题（一直重试直到成功）"""
    attempt = 0
    while attempt < retries:
        try:
            system_message = (
                "你是一个生成反事实推理问题的ai助手，请为用户提供与提示词直接相关的、符合现实生活常识的反事实推理问题。问题应该关注于在生活常识下，提示词中的主体与客体应该处于什么状态或环境中。假设对你的提示词是“xxx”，你应该回答三个基于该提示词的反事实推理问题：如果“xxx”，那么…。其中xxx只是对提示词的统称，无实际意义。下面举出一个例子以便于你更好的理解我对你的要求，假设对你的提示词是“一只熊吃三文鱼”，你应该回答三个基于该提示词的反事实推理问题,分别是场景类：“如果一只熊吃三文鱼，那么这只熊应该在哪里捕猎？”，对象特征类：“如果一只熊吃三文鱼，那么被吃的三文鱼应该是什么样的？”，行为逻辑类：“如果一只熊吃三文鱼，那么这只熊应该是什么状态？”。请记住，提问的问题要符合原提示词所在的语境，请不要提出曲解原提示词语境的问题"
            )
            user_message = (
                f"给定提示词：{original_prompt}，请生成三个反事实推理问题。"
                "问题应探索提示词中的主体或主体与客体之间在上下文环境的互动和表现。"
            )
            response = zhipu_client.chat.completions.create(
                model="glm-4-plus",
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": user_message}
                ],
                timeout=20
            )
            if hasattr(response, 'choices') and response.choices and len(response.choices) > 0:
                content = response.choices[0].message.content
                pattern = re.compile(r"(\d\.\s+.*?(?=\n\d\.|\Z))", re.S)
                questions = [q.strip() for q in pattern.findall(content)]
                return questions
            else:
                print("Warning: No valid content in response.")
                return None
        except Exception as e:
            attempt += 1
            print(f"Attempt {attempt} failed: {e}")
            if attempt < retries:
                print(f"Retrying after {delay} seconds...")
                time.sleep(delay)  # Wait before retrying
            else:
                print("Max retries reached. Returning None.")
                return None

# === 回答问题（ZhipuAI）===
def get_answers_for_questions(client, questions, retries=5, delay=20):
    """重试机制回答问题（一直重试直到成功）"""
    answers = []
    if not questions:  # 检查问题是否有效
        print("Warning: No valid questions to process.")
        return answers  # 如果没有有效问题，直接返回空答案列表

    for question in questions:
        if question.strip():
            attempt = 0
            while attempt < retries:
                try:
                    response = client.chat.completions.create(
                        model="glm-4-plus",
                        messages=[{"role": "system", "content": "你是一个知识助手，你的任务是根据问题提供简要、准确的答案，最好是一句话概括。请用英文回答。"},
                                  {"role": "user", "content": f"问题是：{question}"}],
                        timeout=20
                    )
                    if response and response.choices:
                        answer = response.choices[0].message.content
                        print(f"Answer for question '{question}': {answer}")
                        answers.append(answer)
                        break
                    else:
                        answers.append(None)
                        break
                except Exception as e:
                    attempt += 1
                    print(f"Attempt {attempt} failed for question '{question}': {e}")
                    if attempt < retries:
                        print(f"Retrying after {delay} seconds...")
                        time.sleep(delay)  # Wait before retrying
                    else:
                        print("Max retries reached for this question.")
                        answers.append(None)
    return answers

# === 增强提示词 ===
def enhance_prompt_with_answers(prompt, questions, answers):
    original = prompt.strip()
    answers_combined = " ".join([answer.strip() for answer in answers if answer])
    return f'"{original} {answers_combined}"'

# === PPO优化机制 ===
def apply_ppo(enhanced_prompt):
    content = enhanced_prompt.strip('"')
    parts = content.split()
    if len(parts) < 2:
        return enhanced_prompt
    original_prompt = parts[0]
    expected_answer = " ".join(parts[1:])
    updated_prompt = ppo_training_loop(enhanced_prompt, expected_answer, original_prompt)
    return updated_prompt

def ppo_training_loop(enhanced_prompt, expected_answer, original_prompt, num_iterations=3):
    current_prompt = enhanced_prompt
    for iteration in range(num_iterations):
        candidate_modifications = [" better", " improved", " enhanced", " optimized"]
        best_reward = -float("inf")
        best_prompt = current_prompt
        for mod in candidate_modifications:
            candidate_prompt = current_prompt.rstrip('"') + " " + mod + "\""
            reward = compute_reward(candidate_prompt, expected_answer, original_prompt)
            print(f"Iteration {iteration + 1}, candidate '{mod}': reward {reward:.4f}")
            if reward > best_reward:
                best_reward = reward
                best_prompt = candidate_prompt
        print(f"Best candidate for iteration {iteration + 1}: {best_prompt} with reward {best_reward:.4f}")
        if best_reward >= 0.8:
            current_prompt = best_prompt
            break
        current_prompt = best_prompt
    return current_prompt

def compute_reward(generated_prompt, expected_answer, original_prompt, diversity_weight=0.3):
    embedding_gen = similarity_model.encode(generated_prompt, convert_to_tensor=True)
    embedding_exp = similarity_model.encode(expected_answer, convert_to_tensor=True)
    embedding_orig = similarity_model.encode(original_prompt, convert_to_tensor=True)

    cosine_sim = util.cos_sim(embedding_gen, embedding_exp)
    semantic_reward = (cosine_sim.item() + 1) / 2

    cosine_div = util.cos_sim(embedding_gen, embedding_orig)
    diversity_reward = 1 - ((cosine_div.item() + 1) / 2)

    total_reward = (1 - diversity_weight) * semantic_reward + diversity_weight * diversity_reward
    return total_reward


# === 图像生成主逻辑 ===
def generate_images(client, prompt, prompt_id, i, output_dir):
    print(f"The original prompt is: {prompt}")

    # 反事实推理生成步骤（无限重试）
    counterfactual_questions = generate_counterfactual_prompt_with_retry(prompt)

    # 确保生成的反事实问题是一个有效的可迭代对象
    if not counterfactual_questions or not isinstance(counterfactual_questions, list):
        print("Error: Generated counterfactual questions are not valid.")
        return  # 如果问题无效，跳过该图像生成

    print("Generated counterfactual questions:", counterfactual_questions)
    answers = get_answers_for_questions(zhipu_client, counterfactual_questions)
    print("Answers:", answers)
    enhanced_prompt = enhance_prompt_with_answers(prompt, counterfactual_questions, answers)

    print(f"Enhanced prompt before PPO: {enhanced_prompt}")
    ppo_enhanced_prompt = apply_ppo(enhanced_prompt)
    print(f"Enhanced prompt after PPO: {ppo_enhanced_prompt}")

    model_output_dir = os.path.join(output_dir, f"{MODEL_FOLDER}_images", prompt_id)
    individual_output_dir = os.path.join(model_output_dir, "original")
    grid_output_dir = model_output_dir
    os.makedirs(individual_output_dir, exist_ok=True)
    os.makedirs(grid_output_dir, exist_ok=True)

    images = []
    for j in range(1, 5):
        seed = random.randint(0, 100000)
        modified_prompt = f"{ppo_enhanced_prompt} [Seed: {seed}]"
        print(f"Generating image with modified prompt: {modified_prompt}")
        image = robust_text_to_image_with_retry(client, modified_prompt, model_name=MODEL_FOLDER)
        image_path = os.path.join(individual_output_dir, f"{i}-{j}.jpg")
        image.save(image_path)
        print(f"Saved individual image to {image_path}")
        images.append(image)

    if images:
        grid_image = create_grid(images)
        grid_image_path = os.path.join(grid_output_dir, f"{i}.jpg")
        grid_image.save(grid_image_path)
        print(f"Saved grid image to {grid_image_path}")

# === 主程序入口 ===
if __name__ == "__main__":
    client = load_model_by_name(MODEL_FOLDER)
    if client is None:
        print("Error: Failed to initialize the model. Exiting program.")
        exit(1)

    output_image_dir = r'RLCG\generated_images'

    # 从第几组开始处理（0-based index）
    start_index = 0  # 注意第35组在索引上是34（Python从0开始）
    i = start_index + 1  # 图像命名从 0035 开始

    for idx, item in enumerate(dataset):
        if idx < start_index:
            continue  # 跳过前面已经处理过的组

        prompt1 = item['prompt1']
        prompt2 = item['prompt2']

        generate_images(client, prompt1, "prompt1", f"{i:04d}", output_image_dir)
        generate_images(client, prompt2, "prompt2", f"{i:04d}", output_image_dir)

        i += 1
