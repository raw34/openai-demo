import openai
import configparser
import nltk
from tqdm import tqdm


def get_api_key():
    config = configparser.ConfigParser()
    config.read('../config.ini')
    return config['openai']['api_key']


def chat_completion(model_name, messages):
    openai.api_key = get_api_key()
    response = openai.ChatCompletion.create(
        model=model_name,
        messages=messages,
        temperature=0,
        max_tokens=4096
    )
    return response


def read_file_content(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()
        return content
    except FileNotFoundError:
        return f"No file found at {file_path}"
    except IOError:
        return "Error reading the file"


def append_to_file(file_path, text):
    # 使用 'a' 模式打开文件，如果文件不存在，它将被创建
    with open(file_path, 'a', encoding='utf-8') as file:
        # 将文本追加到文件中
        file.write(text + '\n\n')  # '\n' 是一个换行符，它会在每次追加文本后添加一个新行


def split_text_into_chunks(text, max_tokens):
    # nltk.download('punkt')
    """
    Split the input text into chunks, each containing no more than max_tokens tokens.

    Parameters:
    - text (str): The input text to be split.
    - max_tokens (int): The maximum number of tokens for each chunk.

    Returns:
    - List[str]: A list of strings, each string being a chunk of the input text.
    """
    # Tokenize the input text
    tokens = nltk.word_tokenize(text)

    # Initialize variables
    chunks = []
    current_chunk = []
    current_chunk_tokens = 0

    # Iterate through tokens and build chunks
    for token in tokens:
        # If adding the next token does not exceed the max_tokens limit
        if current_chunk_tokens + len(token) <= max_tokens:
            current_chunk.append(token)
            current_chunk_tokens += len(token)
        else:
            # Add the current chunk to the chunks list
            chunks.append(" ".join(current_chunk))
            # Start a new chunk with the current token
            current_chunk = [token]
            current_chunk_tokens = len(token)

    # Add the last chunk to the chunks list
    chunks.append(" ".join(current_chunk))

    return chunks


if __name__ == "__main__":
    # model_name = "davinci"
    # prompt = "Translate the following English text to French: 'Hello, how are you?'"
    # print(generate_text(model_name, prompt))

    # model_name = "gpt-3.5-turbo"
    model_name = "gpt-4"
    # prompt = "As an intelligent AI model, if you could be any fictional character, who would you choose and why?"
    prompt = "我提取了一个视频中的部分字幕，请帮用中文总结下这部分字幕的内容。以下是字幕：\n\n"
    file_names = [
        # '[English (auto-generated)] LORA + Checkpoint Model Training GUIDE - Get the BEST RESULTS super easy [DownSub.com]',
        # '[English (auto-generated)] LORA_ Install Guide and Super-High Quality Training - with Community Models!!! [DownSub.com]',
        '[English (auto-generated)] SDXL LORA STYLE Training! Get THE PERFECT RESULTS! [DownSub.com]',
        # '[English (auto-generated)] ULTIMATE SDXL LORA Training! Get THE BEST RESULTS! [DownSub.com]',
        # '[English] Become A Master Of SDXL Training With Kohya SS LoRAs - Combine Power Of Automatic1111 & SDXL LoRAs [DownSub.com]',
        # '[English] First Ever SDXL Training With Kohya LoRA - Stable Diffusion XL Training Will Replace Older Models [DownSub.com]',
        # '[English] Generate Studio Quality Realistic Photos By Kohya LoRA Stable Diffusion Training - Full Tutorial [DownSub.com]',
        # '[English] Transform Your Selfie into a Stunning AI Avatar with Stable Diffusion - Better than Lensa for Free [DownSub.com]',
        # '[English] Zero To Hero Stable Diffusion DreamBooth Tutorial By Using Automatic1111 Web UI - Ultra Detailed [DownSub.com]',
    ]

    for file_name in tqdm(file_names, desc='Processing files'):
        content = read_file_content(f'../data/{file_name}.txt')
        content_arr = split_text_into_chunks(content, 4000)
        for i, content in enumerate(tqdm(content_arr, desc=f'Processing chunks for {file_name}', leave=False)):
            messages = [
                {
                    "role": "user",
                    "content": prompt + content,
                },
            ]
            res = chat_completion(model_name, messages)
            append_to_file(f'../data/{file_name}_summary.txt', res['choices'][0]['message']['content'])

        print(f'Done processing {file_name}')
