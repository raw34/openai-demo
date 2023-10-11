import openai
import configparser


def get_api_key():
    config = configparser.ConfigParser()
    config.read('../config.ini')
    return config['openai']['api_key']


def generate_text(model_name, prompt):
    openai.api_key = get_api_key()
    response = openai.Completion.create(
        engine=model_name,
        prompt=prompt,
        temperature=0.7,
        max_tokens=50
    )
    return response


def generate_text_chat(model_name, messages):
    openai.api_key = get_api_key()
    response = openai.ChatCompletion.create(
        model=model_name,
        messages=messages,
        temperature=0,
        max_tokens=8192
    )
    return response


def read_file_content(file_path):
    """
    Read the content of a text file and return it as a string.

    Parameters:
        file_path (str): The path to the file to be read.

    Returns:
        str: The content of the file as a string.
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()
        return content
    except FileNotFoundError:
        return f"No file found at {file_path}"
    except IOError:
        return "Error reading the file"



if __name__ == "__main__":
    # model_name = "davinci"
    # prompt = "Translate the following English text to French: 'Hello, how are you?'"
    # print(generate_text(model_name, prompt))

    # model_name = "gpt-3.5-turbo"
    model_name = "gpt-4"
    # prompt = "As an intelligent AI model, if you could be any fictional character, who would you choose and why?"
    prompt = "我提取了一个视频中的字幕，请帮我总结下这个视频的内容。以下是字幕：\n\n"
    content = read_file_content('../data/[English (auto-generated)] LORA + Checkpoint Model Training GUIDE - Get the BEST RESULTS super easy [DownSub.com].txt')
    messages = [
        {
            "role": "user",
            "content": prompt + content,
        },
    ]
    print(messages)
    # print(generate_text_chat(model_name, messages))
