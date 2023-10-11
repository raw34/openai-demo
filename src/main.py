import openai
import configparser
import nltk


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
        max_tokens=8192
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
    prompt = "我提取了一个视频中的字幕，请帮我总结下这个视频的内容。以下是字幕：\n\n"
    content = read_file_content('../data/[English (auto-generated)] LORA + Checkpoint Model Training GUIDE - Get the BEST RESULTS super easy [DownSub.com].txt')
    content_arr = split_text_into_chunks(content, 2048)
    for content in content_arr:
        messages = [
            {
                "role": "user",
                "content": prompt + content,
            },
        ]
        print(chat_completion(model_name, messages))
