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
        temperature=0.7,
        max_tokens=1024
    )
    return response


if __name__ == "__main__":
    # model_name = "davinci"
    # prompt = "Translate the following English text to French: 'Hello, how are you?'"
    # print(generate_text(model_name, prompt))

    # model_name = "gpt-3.5-turbo"
    model_name = "gpt-4"
    prompt = "As an intelligent AI model, if you could be any fictional character, who would you choose and why?"
    messages = [
        {
            "role": "user",
            "content": prompt,
        },
    ]
    print(generate_text_chat(model_name, messages))
