import os

from dotenv import find_dotenv, load_dotenv
from transformers import pipeline
import torch
import requests
import scipy


load_dotenv(find_dotenv())
token = os.getenv("HUGGING_FACE_API_TOKEN")

# image to text
def image2text(img_path):
    pipe = pipeline("image-to-text", model="Salesforce/blip-image-captioning-large")

    text = pipe(img_path)[0]['generated_text']

    return  text

# llm
def story():


    API_URL = "https://api-inference.huggingface.co/models/HuggingFaceH4/zephyr-7b-beta"
    headers = {"Authorization": f"Bearer {token}"}

    def query(payload):
        response = requests.post(API_URL, headers=headers, json=payload)
        return response.json()

    output = query({
        "inputs": f"{image2text('girl.jpg')} ",
    })
    print(output[0]['generated_text'])
    return output[0]['generated_text']

# text-to-speech


def speak():
    # API_URL = "https://api-inference.huggingface.co/models/suno/bark-small"
    # headers = {"Authorization": f"Bearer {token}"}
    #
    # def query(payload):
    #     response = requests.post(API_URL, headers=headers, json=payload)
    #     return response.content
    #
    # audio_bytes = query({
    #     "inputs": f"{story()}",
    # })
    # with open('audio.flac' ,'wb') as file:
    #     file.write(audio_bytes)

    API_URL = "https://api-inference.huggingface.co/models/espnet/kan-bayashi_ljspeech_vits"
    headers = {"Authorization": f"Bearer {token}"}

    payload = {
        "inputs": f"{story()}",
    }
    response = requests.post(API_URL, headers=headers, json=payload)
    with open('audio.wav', 'wb') as file:
        file.write(response.content)
speak()