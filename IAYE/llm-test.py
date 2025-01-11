import sys
import base64
import dashscope
import pygame
from dashscope.audio.tts import SpeechSynthesizer
from openai import OpenAI

# 将一幅图片的数据编码成base64
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

# QWen模型接口兼容OpenAI API格式，定义license key和模型调用地址
client = OpenAI(
    api_key="sk-xxxxxxxxxxxxxxxxx",
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)
dashscope.api_key = "sk-xxxxxxxxxxxxxxxxx"

def get_image_description():
    base64_image = encode_image("./2.jpg")
    # 调用Vision模型将image解读成text
    response = client.chat.completions.create(
        model="qwen-vl-max-latest",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"} },
                    {"type": "text", "text": "你是一位盲人讲解员，帮忙将照片描述解读为更适合盲人理解的内容描述"}
                ]}
        ])
    description = response.choices[0].message.content
    print(f"Description: {description}")

    # 调用LLM模型将文字描述转化为更自然的描述
    response = client.chat.completions.create(
        model="qwen-plus",
        messages=[
            {'role': 'system', 'content': '这张图片是一个盲人眼前的景象，你为他生成更加自然的场景描述，以便讲解给盲人听，让他感受到周边的环境，同时感受到周边的美，增强他的幸福感和自信心。以 < 在你的面前… > 开始，以 < 感谢你的聆听 > 结束。讲解中不要出现让盲人感到不舒服的内容，比如 < 你看不见 > 类似的描述，总字数保持在200字。'},
            {'role': 'user', 'content': description}],
    )
    description = response.choices[0].message.content
    print(f"Description: {description}")

    return description

def main():
    print("Starting the Vision-to-Voice System...")
    while True:
        print("Getting image description...")
        description = get_image_description()

        # 调用Text2Audio模型将Text转为语音
        print("Converting to speech...")
        audio = SpeechSynthesizer.call(model='sambert-zhiting-v1',
                                        text=description,
                                        sample_rate=48000,
                                        format='wav')

        # 将语音存储为wav文件
        if audio.get_audio_data() is not None:
            with open('response.wav', 'wb') as f:
                f.write(audio.get_audio_data())
            print('SUCCESS: get audio data: %dbytes in response.wav' %(sys.getsizeof(audio.get_audio_data())))
        else:
            print('ERROR: response is %s' % (audio.get_response()))

        # 加载音频并播放wav文件
        pygame.mixer.init()
        pygame.mixer.music.load("response.wav")
        pygame.mixer.music.play()

        # 等待播放完成
        while pygame.mixer.music.get_busy():
            pygame.time.Clock().tick(10)  # 等待

        break

if __name__ == "__main__":
    main()