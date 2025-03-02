
<img width="217" alt="image" src="https://github.com/user-attachments/assets/2dba0a02-a9fd-47a6-a413-b08edba6e960">
<img width="217" alt="image" src="https://github.com/user-attachments/assets/280e5478-ee1f-4d23-ad12-e30e94e54890">
=================================================================================

20250302，折腾了2天，没能更进一步，有点沮丧，不过看到晋级全球赛的项目，突然也好像有点想明白了，这不是一个科技产品竞赛，而是一个人文社科类的竞赛，好行并不关心用到了什么最新技术解决什么问题，更加关心调研数据，用户反馈，想法改进之类的，毕竟没有做出产品而只有原型设计，根据反馈多轮改善设计的项目都能晋级。

=================================================================================

20250212，IAYE入选了全国赛，但是和重要的考试时间冲突了，好悲催

=================================================================================

QWen vision model还挺好用的～

```python
    response = client.chat.completions.create(
        model="qwen-vl-max-latest",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "image_url",
                     "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}
                    },
                    {"type": "text",
                     "text": "你是一位盲人讲解员，帮忙将照片描述解读为更适合盲人理解的内容描述"
                    }
                ]
            }
        ],
    )
    description = response.choices[0].message.content
    print(f"Description: {description}")
```
让LLM改写了一下，感觉的确好一些

```python
    response = client.chat.completions.create(
        model="qwen-plus",
        messages=[
            {'role': 'system', 'content': '这张图片是一个盲人眼前的景象，你为他生成更加自然的场景描述，以便讲解给盲人听，让他感受到周边的环境，同时感受到周边的美，增强他的幸福感和自信心。以 < 在你的面前… > 开始，以 < 感谢你的聆听 > 结束。讲解中不要出现让盲人感到不舒服的内容，比如 < 你看不见 > 类似的描述，总字数保持在200字。'},
            {'role': 'user', 'content': description}],
    )
    description = response.choices[0].message.content
    print(f"Description: {description}")
```

QWen的service和api说明

https://help.aliyun.com/zh/model-studio/user-guide/vision

https://help.aliyun.com/zh/model-studio/user-guide/text-generation


QWen的sambert语音模型目前免费额度还不少

```python
        audio = SpeechSynthesizer.call(model='sambert-zhiting-v1',
                                        text=description,
                                        sample_rate=48000,
                                        format='wav')

        if audio.get_audio_data() is not None:
            with open('response.wav', 'wb') as f:
                f.write(audio.get_audio_data())
            print('SUCCESS: get audio data: %dbytes in response.wav' %
                  (sys.getsizeof(audio.get_audio_data())))
        else:
            print('ERROR: response is %s' % (audio.get_response()))
```

**树莓派Zero2W**
https://www.raspberrypi.com/products/raspberry-pi-zero-2-w/

<img width="478" alt="image" src="https://github.com/user-attachments/assets/30bf0d61-a155-4779-b59e-e4844c94f8d8" />
