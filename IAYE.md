
<img width="217" alt="image" src="https://github.com/user-attachments/assets/2dba0a02-a9fd-47a6-a413-b08edba6e960">

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
