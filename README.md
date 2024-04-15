# Audio Server

这是一个使用 FastAPI 框架构建的音频服务器应用程序。它提供了以下功能:

- 初始化模型管理器,设置最大模型数量
- 添加新的模型,包括模型名称、GPT路径、Sovits路径、参考音频路径、提示文本和提示语言
- 生成单个音频文件并返回输出路径
- 生成单个音频文件,上传到腾讯云存储桶,并通过回调URL返回音频URL
- 批量生成多个音频文件并返回输出路径列表
- 获取当前可用的模型列表
- 下载生成的音频文件

## 安装

1. 克隆仓库:

   ```
   git clone https://github.com/your-username/audio-server.git
   cd audio-server
   ```

2. 安装依赖项:

   ```
   pip install -r requirements.txt
   ```

3. 配置腾讯云对象存储:

   在 `config.py` 文件中设置以下参数:

   ```python
   class Setting:
       cos_secret_id = "your-secret-id"
       cos_secret_key = "your-secret-key"
   ```

   将 `your-secret-id` 和 `your-secret-key` 替换为您的腾讯云对象存储的密钥信息。

## 使用

1. 启动服务器:

   ```
   python main.py --host 127.0.0.1 --port 8000
   ```

   可以使用 `--host` 和 `--port` 参数指定服务器的主机地址和端口号。

2. 初始化模型管理器:

   ```
   POST /init_manager/{max_models}
   ```

   将 `{max_models}` 替换为最大模型数量的整数值。

3. 添加模型:

   ```
   POST /add_model
   ```

   在请求体中以JSON格式提供以下字段:

   - `model_name`: 模型名称
   - `gpt_path`: GPT模型路径
   - `sovits_path`: Sovits模型路径
   - `ref_wav_path`: 参考音频路径
   - `prompt_text`: 提示文本
   - `prompt_language`: 提示语言

4. 生成单个音频文件:

   ```
   POST /generate_audio
   ```

   在请求体中以JSON格式提供以下字段:

   - `model_name`: 模型名称
   - `text`: 要合成的文本
   - `text_language`: 文本语言(可选,默认为"中文")

   服务器将返回生成的音频文件的输出路径。

5. 生成单个音频文件并上传到腾讯云存储桶:

   ```
   POST /generate_audio_url
   ```

   在请求体中以JSON格式提供以下字段:

   - `model_name`: 模型名称
   - `text`: 要合成的文本
   - `text_language`: 文本语言(可选,默认为"中文")
   - `callback_url`: 回调URL(可选)

   服务器将生成音频文件,上传到腾讯云存储桶,并通过回调URL返回音频URL(如果提供了回调URL)。

6. 批量生成多个音频文件:

   ```
   POST /generate_audio_batch
   ```

   在请求体中以JSON格式提供一个包含多个音频信息对象的列表,每个对象包含以下字段:

   - `model_name`: 模型名称
   - `text`: 要合成的文本
   - `text_language`: 文本语言(可选,默认为"中文")

   服务器将返回生成的音频文件的输出路径列表。

7. 获取可用的模型列表:

   ```
   GET /get_models
   ```

   服务器将返回当前可用的模型名称列表。

8. 下载生成的音频文件:

   ```
   GET /download_audio/{output_path}
   ```

   将 `{output_path}` 替换为要下载的音频文件的输出路径。

   服务器将返回指定路径的音频文件。

## 注意事项

- 确保在使用服务器之前正确配置了腾讯云对象存储的密钥信息。
- 生成的音频文件将保存在服务器的 `output` 目录下。
- 服务器使用了线程池来并发处理音频生成任务,以提高性能。
- 服务器使用了 CORS 中间件,允许来自任何来源的跨域请求。

## 许可

本项目基于 [MIT 许可](LICENSE) 发布。