import uvicorn, uuid
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel
import os, argparse
import soundfile as sf
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from qcloud_cos import CosConfig
from qcloud_cos import CosS3Client

from tools.i18n.i18n import I18nAuto
i18n = I18nAuto()
from funcs import TTSModel
from func import generate_filename, upload_bucket, call_back
from config import Setting

class ModelInfo(BaseModel):
    model_name: str
    gpt_path: str
    sovits_path: str
    ref_wav_path: str
    prompt_text: str
    prompt_language: str

class AudioInfo(BaseModel):
    model_name: str
    text: str
    text_language: str = "中文"
    
    callback_url: str = "http://119.3.93.154/url.php"

class AudioResponse(BaseModel):
    task_id: str
    model_name: str
    output_path: str
    text: str

class PredictManager:
    OUTPUT_DIR = "output"

    def __init__(self) -> None:
        self.predict_models = {}
        self.model_info = {}
        self.model_executors = {}
        self.model_usage = {}
        self.max_models = 0
        self.client = self.init_bucket()
        
    def init_bucket(self, ):

        region = 'ap-shanghai'
                                
        token = None
        scheme = 'https'

        config = CosConfig(Region=region, SecretId=Setting.cos_secret_id, SecretKey=Setting.cos_secret_key, Token=token, Scheme=scheme)
        client = CosS3Client(config)
        return client
    
    def upload_bucket(self, output_path):
        # TODO: 用线程或者其他方法上传音频文件至云
        url = upload_bucket(self.client, output_path)
        return url

    def init_manager(self, max_models: int) -> None:
        self.max_models = max_models

    def set_model(self, model_name: str, gpt_path: str, sovits_path: str, ref_wav_path: str, prompt_text: str, prompt_language: str) -> None:
        if len(self.predict_models) >= self.max_models:
            # 根据使用频数和最近使用时间来替换使用较少的模型
            least_used_model = min(self.model_usage, key=lambda x: (self.model_usage[x]['count'], self.model_usage[x]['last_used']))
            del self.predict_models[least_used_model]
            del self.model_info[least_used_model]
            del self.model_executors[least_used_model]
            del self.model_usage[least_used_model]

        self.predict_models[model_name] = TTSModel(None, None)
        self.model_info[model_name] = (ref_wav_path, prompt_text, prompt_language)
        self.model_executors[model_name] = ThreadPoolExecutor()
        self.model_usage[model_name] = {'count': 0, 'last_used': datetime.now()}

    def get_model(self, model_name) -> TTSModel:
        return self.predict_models[model_name]

    def generate_audio(self, model_name: str, text: str, text_language: str) -> str:
        tts = self.get_model(model_name)
        ref_wav_path, prompt_text, prompt_language = self.model_info[model_name]
        sampling_rate, audio_data = next(
                        tts.get_tts_wav(ref_wav_path, prompt_text, prompt_language, text, text_language, how_to_cut=i18n("不切")))
        
        filename = generate_filename(text, model_name)
        output_path = os.path.join(self.OUTPUT_DIR, filename)
        os.makedirs(self.OUTPUT_DIR, exist_ok=True)
        sf.write(output_path, audio_data, samplerate=sampling_rate)

        # 更新模型的使用频数和最近使用时间
        self.model_usage[model_name]['count'] += 1
        self.model_usage[model_name]['last_used'] = datetime.now()

        return output_path

    def submit_generate_audio(self, model_name: str, text: str, text_language: str) -> str:
        executor = self.model_executors[model_name]
        future = executor.submit(self.generate_audio, model_name, text, text_language)
        output_path = future.result()

        # 更新模型的使用频数和最近使用时间
        self.model_usage[model_name]['count'] += 1
        self.model_usage[model_name]['last_used'] = datetime.now()

        return output_path
    
    def submit_generate_audio_batch(self, audio_info_list: list[AudioInfo]) -> list[str]:
        output_paths = []
        model_futures = {}
        for audio_info in audio_info_list:
            if audio_info.model_name not in model_futures:
                model_futures[audio_info.model_name] = []
            future = self.submit_generate_audio(audio_info.model_name, audio_info.text, audio_info.text_language)
            model_futures[audio_info.model_name].append(future)
        
        for model_name, futures in model_futures.items():
            for future in futures:
                output_path = future.result()
                output_paths.append(output_path)
        
        return output_paths

app = FastAPI()
predict_manager = PredictManager()

@app.post("/init_manager/{max_models}")
async def init_manager(max_models: int):
    try:
        predict_manager.init_manager(max_models)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return {"message": "Manager initialized successfully"}

@app.post("/add_model")
async def add_model(model_info: ModelInfo):
    predict_manager.set_model(model_info.model_name, model_info.gpt_path, model_info.sovits_path,
                              model_info.ref_wav_path, model_info.prompt_text, model_info.prompt_language)
    return {"message": "Model added successfully"}

@app.post("/generate_audio")
async def generate_audio(audio_info: AudioInfo):
    if audio_info.model_name not in predict_manager.predict_models:
        raise HTTPException(status_code=404, detail="Model not found")
    try:
        output_path = predict_manager.submit_generate_audio(audio_info.model_name, audio_info.text, audio_info.text_language)
        return {"output_path": output_path}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating audio: {str(e)}")

@app.post("/generate_audio_url")
async def generate_audio_url(audio_info: AudioInfo):
    if audio_info.model_name not in predict_manager.predict_models:
        raise HTTPException(status_code=404, detail="Model not found")
    try:
        output_path = predict_manager.submit_generate_audio(audio_info.model_name, audio_info.text, audio_info.text_language)
        url = predict_manager.upload_bucket(output_path)
        call_back(audio_info.callback_url, url) if audio_info.callback_url else None
        return {"output_path": output_path}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating audio: {str(e)}")        

@app.post("/generate_audio_batch")
async def generate_audio_batch(audio_info_list: list[AudioInfo]):
    output_paths = predict_manager.submit_generate_audio_batch(audio_info_list)
    return {"output_paths": output_paths}

@app.get("/get_models")
async def get_models():
    return {"models": list(predict_manager.predict_models.keys())}

@app.get("/download_audio/{output_path}")
async def download_audio(output_path: str):
    try:
        return FileResponse(output_path, media_type="audio/wav", filename=os.path.basename(output_path))
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Audio file not found")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Audio Server")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Host address")
    parser.add_argument("--port", type=int, default=8000, help="Port number")
    args = parser.parse_args()

    uvicorn.run(app, host=args.host, port=args.port)