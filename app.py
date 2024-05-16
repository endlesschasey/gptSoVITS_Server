import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
import argparse
from concurrent.futures import ThreadPoolExecutor
import soundfile as sf
import logging

from funcs import TTSModel
from tools.i18n.i18n import I18nAuto

i18n = I18nAuto()

import logging
import colorlog
import random, asyncio
from typing import Optional


logger = logging.getLogger()
logger.setLevel(logging.INFO)

# 创建控制台处理器
handler = logging.StreamHandler()

# 设置具有多种颜色的日志格式
color_formatter = colorlog.ColoredFormatter(
    "%(log_color)s%(levelname)s%(reset)s:     %(asctime)s%(reset)s - %(log_color)s%(message)s",
    datefmt="%Y-%m-%d  %H:%M:%S",
    log_colors={
        'DEBUG': 'cyan',
        'INFO': 'green',
        'WARNING': 'yellow',
        'ERROR': 'red',
        'CRITICAL': 'red,bg_white',
    },
    secondary_log_colors={
        'asctime': {
            'DEBUG': 'white',
            'INFO': 'white',
            'WARNING': 'white',
            'ERROR': 'white',
            'CRITICAL': 'white',
        }
    },
    reset=True
)

handler.setFormatter(color_formatter)

# 添加处理器到logger
logger.addHandler(handler)

class ModelInfo(BaseModel):
    role_name: str
    gpt_path: str
    sovits_path: str
    ref_wav_path: str
    prompt_text: str
    prompt_language: str
    disuse_model: str | None = None

class AudioInfo(BaseModel):
    task_id: str
    role_name: str
    text: str
    text_language: str = "中文"

class PredictManager:
    OUTPUT_DIR = "output"
    ROLE_PATH = "永夜"

    def __init__(self) -> None:
        self.predict_models: dict[str, TTSModel] = {}
        self.model_info: dict[str, tuple[str, str, str]] = {}
        self.model_executors: dict[str, ThreadPoolExecutor] = {}
        
    def init_manager(self, max_models: int) -> None:
        self.max_models = max_models

    def set_model(self, model_info: ModelInfo) -> None:
        role_name = model_info.role_name
        if len(self.predict_models) >= self.max_models and model_info.disuse_model in self.predict_models:
            self.remove_model(model_info.disuse_model)
        
        gpt_path, sovits_path = self.find_path(role_name, 'weight')
        self.predict_models[role_name] = TTSModel(sovits_path, gpt_path)
        self.model_info[role_name] = (model_info.ref_wav_path, model_info.prompt_text, model_info.prompt_language)
        self.model_executors[role_name] = ThreadPoolExecutor()

    def remove_model(self, role_name: str) -> None:
        if role_name in self.predict_models:
            del self.predict_models[role_name]
            del self.model_info[role_name]
            del self.model_executors[role_name]

    def generate_audio(self, audio_info: AudioInfo) -> str:
        role_name = audio_info.role_name
        if role_name not in self.predict_models:
            raise HTTPException(status_code=404, detail="Model not found")
        
        tts = self.predict_models[role_name]
        ref_wav_path, prompt_text, prompt_language = self.model_info[role_name]
        ref_wav_path = self.find_path(role_name, 'reference', ref_wav_path)
        try:
            sampling_rate, audio_data = next(tts.get_tts_wav(ref_wav_path, prompt_text, prompt_language, 
                                                        audio_info.text, audio_info.text_language))
            filename = f"{audio_info.task_id}.ogg"
            output_path = os.path.join(self.OUTPUT_DIR, filename)
            os.makedirs(self.OUTPUT_DIR, exist_ok=True)
            sf.write(output_path, audio_data, samplerate=sampling_rate)
            
            # 将相对路径转换为绝对路径
            absolute_output_path = os.path.abspath(output_path)
            return absolute_output_path
        except Exception as e:
            logger.exception(f"Error generating audio for task {audio_info.task_id}: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Error generating audio: {str(e)}")

    def submit_generate_audio(self, audio_info: AudioInfo) -> str:
        future = self.model_executors[audio_info.role_name].submit(self.generate_audio, audio_info)
        return future.result()

    def find_path(self, role_name, search_for='weight', ref_wav_path=None):
        # 获取当前脚本所在的目录
        current_dir = os.path.dirname(os.path.abspath(__file__))
        role_path = os.path.join(current_dir, self.ROLE_PATH, role_name)
        if search_for == 'weight':
            weight_path_pth = os.path.join(role_path, f"{role_name}.pth")
            weight_path_ckpt = os.path.join(role_path, f"{role_name}.ckpt")
            return weight_path_ckpt, weight_path_pth
        elif search_for == 'reference':
            ref_wav_path = os.path.join(role_path, 'audios', ref_wav_path)
            return ref_wav_path

app = FastAPI()
predict_manager = PredictManager()

@app.post("/init_manager/{max_models}")
async def init_manager(max_models: int):
    predict_manager.init_manager(max_models)
    return {"message": "Manager initialized"}

@app.post("/add_model")
async def add_model(model_info: ModelInfo):
    predict_manager.set_model(model_info)
    return {"message": "Model added successfully"}

@app.post("/generate_audio")
async def generate_audio(audio_info: AudioInfo):
    if audio_info.role_name not in predict_manager.predict_models:
        raise HTTPException(status_code=404, detail="Model not found")
    try:
        absolute_output_path = predict_manager.submit_generate_audio(audio_info)
        logger.info(f"Worker App output path: {absolute_output_path}")
        return {"task_id": audio_info.task_id, "status": "success", "output_path": absolute_output_path, "role_name": audio_info.role_name, "text": audio_info.text}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating audio: {str(e)}")

@app.get("/get_models")
async def get_models():
    return {"models": list(predict_manager.predict_models.keys())}

@app.get("/healthcheck")
async def healthcheck():
    return {"status": "ok"}

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