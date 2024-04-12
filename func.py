import datetime 
import os, re
import py7zr
import zipfile
import rarfile
import tarfile
import uuid
import json, requests



ALLOWED_EXTENSIONS = {'zip', 'rar', '7z'}

def change_choices():
    SoVITS_names, GPT_names = get_weights_names()
    return {"data": {"SoVITS_names": sorted(SoVITS_names,key=custom_sort_key), "GPT_names": sorted(GPT_names,key=custom_sort_key)}}

pretrained_sovits_name="GPT_SoVITS/pretrained_models/s2G488k.pth"
pretrained_gpt_name="GPT_SoVITS/pretrained_models/s1bert25hz-2kh-longer-epoch=68e-step=50232.ckpt"
SoVITS_weight_root="SoVITS_weights"
GPT_weight_root="GPT_weights"
os.makedirs(SoVITS_weight_root,exist_ok=True)
os.makedirs(GPT_weight_root,exist_ok=True)
def get_weights_names():
    SoVITS_names = [pretrained_sovits_name]
    for name in os.listdir(SoVITS_weight_root):
        if name.endswith(".pth"):SoVITS_names.append("%s/%s"%(SoVITS_weight_root,name))
    GPT_names = [pretrained_gpt_name]
    for name in os.listdir(GPT_weight_root):
        if name.endswith(".ckpt"): GPT_names.append("%s/%s"%(GPT_weight_root,name))
    return SoVITS_names,GPT_names


def custom_sort_key(s):
    # 使用正则表达式提取字符串中的数字部分和非数字部分
    parts = re.split('(\d+)', s)
    # 将数字部分转换为整数，非数字部分保持不变
    parts = [int(part) if part.isdigit() else part for part in parts]
    return parts

def sanitize_filename(text):
    """将文本转换为安全的文件名。"""
    text = re.sub(r'[\\/*?:"<>|\n]', "", text)  # 移除不合法字符，包括换行符
    return text[:15]  # 限制文件名长度，以防文本过长

def generate_filename(text, model_name, extension=".wav"):
    """生成一个基于文本和时间戳的唯一文件名。"""
    sanitized_text = sanitize_filename(text)
    timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    return f"{sanitized_text}_{model_name}_{timestamp}{extension}"


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def extract_file(file_path, extract_to):
    """根据文件扩展名解压文件到指定目录"""
    try:
        if file_path.endswith('.zip'):
            with zipfile.ZipFile(file_path, 'r') as zip_ref:
                zip_ref.extractall(extract_to)
        elif file_path.endswith('.rar'):
            with rarfile.RarFile(file_path, 'r') as rar_ref:
                rar_ref.extractall(extract_to)
        elif file_path.endswith('.tar.gz') or file_path.endswith('.tgz'):
            with tarfile.open(file_path, 'r:gz') as tar_ref:
                tar_ref.extractall(extract_to)
        elif file_path.endswith('.7z'):
            with py7zr.SevenZipFile(file_path, mode='r') as seven_ref:
                seven_ref.extractall(path=extract_to)
        else:
            print(f"不支持的文件格式: {file_path}")
    except Exception as e:
        print(f"解压文件时出错: {e}")

    

def SendMsg(content, mentioned_list=[], key = "911b38ce-d056-4182-80fa-3b2485e9718a"):
    key = "fd13490d-aadd-4266-8a20-4aa206473943"
    body = {"msgtype": "text", "text": {"content": content, "mentioned_list": mentioned_list}}
    headers = {'content-type': "application/json"}
    data = json.dumps(body)
    rtxUrl = f"https://qyapi.weixin.qq.com/cgi-bin/webhook/send?key={key}"
    # print(rtxUrl)
    response = requests.post(rtxUrl, data = data, headers = headers)
    # print(response)


def upload_bucket(client, file_path):
    key = str(uuid.uuid4()) + file_path[-4:]
    bucket='up-down-1256453865'

    response = client.upload_file(
        Bucket=bucket,
        Key=key,
        LocalFilePath=file_path,
        EnableMD5=False,
        progress_callback=None
    )
    url = f"https://{bucket}.cos.accelerate.myqcloud.com/{key}"
    return url

def call_back(call_back_url, output_url):
    try:
        response = requests.get(call_back_url, params={"url": output_url})
        response.raise_for_status()
        return {"message": "User created and webhook notified"}
    except requests.RequestException as e:
        print(f"An error occurred: {str(e)}")