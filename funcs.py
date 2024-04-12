import os, sys
from time import time as ttime
now_dir = os.getcwd()
sys.path.append(now_dir)
import torch
import librosa
from transformers import AutoModelForMaskedLM, AutoTokenizer
import numpy as np
from GPT_SoVITS.feature_extractor import cnhubert
from GPT_SoVITS.module.models import SynthesizerTrn
from GPT_SoVITS.AR.models.t2s_lightning_module import Text2SemanticLightningModule
from GPT_SoVITS.text import cleaned_text_to_sequence
from GPT_SoVITS.text.cleaner import clean_text
from GPT_SoVITS.module.mel_processing import spectrogram_torch
from GPT_SoVITS.my_utils import load_audio
import config as global_config
from tools.i18n.i18n import I18nAuto
i18n = I18nAuto()
import re

g_config = global_config.Config()

sovits_path = g_config.sovits_path
gpt_path = g_config.gpt_path
device = g_config.infer_device
cnhubert_base_path = g_config.cnhubert_path
bert_path = g_config.bert_path
is_half = g_config.is_half

if sovits_path == "":
    sovits_path = g_config.pretrained_sovits_path
    print(f"[WARN] 未指定SoVITS模型路径, fallback后当前值: {sovits_path}")
if gpt_path == "":
    gpt_path = g_config.pretrained_gpt_path
    print(f"[WARN] 未指定GPT模型路径, fallback后当前值: {gpt_path}")

print(f"[INFO] 半精: {is_half}")

dict_language={
    i18n("中文"):"zh",
    i18n("英文"):"en",
    i18n("日文"):"ja"
}
cnhubert.cnhubert_base_path = cnhubert_base_path
tokenizer = AutoTokenizer.from_pretrained(bert_path)
bert_model = AutoModelForMaskedLM.from_pretrained(bert_path)
if is_half:
    bert_model = bert_model.half().to(device)
else:
    bert_model = bert_model.to(device)


def get_bert_feature(text, word2ph):
    with torch.no_grad():
        inputs = tokenizer(text, return_tensors="pt")
        for i in inputs:
            inputs[i] = inputs[i].to(device)  #####输入是long不用管精度问题，精度随bert_model
        res = bert_model(**inputs, output_hidden_states=True)
        res = torch.cat(res["hidden_states"][-3:-2], -1)[0].cpu()[1:-1]
    assert len(word2ph) == len(text)
    phone_level_feature = []
    for i in range(len(word2ph)):
        repeat_feature = res[i].repeat(word2ph[i], 1)
        phone_level_feature.append(repeat_feature)
    phone_level_feature = torch.cat(phone_level_feature, dim=0)
    # if(is_half==True):phone_level_feature=phone_level_feature.half()
    return phone_level_feature.T

class DictToAttrRecursive:
    def __init__(self, input_dict):
        for key, value in input_dict.items():
            if isinstance(value, dict):
                # 如果值是字典，递归调用构造函数
                setattr(self, key, DictToAttrRecursive(value))
            else:
                setattr(self, key, value)

    def to_dict(self):
        """
        递归地将此对象转换回字典。
        """
        output_dict = {}
        for key, value in self.__dict__.items():
            if isinstance(value, DictToAttrRecursive):
                output_dict[key] = value.to_dict()
            else:
                output_dict[key] = value
        return output_dict

def get_spepc(hps, filename):
    base_path = os.getcwd()
    filename = os.path.join(base_path, filename)
    audio = load_audio(filename, int(hps.data.sampling_rate))
    audio = torch.FloatTensor(audio)
    audio_norm = audio
    audio_norm = audio_norm.unsqueeze(0)
    spec = spectrogram_torch(audio_norm, hps.data.filter_length, hps.data.sampling_rate, hps.data.hop_length,
                             hps.data.win_length, center=False)
    return spec


splits = {"，","。","？","！",",",".","?","!","~",":","：","—","…",}



def split(todo_text):
    todo_text = todo_text.replace("……", "。").replace("——", "，")
    if todo_text[-1] not in splits:
        todo_text += "。"
    i_split_head = i_split_tail = 0
    len_text = len(todo_text)
    todo_texts = []
    while 1:
        if i_split_head >= len_text:
            break  # 结尾一定有标点，所以直接跳出即可，最后一段在上次已加入
        if todo_text[i_split_head] in splits:
            i_split_head += 1
            todo_texts.append(todo_text[i_split_tail:i_split_head])
            i_split_tail = i_split_head
        else:
            i_split_head += 1
    return todo_texts


def cut1(inp):
    inp = inp.strip("\n")
    inps = split(inp)
    split_idx = list(range(0, len(inps), 4))
    split_idx[-1] = None
    if len(split_idx) > 1:
        opts = []
        for idx in range(len(split_idx) - 1):
            opts.append("".join(inps[split_idx[idx] : split_idx[idx + 1]]))
    else:
        opts = [inp]
    return "\n".join(opts)


def cut2(inp):
    inp = inp.strip("\n")
    inps = split(inp)
    if len(inps) < 2:
        return inp
    opts = []
    summ = 0
    tmp_str = ""
    for i in range(len(inps)):
        summ += len(inps[i])
        tmp_str += inps[i]
        if summ > 50:
            summ = 0
            opts.append(tmp_str)
            tmp_str = ""
    if tmp_str != "":
        opts.append(tmp_str)
    # print(opts)
    if len(opts)>1 and len(opts[-1]) < 50:  ##如果最后一个太短了，和前一个合一起
        opts[-2] = opts[-2] + opts[-1]
        opts = opts[:-1]
    return "\n".join(opts)


def cut3(inp):
    inp = inp.strip("\n")
    return "\n".join(["%s。" % item for item in inp.strip("。").split("。")])


def cut4(inp):
    inp = inp.strip("\n")
    return "\n".join(["%s." % item for item in inp.strip(".").split(".")])


def splite_en_inf(sentence, language):
    pattern = re.compile(r'[a-zA-Z. ]+')
    textlist = []
    langlist = []
    pos = 0
    for match in pattern.finditer(sentence):
        start, end = match.span()
        if start > pos:
            textlist.append(sentence[pos:start])
            langlist.append(language)
        textlist.append(sentence[start:end])
        langlist.append("en")
        pos = end
    if pos < len(sentence):
        textlist.append(sentence[pos:])
        langlist.append(language)

    return textlist, langlist


def clean_text_inf(text, language):
    phones, word2ph, norm_text = clean_text(text, language)
    phones = cleaned_text_to_sequence(phones)

    return phones, word2ph, norm_text


def get_bert_inf(phones, word2ph, norm_text, language):
    if language == "zh":
        bert = get_bert_feature(norm_text, word2ph).to(device)
    else:
        bert = torch.zeros(
            (1024, len(phones)),
            dtype=torch.float16 if is_half == True else torch.float32,
        ).to(device)

    return bert


def nonen_clean_text_inf(text, language):
    textlist, langlist = splite_en_inf(text, language)
    phones_list = []
    word2ph_list = []
    norm_text_list = []
    for i in range(len(textlist)):
        lang = langlist[i]
        phones, word2ph, norm_text = clean_text_inf(textlist[i], lang)
        phones_list.append(phones)
        if lang == "en" or "ja":
            pass
        else:
            word2ph_list.append(word2ph)
        norm_text_list.append(norm_text)
    phones = sum(phones_list, [])
    word2ph = sum(word2ph_list, [])
    norm_text = ' '.join(norm_text_list)

    return phones, word2ph, norm_text


def nonen_get_bert_inf(text, language):
    textlist, langlist = splite_en_inf(text, language)
    bert_list = []
    for i in range(len(textlist)):
        text = textlist[i]
        lang = langlist[i]
        phones, word2ph, norm_text = clean_text_inf(text, lang)
        bert = get_bert_inf(phones, word2ph, norm_text, lang)
        bert_list.append(bert)
    bert = torch.cat(bert_list, dim=1)

    return bert


def get_first(text):
    pattern = "[" + "".join(re.escape(sep) for sep in splits) + "]"
    text = re.split(pattern, text)[0].strip()
    return text


class TTSModel:
    def __init__(self, sovits_path=None, gpt_path=None) -> None:
        try:
            if sovits_path == None:
                sovits_path = g_config.pretrained_sovits_path
            dict_s2 = torch.load(sovits_path, map_location="cpu")
            self.hps = dict_s2["config"]
            self.hps = DictToAttrRecursive(self.hps)
            self.hps.model.semantic_frame_rate = "25hz"
        except Exception as e:
            raise RuntimeError(f"加载sovits模型配置失败: {e}")

        # 尝试加载gpt模型配置
        try:
            if gpt_path is None:
                gpt_path = g_config.pretrained_gpt_path
            dict_s1 = torch.load(gpt_path, map_location="cpu")
            self.config = dict_s1["config"]
        except Exception as e:
            raise RuntimeError(f"加载gpt模型配置失败: {e}")

        self.ssl_model = cnhubert.get_model()
        if is_half:
            self.ssl_model = self.ssl_model.half().to(device)
        else:
            self.ssl_model = self.ssl_model.to(device)

        self.vq_model = SynthesizerTrn(
            self.hps.data.filter_length // 2 + 1,
            self.hps.train.segment_size // self.hps.data.hop_length,
            n_speakers=self.hps.data.n_speakers,
            **self.hps.model.to_dict())
        if is_half:
            self.vq_model = self.vq_model.half().to(device)
        else:
            self.vq_model = self.vq_model.to(device)
        self.vq_model.eval()
        self.hz = 50
        self.max_sec = self.config['data']['max_sec']
        self.t2s_model = Text2SemanticLightningModule(self.config, "****", is_train=False)
        self.t2s_model.load_state_dict(dict_s1["weight"])
        if is_half:
            self.t2s_model = self.t2s_model.half()
        self.t2s_model = self.t2s_model.to(device)
        self.t2s_model.eval()
        total = sum([param.nelement() for param in self.t2s_model.parameters()])

    def get_tts_wav(self, ref_wav_path, prompt_text, prompt_language, text, text_language, how_to_cut=i18n("不切")):
        t0 = ttime()
        try:
            prompt_text = prompt_text.strip("\n")
            if(prompt_text[-1]not in splits):prompt_text+="。"if prompt_text!="en"else "."
            text = text.strip("\n")
            if(len(get_first(text))<4):text+="。"if text!="en"else "."
        except Exception as e:
            print(f"文本预处理出错: {str(e)}")
            raise e

        zero_wav = np.zeros(
            int(self.hps.data.sampling_rate * 0.3),
            dtype=np.float16 if is_half == True else np.float32,
        )

        try:
            if os.path.getsize(ref_wav_path) == 0:
                raise OSError(i18n("参考音频为空，请更换！"))
        except Exception as e:
            print(f"检查参考音频文件出错: {str(e)}")
            raise e

        try:
            with torch.no_grad():
                wav16k, sr = librosa.load(ref_wav_path, sr=16000)
                if(wav16k.shape[0]>160000 or wav16k.shape[0]<48000):
                    raise OSError(i18n("参考音频在3~10秒范围外，请更换！"))
                wav16k = torch.from_numpy(wav16k)
                zero_wav_torch = torch.from_numpy(zero_wav)
                if is_half == True:
                    wav16k = wav16k.half().to(device)
                    zero_wav_torch = zero_wav_torch.half().to(device)
                else:
                    wav16k = wav16k.to(device)
                    zero_wav_torch = zero_wav_torch.to(device)
                wav16k=torch.cat([wav16k,zero_wav_torch])
                ssl_content = self.ssl_model.model(wav16k.unsqueeze(0))[
                    "last_hidden_state"
                ].transpose(
                    1, 2
                )  # .float()
                codes = self.vq_model.extract_latent(ssl_content)
                prompt_semantic = codes[0, 0]
        except Exception as e:
            print(f"音频处理和特征提取出错: {str(e)}")
            raise e

        t1 = ttime()

        prompt_language = dict_language.get(prompt_language, "en")
        text_language = dict_language.get(text_language, "en")

        try:
            if prompt_language == "en":
                phones1, word2ph1, norm_text1 = clean_text_inf(prompt_text, prompt_language)
            else:
                phones1, word2ph1, norm_text1 = nonen_clean_text_inf(prompt_text, prompt_language)
            if(how_to_cut==i18n("凑四句一切")):text=cut1(text)
            elif(how_to_cut==i18n("凑50字一切")):text=cut2(text)
            elif(how_to_cut==i18n("按中文句号。切")):text=cut3(text)
            elif(how_to_cut==i18n("按英文句号.切")):text=cut4(text)
            text = text.replace("\n\n","\n").replace("\n\n","\n").replace("\n\n","\n")
            if(text[-1]not in splits):text+="。"if text_language!="en"else "."
            texts=text.split("\n")
        except Exception as e:
            print(f"文本切割和语言处理出错: {str(e)}")
            raise e

        audio_opt = []
        try:
            if prompt_language == "en":
                bert1 = get_bert_inf(phones1, word2ph1, norm_text1, prompt_language)
            else:
                bert1 = nonen_get_bert_inf(prompt_text, prompt_language)
        except Exception as e:
            print(f"BERT特征提取（提示文本）出错: {str(e)}")
            raise e

        for text in texts:
            try:
                if (len(text.strip()) == 0):
                    continue
                if text_language == "en":
                    phones2, word2ph2, norm_text2 = clean_text_inf(text, text_language)
                else:
                    phones2, word2ph2, norm_text2 = nonen_clean_text_inf(text, text_language)

                if text_language == "en":
                    bert2 = get_bert_inf(phones2, word2ph2, norm_text2, text_language)
                else:
                    bert2 = nonen_get_bert_inf(text, text_language)

                bert = torch.cat([bert1, bert2], 1)
                all_phoneme_ids = torch.LongTensor(phones1 + phones2).to(device).unsqueeze(0)
                bert = bert.to(device).unsqueeze(0)
                all_phoneme_len = torch.tensor([all_phoneme_ids.shape[-1]]).to(device)
                prompt = prompt_semantic.unsqueeze(0).to(device)
            except Exception as e:
                print(f"BERT特征提取（目标文本）或数据准备出错: {str(e)}")
                continue  # 跳过当前文本，处理下一个

            t2 = ttime()
            try:
                with torch.no_grad():
                    # pred_semantic = t2s_model.model.infer(
                    pred_semantic, idx = self.t2s_model.model.infer_panel(
                        all_phoneme_ids,
                        all_phoneme_len,
                        prompt,
                        bert,
                        top_k=self.config["inference"]["top_k"],
                        early_stop_num=self.hz * self.max_sec,
                    )
            except Exception as e:
                print(f"TTS模型推理出错: {str(e)}")
                raise e

            t3 = ttime()
            try:
                pred_semantic = pred_semantic[:, -idx:].unsqueeze(0)
                refer = get_spepc(self.hps, ref_wav_path)
                if is_half == True:
                    refer = refer.half().to(device)
                else:
                    refer = refer.to(device)
                
                audio = (
                    self.vq_model.decode(
                        pred_semantic, torch.LongTensor(phones2).to(device).unsqueeze(0), refer
                    )
                    .detach()
                    .cpu()
                    .numpy()[0, 0]
                )
                audio_opt.append(audio)
                audio_opt.append(zero_wav)
            except Exception as e:
                print(f"声音合成或后处理出错: {str(e)}")
                # 不抛出异常，尝试合成下一段文本

            t4 = ttime()

        try:
            print("%.3f\t%.3f\t%.3f\t%.3f" % (t1 - t0, t2 - t1, t3 - t2, t4 - t3))
            final_audio = (np.concatenate(audio_opt, 0) * 32768).astype(np.int16)
        except Exception as e:
            print(f"最终音频拼接或输出格式转换出错: {str(e)}")
            raise e

        yield self.hps.data.sampling_rate, final_audio
