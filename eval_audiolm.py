import argparse
import random
import time
import warnings
import wave
from pathlib import Path

import numpy as np
import torch
from scipy.signal import resample_poly
from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer

from model.model_audiolm import AudioLMConfig, MiniMindAudioLM
from trainer.trainer_utils import get_model_params, setup_seed

warnings.filterwarnings('ignore')
PROJECT_ROOT = Path(__file__).resolve().parent
TARGET_SAMPLING_RATE = 16000


def resolve_path(path_str: str) -> Path:
    path = Path(path_str)
    return path if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def resample_audio(audio: np.ndarray, orig_sr: int, target_sr: int = TARGET_SAMPLING_RATE):
    if orig_sr == target_sr:
        return audio, orig_sr, False

    gcd = np.gcd(orig_sr, target_sr)
    audio = resample_poly(audio, up=target_sr // gcd, down=orig_sr // gcd).astype(np.float32)
    return audio, target_sr, True


def load_wav(audio_path: Path):
    with wave.open(str(audio_path), 'rb') as wav_file:
        sample_rate = wav_file.getframerate()
        num_channels = wav_file.getnchannels()
        sample_width = wav_file.getsampwidth()
        num_frames = wav_file.getnframes()
        frames = wav_file.readframes(num_frames)

    if sample_width == 1:
        audio = np.frombuffer(frames, dtype=np.uint8).astype(np.float32)
        audio = (audio - 128.0) / 128.0
    elif sample_width == 2:
        audio = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
    elif sample_width == 4:
        audio = np.frombuffer(frames, dtype=np.int32).astype(np.float32) / 2147483648.0
    else:
        raise ValueError(f'Unsupported wav sample width: {sample_width}')

    if num_channels > 1:
        audio = audio.reshape(-1, num_channels).mean(axis=1)

    return audio, sample_rate


def get_weight_path(args) -> Path:
    if args.weight_path:
        return resolve_path(args.weight_path)
    moe_suffix = '_moe' if args.use_moe else ''
    return resolve_path(args.save_dir) / f'{args.weight}_{args.hidden_size}{moe_suffix}.pth'


def init_native_model(args):
    tokenizer = AutoTokenizer.from_pretrained(resolve_path(args.tokenizer_path))
    model = MiniMindAudioLM(
        AudioLMConfig(
            hidden_size=args.hidden_size,
            num_hidden_layers=args.num_hidden_layers,
            use_moe=bool(args.use_moe),
        ),
        audio_model_path=str(resolve_path(args.audio_model_path)),
    )
    weight_path = get_weight_path(args)
    state_dict = torch.load(weight_path, map_location=args.device)
    model.load_state_dict({k: v for k, v in state_dict.items() if 'mask' not in k}, strict=False)
    get_model_params(model, model.config, ignore_patterns=['audio_encoder'])
    return model, tokenizer, model.processor


def init_hf_model(args):
    model_path = resolve_path(args.load_from) if Path(args.load_from).exists() else args.load_from
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True)
    if not hasattr(model, 'processor') or getattr(model, 'processor', None) is None:
        model.audio_encoder, model.processor = MiniMindAudioLM.get_audio_model(str(resolve_path(args.audio_model_path)))
    get_model_params(model, model.config, ignore_patterns=['audio_encoder'])
    return model, tokenizer, model.processor


def init_model(args):
    if args.load_from == 'model':
        model, tokenizer, preprocess = init_native_model(args)
    else:
        model, tokenizer, preprocess = init_hf_model(args)

    if 'cuda' in args.device and args.dtype == 'float16':
        model = model.half()
    elif 'cuda' in args.device and args.dtype == 'bfloat16':
        model = model.to(dtype=torch.bfloat16)

    model = model.eval().to(args.device)
    return model, tokenizer, preprocess


def collect_audio_files(args):
    if args.audio_path:
        audio_path = resolve_path(args.audio_path)
        if not audio_path.exists():
            raise SystemExit(f'Audio file not found: {audio_path}')
        return [audio_path]

    audio_dir = resolve_path(args.audio_dir)
    if not audio_dir.exists():
        raise SystemExit(f'Audio directory not found: {audio_dir}')

    files = sorted(audio_dir.rglob('*.wav'))
    if args.limit > 0:
        files = files[:args.limit]
    if not files:
        raise SystemExit(f'No wav files found under: {audio_dir}')
    return files


def build_messages(model, prompt):
    audio_tokens = model.config.audio_special_token * model.config.audio_token_len
    return [{'role': 'user', 'content': prompt.replace('<audio>', audio_tokens)}]


def main():
    parser = argparse.ArgumentParser(description='MiniMind-AudioLM Eval')
    parser.add_argument('--load_from', default='model', type=str, help='模型加载路径（model=原生torch权重，其他路径=transformers格式）')
    parser.add_argument('--tokenizer_path', default='model', type=str, help='tokenizer路径')
    parser.add_argument('--audio_model_path', default='model/audio_model', type=str, help='Whisper模型路径')
    parser.add_argument('--save_dir', default='out', type=str, help='模型权重目录')
    parser.add_argument('--weight', default='sft_audiolm', type=str, help='权重名称前缀')
    parser.add_argument('--weight_path', default='', type=str, help='直接指定权重文件路径，优先级高于 save_dir/weight/hidden_size 规则')
    parser.add_argument('--hidden_size', default=768, type=int, help='隐藏层维度')
    parser.add_argument('--num_hidden_layers', default=8, type=int, help='隐藏层数量')
    parser.add_argument('--use_moe', default=0, type=int, choices=[0, 1], help='是否使用MoE架构（0=否，1=是）')
    parser.add_argument('--audio_path', default='', type=str, help='单个待测试 wav 文件路径')
    parser.add_argument('--audio_dir', default='dataset/AISHELL-1/data_aishell/test', type=str, help='测试音频目录，会递归查找 wav 文件')
    parser.add_argument('--prompt', default='请转写这段音频内容：\n\n<audio>', type=str, help='用户提示词，使用 <audio> 作为音频占位符')
    parser.add_argument('--max_new_tokens', default=256, type=int, help='最大生成长度')
    parser.add_argument('--temperature', default=0.2, type=float, help='生成温度')
    parser.add_argument('--top_p', default=0.9, type=float, help='nucleus采样阈值')
    parser.add_argument('--show_speed', default=1, type=int, help='显示decode速度（tokens/s）')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu', type=str, help='运行设备')
    parser.add_argument('--dtype', default='float16' if torch.cuda.is_available() else 'float32', choices=['float32', 'float16', 'bfloat16'], help='推理精度')
    parser.add_argument('--open_thinking', default=0, type=int, help='是否开启自适应思考（0=否，1=是）')
    parser.add_argument('--limit', default=0, type=int, help='最多测试多少条音频，0表示不限制')
    args = parser.parse_args()

    model, tokenizer, preprocess = init_model(args)
    streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    audio_files = collect_audio_files(args)

    for audio_file in audio_files:
        setup_seed(random.randint(1, 31415926))
        audio, original_sample_rate = load_wav(audio_file)
        audio, sample_rate, was_resampled = resample_audio(audio, original_sample_rate, TARGET_SAMPLING_RATE)
        audio_encoder = getattr(model, 'audio_encoder', None)
        audio_dtype = next(audio_encoder.parameters()).dtype if audio_encoder is not None else next(model.parameters()).dtype
        input_features = {
            k: v.to(args.device, dtype=audio_dtype)
            for k, v in MiniMindAudioLM.audio2tensor(audio, preprocess, sampling_rate=sample_rate).items()
        }

        messages = build_messages(model, args.prompt)
        inputs_text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            open_thinking=bool(args.open_thinking),
        )
        inputs = tokenizer(inputs_text, return_tensors='pt', truncation=True).to(args.device)

        print(f'[音频]: {audio_file}')
        if was_resampled:
            print(f'[采样率]: 原始 {original_sample_rate} Hz -> 重采样到 {sample_rate} Hz')
        else:
            print(f'[采样率]: {sample_rate} Hz，无需重采样')
        print(f'Prompt: {repr(args.prompt)}')
        print('Output: ', end='')
        st = time.time()
        generated_ids = model.generate(
            inputs=inputs['input_ids'],
            attention_mask=inputs['attention_mask'],
            max_new_tokens=args.max_new_tokens,
            do_sample=args.temperature > 0,
            streamer=streamer,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            top_p=args.top_p,
            temperature=max(args.temperature, 1e-5),
            input_features=input_features,
        )
        gen_tokens = len(generated_ids[0]) - len(inputs['input_ids'][0])
        if args.show_speed:
            print(f'\n[Speed]: {gen_tokens / max(time.time() - st, 1e-6):.2f} tokens/s\n')
        else:
            print('\n')


if __name__ == '__main__':
    main()
