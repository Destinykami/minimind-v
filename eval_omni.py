import argparse
import random
import time
import warnings
import wave
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from scipy.signal import resample_poly
from transformers import AutoTokenizer, TextStreamer

from model.model_omni import MiniMindOmni, OmniConfig
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


def init_model(args):
    tokenizer = AutoTokenizer.from_pretrained(resolve_path(args.tokenizer_path))
    model = MiniMindOmni(
        OmniConfig(
            hidden_size=args.hidden_size,
            num_hidden_layers=args.num_hidden_layers,
            use_moe=bool(args.use_moe),
        ),
        vision_model_path=str(resolve_path(args.vision_model_path)),
        audio_model_path=str(resolve_path(args.audio_model_path)),
    )
    state_dict = torch.load(get_weight_path(args), map_location=args.device)
    model.load_state_dict({k: v for k, v in state_dict.items() if 'mask' not in k}, strict=False)
    get_model_params(model, model.config, ignore_patterns=['vision_encoder', 'audio_encoder'])

    if 'cuda' in args.device and args.dtype == 'float16':
        model = model.half()
    elif 'cuda' in args.device and args.dtype == 'bfloat16':
        model = model.to(dtype=torch.bfloat16)
    return model.eval().to(args.device), tokenizer, model.processor


def build_prompt(model, raw_prompt, has_image, has_audio):
    prompt = raw_prompt
    if has_image:
        prompt = prompt.replace('<image>', model.config.image_special_token * model.config.image_token_len)
    else:
        prompt = prompt.replace('<image>', '').strip()
    if has_audio:
        prompt = prompt.replace('<audio>', model.config.audio_special_token * model.config.audio_token_len)
    else:
        prompt = prompt.replace('<audio>', '').strip()
    return prompt


def collect_examples(args):
    image_path = resolve_path(args.image_path) if args.image_path else None
    audio_path = resolve_path(args.audio_path) if args.audio_path else None
    image_dir = resolve_path(args.image_dir) if args.image_dir else None
    audio_dir = resolve_path(args.audio_dir) if args.audio_dir else None

    if image_path or audio_path:
        return [{'image': image_path if image_path and image_path.exists() else None, 'audio': audio_path if audio_path and audio_path.exists() else None}]

    image_files = []
    audio_files = []
    if image_dir and image_dir.exists():
        image_files = sorted([path for path in image_dir.rglob('*') if path.suffix.lower() in {'.png', '.jpg', '.jpeg', '.bmp'}])
    if audio_dir and audio_dir.exists():
        audio_files = sorted(audio_dir.rglob('*.wav'))

    if image_files and audio_files:
        audio_by_stem = {path.stem: path for path in audio_files}
        pairs = [{'image': image, 'audio': audio_by_stem.get(image.stem)} for image in image_files if image.stem in audio_by_stem]
        if not pairs:
            raise SystemExit('image_dir 和 audio_dir 同时提供时，没有找到同名 stem 的配对样本。')
        return pairs[:args.limit] if args.limit > 0 else pairs

    if image_files:
        files = image_files[:args.limit] if args.limit > 0 else image_files
        return [{'image': image, 'audio': None} for image in files]
    if audio_files:
        files = audio_files[:args.limit] if args.limit > 0 else audio_files
        return [{'image': None, 'audio': audio} for audio in files]

    raise SystemExit('请提供 --image_path / --audio_path，或至少一个目录参数。')


def main():
    parser = argparse.ArgumentParser(description='MiniMind-Omni Eval')
    parser.add_argument('--tokenizer_path', default='model', type=str, help='tokenizer路径')
    parser.add_argument('--vision_model_path', default='model/vision_model/siglip2-base-p16-ve', type=str, help='视觉模型路径')
    parser.add_argument('--audio_model_path', default='model/audio_model', type=str, help='音频模型路径')
    parser.add_argument('--save_dir', default='out', type=str, help='模型权重目录')
    parser.add_argument('--weight', default='sft_omni', type=str, help='权重名称前缀')
    parser.add_argument('--weight_path', default='', type=str, help='直接指定权重文件路径')
    parser.add_argument('--hidden_size', default=768, type=int, help='隐藏层维度')
    parser.add_argument('--num_hidden_layers', default=8, type=int, help='隐藏层数量')
    parser.add_argument('--use_moe', default=0, type=int, choices=[0, 1], help='是否使用MoE架构（0=否，1=是）')
    parser.add_argument('--image_path', default='', type=str, help='单张待测试图像路径')
    parser.add_argument('--audio_path', default='', type=str, help='单个待测试音频路径')
    parser.add_argument('--image_dir', default='dataset/ocr_eval_images', type=str, help='图像目录')
    parser.add_argument('--audio_dir', default='', type=str, help='音频目录')
    parser.add_argument('--prompt', default='请结合给定的输入模态回答问题：\n\n<image>\n<audio>', type=str, help='提示词，可包含 <image> 和 <audio> 占位符')
    parser.add_argument('--max_new_tokens', default=256, type=int, help='最大生成长度')
    parser.add_argument('--temperature', default=0.2, type=float, help='生成温度')
    parser.add_argument('--top_p', default=0.9, type=float, help='nucleus采样阈值')
    parser.add_argument('--show_speed', default=1, type=int, help='显示decode速度（tokens/s）')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu', type=str, help='运行设备')
    parser.add_argument('--dtype', default='float16' if torch.cuda.is_available() else 'float32', choices=['float32', 'float16', 'bfloat16'], help='推理精度')
    parser.add_argument('--open_thinking', default=0, type=int, help='是否开启自适应思考（0=否，1=是）')
    parser.add_argument('--limit', default=0, type=int, help='目录模式下最多测试多少条样本')
    args = parser.parse_args()

    model, tokenizer, processors = init_model(args)
    streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    examples = collect_examples(args)

    vision_dtype = next(model.vision_encoder.parameters()).dtype if model.vision_encoder is not None else next(model.parameters()).dtype
    audio_dtype = next(model.audio_encoder.parameters()).dtype if model.audio_encoder is not None else next(model.parameters()).dtype

    for example in examples:
        setup_seed(random.randint(1, 31415926))
        pixel_values = None
        input_features = None

        if example['image'] is not None:
            image = Image.open(example['image']).convert('RGB')
            pixel_values = {
                key: (value.to(args.device, dtype=vision_dtype) if torch.is_floating_point(value) else value.to(args.device))
                for key, value in MiniMindOmni.image2tensor(image, processors['image']).items()
            }

        sample_rate_info = None
        if example['audio'] is not None:
            audio, original_sample_rate = load_wav(example['audio'])
            audio, sample_rate, was_resampled = resample_audio(audio, original_sample_rate, TARGET_SAMPLING_RATE)
            input_features = {
                key: value.to(args.device, dtype=audio_dtype)
                for key, value in MiniMindOmni.audio2tensor(audio, processors['audio'], sampling_rate=sample_rate).items()
            }
            sample_rate_info = (original_sample_rate, sample_rate, was_resampled)

        prompt = build_prompt(model, args.prompt, example['image'] is not None, example['audio'] is not None)
        messages = [{'role': 'user', 'content': prompt}]
        inputs_text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            open_thinking=bool(args.open_thinking),
        )
        inputs = tokenizer(inputs_text, return_tensors='pt', truncation=True).to(args.device)

        print(f'[图像]: {example["image"]}' if example['image'] is not None else '[图像]: None')
        print(f'[音频]: {example["audio"]}' if example['audio'] is not None else '[音频]: None')
        if sample_rate_info is not None:
            original_sr, sample_rate, was_resampled = sample_rate_info
            if was_resampled:
                print(f'[采样率]: 原始 {original_sr} Hz -> 重采样到 {sample_rate} Hz')
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
            pixel_values=pixel_values,
            input_features=input_features,
        )
        gen_tokens = len(generated_ids[0]) - len(inputs['input_ids'][0])
        if args.show_speed:
            print(f'\n[Speed]: {gen_tokens / max(time.time() - st, 1e-6):.2f} tokens/s\n')
        else:
            print('\n')


if __name__ == '__main__':
    main()
