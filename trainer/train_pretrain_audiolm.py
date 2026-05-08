import os
import sys

__package__ = "trainer"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
import io
import json
import time
import warnings
import wave
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
import torch.distributed as dist
from torch import optim
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from transformers import AutoTokenizer

from dataset.lm_dataset import pre_processing_chat, post_processing_chat
from model.model_audiolm import AudioLMConfig, MiniMindAudioLM
from trainer.trainer_utils import (
    Logger,
    SkipBatchSampler,
    get_lr,
    get_model_params,
    init_distributed_mode,
    is_main_process,
    setup_seed,
)

warnings.filterwarnings('ignore')
# 项目根目录，用来把相对路径统一解析成仓库内的绝对路径。
PROJECT_ROOT = Path(__file__).resolve().parent.parent


def resolve_path(path_str: str) -> Path:
    # 所有命令行传入的路径都允许写成相对路径，这里统一解析，减少脚本对 cwd 的依赖。
    path = Path(path_str)
    return path if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def load_wav(audio_source):
    """使用标准库 wave 读取 wav，避免额外音频依赖。"""
    if isinstance(audio_source, (str, os.PathLike, Path)):
        handle = wave.open(str(audio_source), 'rb')
    else:
        handle = wave.open(io.BytesIO(audio_source), 'rb')

    with handle as wav_file:
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
        raise ValueError(f"Unsupported wav sample width: {sample_width}")

    if num_channels > 1:
        audio = audio.reshape(-1, num_channels).mean(axis=1)

    return audio, sample_rate


class AudioLMDataset(Dataset):
    def __init__(
        self,
        parquet_path,
        tokenizer,
        preprocess=None,
        max_length=512,
        audio_special_token='<|audio_pad|>',
        audio_token_len=64,
    ):
        super().__init__()
        # 直接把 parquet 全量读入成 Arrow Table，适合当前这种以顺序训练为主的场景。
        self.table = pa.Table.from_batches(pq.ParquetFile(parquet_path).iter_batches())
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.preprocess = preprocess
        # 训练时会把 <audio> 替换成连续的音频占位 token，数量等于 audio_token_len。
        self.audio_special_token = audio_special_token * audio_token_len
        # 用 assistant 段的起止标记来构造 labels，只监督 assistant 的输出。
        self.bos_id = tokenizer(f'{tokenizer.bos_token}assistant\n', add_special_tokens=False).input_ids
        self.eos_id = tokenizer(f'{tokenizer.eos_token}\n', add_special_tokens=False).input_ids
        # 兼容几种可能的音频字段命名，便于后续不同转换脚本产出的 parquet 复用。
        self.audio_columns = [
            name for name in ['audio_path', 'audio_paths', 'wav_path', 'audio_bytes']
            if name in self.table.column_names
        ]
        if not self.audio_columns:
            raise ValueError(
                f"No supported audio column found in parquet. Available columns: {self.table.column_names}"
            )
        if self.preprocess is None:
            raise ValueError('Audio preprocess/processor is required for AudioLMDataset.')

    def __len__(self):
        return len(self.table)

    def ensure_audio_placeholder(self, conversations):
        # 如果样本里已经显式写了 <audio> / <speech>，就保持原样。
        if any('<audio>' in turn.get('content', '') or '<speech>' in turn.get('content', '') for turn in conversations):
            return conversations

        # 否则自动把音频占位插到第一条 user 消息里，保证 prompt 和音频模态能对齐。
        for turn in conversations:
            if turn.get('role') == 'user':
                content = turn.get('content', '')
                turn['content'] = f"{content.rstrip()}\n<audio>" if content else '<audio>'
                return conversations

        if conversations:
            content = conversations[0].get('content', '')
            conversations[0]['content'] = f"{content.rstrip()}\n<audio>" if content else '<audio>'
        return conversations

    def create_chat_prompt(self, conversations):
        # 这里沿用 tokenizer 自带的 chat template，把多轮对话拼成模型最终看到的训练文本。
        messages = []
        for turn in conversations:
            content = turn.get('content', '')
            if turn.get('role') != 'system':
                # 文本里的 <audio> / <speech> 只是可读占位，真正喂给模型的是重复的 audio token。
                content = content.replace('<audio>', self.audio_special_token)
                content = content.replace('<speech>', self.audio_special_token)
            messages.append({"role": turn['role'], "content": content})
        tools = conversations[0]["functions"] if (conversations and conversations[0]["role"] == "system" and conversations[0].get("functions")) else None
        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
            tools=tools
        )

    def generate_labels(self, input_ids):
        # 训练标签只覆盖 assistant 回复区间，用户输入和系统提示全部置为 -100。
        labels = [-100] * len(input_ids)
        i = 0
        while i < len(input_ids):
            if input_ids[i:i + len(self.bos_id)] == self.bos_id:
                start = i + len(self.bos_id)
                end = start
                while end < len(input_ids):
                    if input_ids[end:end + len(self.eos_id)] == self.eos_id:
                        break
                    end += 1
                for j in range(start, min(end + len(self.eos_id), self.max_length)):
                    labels[j] = input_ids[j]
                i = end + len(self.eos_id) if end < len(input_ids) else len(input_ids)
            else:
                i += 1
        return labels

    def get_audio_entries(self, index: int):
        # 每条样本支持单段音频或多段音频；返回统一的 [(来源类型, 内容), ...] 结构。
        for column in self.audio_columns:
            value = self.table[column][index].as_py()
            if value is None:
                continue
            if column == 'audio_bytes':
                return [('bytes', item) for item in (value if isinstance(value, list) else [value])]
            return [('path', item) for item in (value if isinstance(value, list) else [value])]
        raise ValueError(f'No audio payload found for sample index {index}.')

    def __getitem__(self, index: int):
        # conversations 约定为 JSON 字符串；如果上游已经存成对象，也兼容直接使用。
        conversations = self.table['conversations'][index].as_py()
        conversations = json.loads(conversations) if isinstance(conversations, str) else conversations
        conversations = pre_processing_chat(conversations)
        conversations = self.ensure_audio_placeholder(conversations)

        # 生成模型输入文本，并做长度截断与 pad。
        prompt = self.create_chat_prompt(conversations)
        prompt = post_processing_chat(prompt)
        input_ids = self.tokenizer(prompt, add_special_tokens=False).input_ids[:self.max_length]
        input_ids += [self.tokenizer.pad_token_id] * (self.max_length - len(input_ids))
        labels = self.generate_labels(input_ids)

        # 每段音频都会被转成 Whisper 所需的 input_features。
        audio_inputs_list = []
        for _, payload in self.get_audio_entries(index):
            audio, sample_rate = load_wav(payload)
            audio_inputs_list.append(MiniMindAudioLM.audio2tensor(audio, self.preprocess, sampling_rate=sample_rate))

        # 如果一条样本有多段音频，这里在样本内先拼接成 [num_audios, ...]，后续 collate 再堆成 batch。
        audio_data = {
            key: torch.cat([item[key] for item in audio_inputs_list], dim=0)
            for key in audio_inputs_list[0].keys()
        }
        return torch.tensor(input_ids, dtype=torch.long), torch.tensor(labels, dtype=torch.long), audio_data


def audiolm_collate_fn(batch):
    # 把样本级的音频特征堆成 batch；dict 分支对应 WhisperProcessor 的返回结构。
    input_ids = torch.stack([b[0] for b in batch])
    labels = torch.stack([b[1] for b in batch])
    audio_data = [b[2] for b in batch]
    if hasattr(audio_data[0], 'keys'):
        input_features = {k: torch.stack([d[k] for d in audio_data]) for k in audio_data[0].keys()}
    else:
        input_features = torch.stack(audio_data)
    return input_ids, labels, input_features


def init_audiolm_model(
    audiolm_config,
    from_weight='llm',
    tokenizer_path='model',
    audio_model_path='model/audio_model',
    save_dir='out',
    device='cuda',
    freeze_llm=0,
):
    # tokenizer 仍然使用项目里的主 tokenizer，音频分支只额外引入了 audio 特殊 token。
    tokenizer = AutoTokenizer.from_pretrained(resolve_path(tokenizer_path))
    model = MiniMindAudioLM(audiolm_config, audio_model_path=str(resolve_path(audio_model_path)))

    # from_weight 一般用于从已有 LLM / AudioLM 权重继续训练。
    if from_weight != 'none':
        moe_suffix = '_moe' if audiolm_config.use_moe else ''
        weight_path = resolve_path(save_dir) / f'{from_weight}_{audiolm_config.hidden_size}{moe_suffix}.pth'
        weights = torch.load(weight_path, map_location=device)
        model.load_state_dict(weights, strict=False)

    # 默认先冻结全部参数，只给 audio_proj 开梯度，后面再根据 freeze_llm 策略选择性解冻。
    for name, param in model.named_parameters():
        if 'audio_proj' not in name:
            param.requires_grad = False

    if freeze_llm == 0:
        # 全量训练 LLM + audio_proj，只保留 Whisper encoder 冻结。
        for name, param in model.named_parameters():
            if 'audio_encoder' not in name:
                param.requires_grad = True
    elif freeze_llm == 1:
        # 轻量微调策略：除了 audio_proj，只解冻 LLM 第 0 层，降低训练开销。
        for name, param in model.model.named_parameters():
            if 'layers.0.' in name:
                param.requires_grad = True
    elif freeze_llm == 2:
        # 最保守策略：只训练 audio_proj。
        pass

    get_model_params(model, audiolm_config, ignore_patterns=['audio_encoder'])
    Logger(f'Trainable Params: {sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6:.3f}M')
    return model.to(device), tokenizer, model.processor


def audiolm_checkpoint(audiolm_config, weight='pretrain_audiolm', model=None, optimizer=None, epoch=0, step=0, wandb=None, save_dir='checkpoints', **kwargs):
    # 保存两份内容：
    # 1. 纯模型权重（便于推理/继续训练）
    # 2. resume 状态（包含优化器、scaler、step 等）
    save_dir = resolve_path(save_dir)
    os.makedirs(save_dir, exist_ok=True)
    moe_path = '_moe' if audiolm_config.use_moe else ''
    ckp_path = save_dir / f'{weight}_{audiolm_config.hidden_size}{moe_path}.pth'
    resume_path = save_dir / f'{weight}_{audiolm_config.hidden_size}{moe_path}_resume.pth'

    if model is not None:
        raw_model = model.module if isinstance(model, DistributedDataParallel) else model
        raw_model = getattr(raw_model, '_orig_mod', raw_model)
        state_dict = raw_model.state_dict()
        # Whisper encoder 作为固定特征提取器，不重复保存，减小 checkpoint 体积。
        clean_state_dict = {k: v for k, v in state_dict.items() if not k.startswith('audio_encoder.')}
        ckp_tmp = str(ckp_path) + '.tmp'
        torch.save({k: v.half().cpu() for k, v in clean_state_dict.items()}, ckp_tmp)
        os.replace(ckp_tmp, ckp_path)

        wandb_id = None
        if wandb:
            if hasattr(wandb, 'get_run'):
                run = wandb.get_run()
                wandb_id = getattr(run, 'id', None) if run else None
            else:
                wandb_id = getattr(wandb, 'id', None)

        resume_data = {
            'model': state_dict,
            'optimizer': optimizer.state_dict(),
            'epoch': epoch,
            'step': step,
            'world_size': dist.get_world_size() if dist.is_initialized() else 1,
            'wandb_id': wandb_id,
        }
        for key, value in kwargs.items():
            if value is None:
                continue
            if hasattr(value, 'state_dict'):
                raw_value = value.module if isinstance(value, DistributedDataParallel) else value
                raw_value = getattr(raw_value, '_orig_mod', raw_value)
                resume_data[key] = raw_value.state_dict()
            else:
                resume_data[key] = value

        resume_tmp = str(resume_path) + '.tmp'
        torch.save(resume_data, resume_tmp)
        os.replace(resume_tmp, resume_path)
        del state_dict, clean_state_dict, resume_data
        torch.cuda.empty_cache()
    else:
        if resume_path.exists():
            ckp_data = torch.load(resume_path, map_location='cpu')
            saved_ws = ckp_data.get('world_size', 1)
            current_ws = dist.get_world_size() if dist.is_initialized() else 1
            # GPU 数量变化时，尽量把 step 映射到新的 world size，减少续训错位。
            if saved_ws != current_ws:
                ckp_data['step'] = ckp_data['step'] * saved_ws // current_ws
                Logger(f'GPU数量变化({saved_ws}→{current_ws})，step已自动转换为{ckp_data["step"]}')
            return ckp_data
        return None


def train_epoch(epoch, loader, iters, start_step=0, wandb=None):
    start_time = time.time()
    last_step = start_step
    for step, (input_ids, labels, input_features) in enumerate(loader, start=start_step + 1):
        input_ids = input_ids.to(args.device)
        labels = labels.to(args.device)
        # WhisperProcessor 返回 dict；这里统一搬到训练设备。
        input_features = {k: v.to(args.device) for k, v in input_features.items()} if isinstance(input_features, dict) else input_features.to(args.device)
        last_step = step
        lr = get_lr(epoch * iters + step, args.epochs * iters, args.learning_rate)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        with autocast_ctx:
            # AudioLM 的前向入口已经统一使用 input_features，不再走旧的 pixel_values 接口。
            res = model(input_ids, labels=labels, input_features=input_features)
            loss = res.loss + res.aux_loss
            loss = loss / args.accumulation_steps

        scaler.scale(loss).backward()

        if step % args.accumulation_steps == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        if step % args.log_interval == 0 or step == iters:
            spend_time = time.time() - start_time
            current_loss = loss.item() * args.accumulation_steps
            current_aux_loss = res.aux_loss.item() if res.aux_loss is not None else 0.0
            current_logits_loss = current_loss - current_aux_loss
            current_lr = optimizer.param_groups[-1]['lr']
            eta_min = spend_time / max(step - start_step, 1) * (iters - step) // 60
            Logger(
                f'Epoch:[{epoch + 1}/{args.epochs}]({step}/{iters}), '
                f'loss: {current_loss:.4f}, logits_loss: {current_logits_loss:.4f}, '
                f'aux_loss: {current_aux_loss:.4f}, lr: {current_lr:.8f}, epoch_time: {eta_min:.1f}min'
            )
            if wandb:
                wandb.log({
                    'loss': current_loss,
                    'logits_loss': current_logits_loss,
                    'aux_loss': current_aux_loss,
                    'learning_rate': current_lr,
                    'epoch_time': eta_min,
                })

        if (step % args.save_interval == 0 or step == iters) and is_main_process():
            model.eval()
            audiolm_checkpoint(
                audiolm_config,
                weight=args.save_weight,
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                step=step,
                wandb=wandb,
                save_dir=args.checkpoint_dir,
                scaler=scaler,
            )
            model.train()

        del input_ids, labels, input_features, res, loss

    if last_step > start_step and last_step % args.accumulation_steps != 0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MiniMind-AudioLM Pretrain")
    parser.add_argument('--save_dir', type=str, default='out', help='模型权重保存目录')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints', help='断点续训状态保存目录')
    parser.add_argument('--save_weight', default='pretrain_audiolm', type=str, help='保存权重的前缀名')
    parser.add_argument('--epochs', type=int, default=3, help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=16, help='batch size')
    parser.add_argument('--learning_rate', type=float, default=4e-4, help='初始学习率')
    parser.add_argument('--device', type=str, default='cuda:0' if torch.cuda.is_available() else 'cpu', help='训练设备')
    parser.add_argument('--dtype', type=str, default='bfloat16', help='混合精度类型')
    parser.add_argument('--num_workers', type=int, default=8, help='数据加载线程数')
    parser.add_argument('--accumulation_steps', type=int, default=1, help='梯度累积步数')
    parser.add_argument('--grad_clip', type=float, default=1.0, help='梯度裁剪阈值')
    parser.add_argument('--log_interval', type=int, default=100, help='日志打印间隔')
    parser.add_argument('--save_interval', type=int, default=1000, help='模型保存间隔')
    parser.add_argument('--hidden_size', default=768, type=int, help='隐藏层维度')
    parser.add_argument('--num_hidden_layers', default=8, type=int, help='隐藏层数量')
    parser.add_argument('--max_seq_len', default=360, type=int, help='训练的最大截断长度')
    parser.add_argument('--use_moe', default=0, type=int, choices=[0, 1], help='是否使用MoE架构（0=否，1=是）')
    parser.add_argument('--data_path', type=str, default='dataset/aishell1_train_pretrain.parquet', help='训练数据路径')
    parser.add_argument('--tokenizer_path', type=str, default='model', help='tokenizer路径')
    parser.add_argument('--audio_model_path', type=str, default='model/audio_model', help='Whisper模型路径')
    parser.add_argument('--from_weight', default='llm', type=str, help='基于哪个权重训练，为none则不基于任何权重训练')
    parser.add_argument('--from_resume', default=0, type=int, choices=[0, 1], help='是否自动检测并续训（0=否，1=是）')
    parser.add_argument('--freeze_llm', default=1, type=int, choices=[0, 1, 2], help='冻结策略（0=完全可训练，1=冻结+解冻第0层，2=完全冻结仅训练proj）')
    parser.add_argument('--use_compile', default=0, type=int, choices=[0, 1], help='是否使用torch.compile加速（0=否，1=是）')
    parser.add_argument('--use_wandb', action='store_true', help='是否使用wandb')
    parser.add_argument('--wandb_project', type=str, default='MiniMind-AudioLM-Pretrain', help='wandb项目名')
    args = parser.parse_args()

    # 标准 DDP 初始化逻辑；单卡时 local_rank 为 0。
    local_rank = init_distributed_mode()
    if dist.is_initialized():
        args.device = f'cuda:{local_rank}'
    setup_seed(42 + (dist.get_rank() if dist.is_initialized() else 0))

    resolve_path(args.save_dir).mkdir(parents=True, exist_ok=True)
    audiolm_config = AudioLMConfig(
        hidden_size=args.hidden_size,
        num_hidden_layers=args.num_hidden_layers,
        max_seq_len=args.max_seq_len,
        use_moe=bool(args.use_moe),
    )
    ckp_data = audiolm_checkpoint(audiolm_config, weight=args.save_weight, save_dir=args.checkpoint_dir) if args.from_resume == 1 else None

    device_type = 'cuda' if 'cuda' in args.device else 'cpu'
    dtype = torch.bfloat16 if args.dtype == 'bfloat16' else torch.float16
    autocast_ctx = nullcontext() if device_type == 'cpu' else torch.cuda.amp.autocast(dtype=dtype)

    wandb = None
    if args.use_wandb and is_main_process():
        import swanlab as wandb
        wandb_id = ckp_data.get('wandb_id') if ckp_data else None
        resume = 'must' if wandb_id else None
        wandb_run_name = f'MiniMind-AudioLM-Pretrain-Epoch-{args.epochs}-BatchSize-{args.batch_size}-LearningRate-{args.learning_rate}'
        wandb.init(project=args.wandb_project, name=wandb_run_name, id=wandb_id, resume=resume)

    # 初始化模型、tokenizer、Whisper processor。
    model, tokenizer, preprocess = init_audiolm_model(
        audiolm_config,
        from_weight=args.from_weight,
        tokenizer_path=args.tokenizer_path,
        audio_model_path=args.audio_model_path,
        save_dir=args.save_dir,
        device=args.device,
        freeze_llm=args.freeze_llm,
    )
    train_ds = AudioLMDataset(
        resolve_path(args.data_path),
        tokenizer,
        preprocess=preprocess,
        audio_special_token=audiolm_config.audio_special_token,
        audio_token_len=audiolm_config.audio_token_len,
        max_length=args.max_seq_len,
    )
    train_sampler = DistributedSampler(train_ds) if dist.is_initialized() else None
    scaler = torch.cuda.amp.GradScaler(enabled=(args.dtype == 'float16'))
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.learning_rate)

    # 恢复断点时，把模型参数、优化器状态、GradScaler 一起接回来。
    start_epoch, start_step = 0, 0
    if ckp_data:
        model.load_state_dict(ckp_data['model'], strict=False)
        optimizer.load_state_dict(ckp_data['optimizer'])
        scaler.load_state_dict(ckp_data['scaler'])
        start_epoch = ckp_data['epoch']
        start_step = ckp_data.get('step', 0)

    if args.use_compile == 1:
        model = torch.compile(model)
        Logger('torch.compile enabled')
    if dist.is_initialized():
        # 这些 buffer 不参与梯度同步，显式忽略可以减少 DDP 的干扰。
        model._ddp_params_and_buffers_to_ignore = {'freqs_cos', 'freqs_sin'}
        model = DistributedDataParallel(model, device_ids=[local_rank])

    for epoch in range(start_epoch, args.epochs):
        train_sampler and train_sampler.set_epoch(epoch)
        setup_seed(42 + epoch)
        indices = torch.randperm(len(train_ds)).tolist()
        # 续训时会跳过当前 epoch 已经跑过的 step。
        skip = start_step if (epoch == start_epoch and start_step > 0) else 0
        batch_sampler = SkipBatchSampler(train_sampler or indices, args.batch_size, skip)
        loader = DataLoader(
            train_ds,
            batch_sampler=batch_sampler,
            num_workers=args.num_workers,
            pin_memory=True,
            collate_fn=audiolm_collate_fn,
        )
        if skip > 0:
            Logger(f'Epoch [{epoch + 1}/{args.epochs}]: 跳过前{start_step}个step，从step {start_step + 1}开始')
            train_epoch(epoch, loader, len(loader) + skip, start_step, wandb)
        else:
            train_epoch(epoch, loader, len(loader), 0, wandb)

    if dist.is_initialized():
        dist.destroy_process_group()
