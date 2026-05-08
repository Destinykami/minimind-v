import argparse
import os
import sys
import time
import warnings
from contextlib import nullcontext
from pathlib import Path

import torch
import torch.distributed as dist
from torch import optim
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler
from transformers import AutoTokenizer

__package__ = "trainer"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from dataset.omni_lm_dataset import OmniLMDataset, omni_collate_fn
from model.model_omni import MiniMindOmni, OmniConfig
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
PROJECT_ROOT = Path(__file__).resolve().parent.parent


def resolve_path(path_str: str) -> Path:
    path = Path(path_str)
    return path if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def load_partial_state_dict(model, weight_path, allowed_prefixes, device):
    if not weight_path.exists():
        raise FileNotFoundError(f'Weight not found: {weight_path}')
    state_dict = torch.load(weight_path, map_location=device)
    filtered_state = {
        key: value for key, value in state_dict.items()
        if any(key.startswith(prefix) for prefix in allowed_prefixes)
    }
    missing, unexpected = model.load_state_dict(filtered_state, strict=False)
    Logger(
        f'Loaded {len(filtered_state)} tensors from {weight_path.name}; '
        f'missing={len(missing)}, unexpected={len(unexpected)}'
    )


def init_omni_model(
    omni_config,
    tokenizer_path='model',
    vision_model_path='model/vision_model/siglip2-base-p16-ve',
    audio_model_path='model/audio_model',
    save_dir='out',
    device='cuda',
    from_weight='none',
    from_vlm_weight='none',
    from_audiolm_weight='none',
    freeze_llm=1,
):
    tokenizer = AutoTokenizer.from_pretrained(resolve_path(tokenizer_path))
    model = MiniMindOmni(
        omni_config,
        vision_model_path=str(resolve_path(vision_model_path)),
        audio_model_path=str(resolve_path(audio_model_path)),
    )

    save_dir = resolve_path(save_dir)
    moe_suffix = '_moe' if omni_config.use_moe else ''
    backbone_loaded = False

    if from_weight != 'none':
        full_weight_path = save_dir / f'{from_weight}_{omni_config.hidden_size}{moe_suffix}.pth'
        state_dict = torch.load(full_weight_path, map_location=device)
        model.load_state_dict(state_dict, strict=False)
        Logger(f'Loaded omni checkpoint: {full_weight_path.name}')
        backbone_loaded = True

    if from_vlm_weight != 'none':
        vlm_path = save_dir / f'{from_vlm_weight}_{omni_config.hidden_size}{moe_suffix}.pth'
        load_partial_state_dict(model, vlm_path, ['model.', 'lm_head.', 'vision_proj.'], device)
        backbone_loaded = True

    if from_audiolm_weight != 'none':
        prefixes = ['audio_proj.']
        if not backbone_loaded:
            prefixes = ['model.', 'lm_head.', 'audio_proj.']
        audiolm_path = save_dir / f'{from_audiolm_weight}_{omni_config.hidden_size}{moe_suffix}.pth'
        load_partial_state_dict(model, audiolm_path, prefixes, device)

    # 先全部冻结，只开放两个 projector。
    for name, param in model.named_parameters():
        param.requires_grad = ('vision_proj' in name) or ('audio_proj' in name)

    if freeze_llm == 0:
        for name, param in model.named_parameters():
            if 'vision_encoder' not in name and 'audio_encoder' not in name:
                param.requires_grad = True
    elif freeze_llm == 1:
        for name, param in model.model.named_parameters():
            if 'layers.0.' in name:
                param.requires_grad = True
    elif freeze_llm == 2:
        pass

    get_model_params(model, omni_config, ignore_patterns=['vision_encoder', 'audio_encoder'])
    Logger(f'Trainable Params: {sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6:.3f}M')
    return model.to(device), tokenizer, model.processor


def omni_checkpoint(omni_config, weight='pretrain_omni', model=None, optimizer=None, epoch=0, step=0, wandb=None, save_dir='checkpoints', **kwargs):
    save_dir = resolve_path(save_dir)
    os.makedirs(save_dir, exist_ok=True)
    moe_path = '_moe' if omni_config.use_moe else ''
    ckp_path = save_dir / f'{weight}_{omni_config.hidden_size}{moe_path}.pth'
    resume_path = save_dir / f'{weight}_{omni_config.hidden_size}{moe_path}_resume.pth'

    if model is not None:
        raw_model = model.module if isinstance(model, DistributedDataParallel) else model
        raw_model = getattr(raw_model, '_orig_mod', raw_model)
        state_dict = raw_model.state_dict()
        clean_state_dict = {
            key: value for key, value in state_dict.items()
            if not key.startswith('vision_encoder.') and not key.startswith('audio_encoder.')
        }
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
            if saved_ws != current_ws:
                ckp_data['step'] = ckp_data['step'] * saved_ws // current_ws
                Logger(f'GPU数量变化({saved_ws}→{current_ws})，step已自动转换为{ckp_data["step"]}')
            return ckp_data
        return None


def move_to_device(modal_batch, device, dtype):
    moved = []
    for item in modal_batch:
        if item is None:
            moved.append(None)
        elif hasattr(item, 'keys'):
            moved_item = {}
            for key, value in item.items():
                # SigLIP2 的 spatial_shapes 等元信息必须保持整数类型，只搬设备不改 dtype。
                moved_item[key] = value.to(device=device, dtype=dtype) if torch.is_floating_point(value) else value.to(device=device)
            moved.append(moved_item)
        else:
            moved.append(item.to(device=device, dtype=dtype) if torch.is_floating_point(item) else item.to(device=device))
    return moved


def train_epoch(epoch, loader, iters, start_step=0, wandb=None):
    start_time = time.time()
    last_step = start_step
    vision_dtype = next(model.vision_encoder.parameters()).dtype if model.vision_encoder is not None else next(model.parameters()).dtype
    audio_dtype = next(model.audio_encoder.parameters()).dtype if model.audio_encoder is not None else next(model.parameters()).dtype

    for step, (input_ids, labels, pixel_values, input_features) in enumerate(loader, start=start_step + 1):
        input_ids = input_ids.to(args.device)
        labels = labels.to(args.device)
        pixel_values = move_to_device(pixel_values, args.device, vision_dtype)
        input_features = move_to_device(input_features, args.device, audio_dtype)
        last_step = step

        lr = get_lr(epoch * iters + step, args.epochs * iters, args.learning_rate)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        with autocast_ctx:
            res = model(input_ids, labels=labels, pixel_values=pixel_values, input_features=input_features)
            loss = (res.loss + res.aux_loss) / args.accumulation_steps

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
            omni_checkpoint(
                omni_config,
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

        del input_ids, labels, pixel_values, input_features, res, loss

    if last_step > start_step and last_step % args.accumulation_steps != 0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='MiniMind-Omni Pretrain')
    parser.add_argument('--save_dir', type=str, default='out', help='模型权重保存目录')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints', help='断点续训状态保存目录')
    parser.add_argument('--save_weight', default='pretrain_omni', type=str, help='保存权重的前缀名')
    parser.add_argument('--epochs', type=int, default=3, help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size')
    parser.add_argument('--learning_rate', type=float, default=2e-4, help='初始学习率')
    parser.add_argument('--device', type=str, default='cuda:0' if torch.cuda.is_available() else 'cpu', help='训练设备')
    parser.add_argument('--dtype', type=str, default='bfloat16', help='混合精度类型')
    parser.add_argument('--num_workers', type=int, default=12, help='数据加载线程数')
    parser.add_argument('--prefetch_factor', type=int, default=4, help='每个 worker 预取多少个 batch，仅在 num_workers>0 时生效')
    parser.add_argument('--accumulation_steps', type=int, default=1, help='梯度累积步数')
    parser.add_argument('--grad_clip', type=float, default=1.0, help='梯度裁剪阈值')
    parser.add_argument('--log_interval', type=int, default=100, help='日志打印间隔')
    parser.add_argument('--save_interval', type=int, default=1000, help='模型保存间隔')
    parser.add_argument('--hidden_size', default=768, type=int, help='隐藏层维度')
    parser.add_argument('--num_hidden_layers', default=8, type=int, help='隐藏层数量')
    parser.add_argument('--max_seq_len', default=768, type=int, help='训练的最大截断长度')
    parser.add_argument('--use_moe', default=0, type=int, choices=[0, 1], help='是否使用MoE架构（0=否，1=是）')
    parser.add_argument('--data_path', type=str, default='dataset/omni_pretrain.parquet', help='训练数据路径')
    parser.add_argument('--tokenizer_path', type=str, default='model', help='tokenizer路径')
    parser.add_argument('--vision_model_path', type=str, default='model/vision_model/siglip2-base-p16-ve', help='视觉模型路径')
    parser.add_argument('--audio_model_path', type=str, default='model/audio_model', help='音频模型路径')
    parser.add_argument('--from_weight', default='none', type=str, help='继续训练已有 omni 权重，为 none 则不加载')
    parser.add_argument('--from_vlm_weight', default='sft_vlm', type=str, help='用于初始化视觉分支/LLM 的 VLM 权重')
    parser.add_argument('--from_audiolm_weight', default='sft_audiolm', type=str, help='用于初始化音频分支的 AudioLM 权重')
    parser.add_argument('--from_resume', default=0, type=int, choices=[0, 1], help='是否自动检测并续训（0=否，1=是）')
    parser.add_argument('--freeze_llm', default=1, type=int, choices=[0, 1, 2], help='冻结策略（0=完全可训练，1=冻结+解冻第0层，2=完全冻结仅训练proj）')
    parser.add_argument('--use_compile', default=1, type=int, choices=[0, 1], help='是否使用torch.compile加速（0=否，1=是）')
    parser.add_argument('--use_wandb', action='store_true', help='是否使用wandb')
    parser.add_argument('--wandb_project', type=str, default='MiniMind-Omni-Pretrain', help='wandb项目名')
    args = parser.parse_args()

    local_rank = init_distributed_mode()
    if dist.is_initialized():
        args.device = f'cuda:{local_rank}'
    setup_seed(42 + (dist.get_rank() if dist.is_initialized() else 0))

    resolve_path(args.save_dir).mkdir(parents=True, exist_ok=True)
    omni_config = OmniConfig(
        hidden_size=args.hidden_size,
        num_hidden_layers=args.num_hidden_layers,
        max_seq_len=args.max_seq_len,
        use_moe=bool(args.use_moe),
    )
    ckp_data = omni_checkpoint(omni_config, weight=args.save_weight, save_dir=args.checkpoint_dir) if args.from_resume == 1 else None

    device_type = 'cuda' if 'cuda' in args.device else 'cpu'
    dtype = torch.bfloat16 if args.dtype == 'bfloat16' else torch.float16
    autocast_ctx = nullcontext() if device_type == 'cpu' else torch.cuda.amp.autocast(dtype=dtype)

    wandb = None
    if args.use_wandb and is_main_process():
        import swanlab as wandb

        wandb_id = ckp_data.get('wandb_id') if ckp_data else None
        resume = 'must' if wandb_id else None
        wandb_run_name = f'MiniMind-Omni-Pretrain-Epoch-{args.epochs}-BatchSize-{args.batch_size}-LearningRate-{args.learning_rate}'
        wandb.init(project=args.wandb_project, name=wandb_run_name, id=wandb_id, resume=resume)

    model, tokenizer, processors = init_omni_model(
        omni_config,
        tokenizer_path=args.tokenizer_path,
        vision_model_path=args.vision_model_path,
        audio_model_path=args.audio_model_path,
        save_dir=args.save_dir,
        device=args.device,
        from_weight=args.from_weight,
        from_vlm_weight=args.from_vlm_weight,
        from_audiolm_weight=args.from_audiolm_weight,
        freeze_llm=args.freeze_llm,
    )
    train_ds = OmniLMDataset(
        resolve_path(args.data_path),
        tokenizer,
        image_processor=processors.get('image'),
        audio_processor=processors.get('audio'),
        image_special_token=omni_config.image_special_token,
        image_token_len=omni_config.image_token_len,
        audio_special_token=omni_config.audio_special_token,
        audio_token_len=omni_config.audio_token_len,
        max_length=args.max_seq_len,
    )
    train_sampler = DistributedSampler(train_ds) if dist.is_initialized() else None
    scaler = torch.cuda.amp.GradScaler(enabled=(args.dtype == 'float16'))
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.learning_rate)

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
        model._ddp_params_and_buffers_to_ignore = {'freqs_cos', 'freqs_sin'}
        model = DistributedDataParallel(model, device_ids=[local_rank])

    for epoch in range(start_epoch, args.epochs):
        train_sampler and train_sampler.set_epoch(epoch)
        setup_seed(42 + epoch)
        indices = torch.randperm(len(train_ds)).tolist()
        skip = start_step if (epoch == start_epoch and start_step > 0) else 0
        batch_sampler = SkipBatchSampler(train_sampler or indices, args.batch_size, skip)
        loader = DataLoader(
            train_ds,
            batch_sampler=batch_sampler,
            num_workers=args.num_workers,
            pin_memory=True,
            persistent_workers=args.num_workers > 0,
            prefetch_factor=args.prefetch_factor if args.num_workers > 0 else None,
            collate_fn=omni_collate_fn,
        )
        if skip > 0:
            Logger(f'Epoch [{epoch + 1}/{args.epochs}]: 跳过前{start_step}个step，从step {start_step + 1}开始')
            train_epoch(epoch, loader, len(loader) + skip, start_step, wandb)
        else:
            train_epoch(epoch, loader, len(loader), 0, wandb)

    if dist.is_initialized():
        dist.destroy_process_group()
