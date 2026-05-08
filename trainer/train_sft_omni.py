import argparse
import os
import sys
import time
import warnings
from contextlib import nullcontext

import torch
import torch.distributed as dist
from torch import optim
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler

__package__ = "trainer"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from model.model_omni import OmniConfig
from trainer.train_pretrain_omni import (
    OmniLMDataset,
    init_omni_model,
    move_to_device,
    omni_checkpoint,
    omni_collate_fn,
    resolve_path,
)
from trainer.trainer_utils import (
    Logger,
    SkipBatchSampler,
    get_lr,
    init_distributed_mode,
    is_main_process,
    setup_seed,
)

warnings.filterwarnings('ignore')


def train_epoch(epoch, loader, iters, start_step=0, wandb=None):
    """Omni SFT 阶段的训练主循环。

    这里复用 omni 预训练阶段的数据组织与 checkpoint 逻辑，SFT 侧主要差别是：
    1. 默认读取 `omni_sft.parquet`；
    2. 默认从 `pretrain_omni` 继续训练；
    3. 学习率更低，并默认解冻 LLM 主干做指令跟随微调。
    """
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
    parser = argparse.ArgumentParser(description='MiniMind-Omni SFT')
    parser.add_argument('--save_dir', type=str, default='out', help='模型权重保存目录')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints', help='断点续训状态保存目录')
    parser.add_argument('--save_weight', default='sft_omni', type=str, help='保存权重的前缀名')
    parser.add_argument('--epochs', type=int, default=6, help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=48, help='batch size')
    parser.add_argument('--learning_rate', type=float, default=5e-6, help='初始学习率')
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
    parser.add_argument('--data_path', type=str, default='dataset/omni_sft.parquet', help='SFT 训练数据路径')
    parser.add_argument('--tokenizer_path', type=str, default='model', help='tokenizer路径')
    parser.add_argument('--vision_model_path', type=str, default='model/vision_model/siglip2-base-p16-ve', help='视觉模型路径')
    parser.add_argument('--audio_model_path', type=str, default='model/audio_model', help='音频模型路径')
    parser.add_argument('--from_weight', default='pretrain_omni', type=str, help='基于哪个 omni 权重训练，为 none 则不基于任何权重训练')
    parser.add_argument('--from_vlm_weight', default='none', type=str, help='仅在不加载 omni 权重时，用于初始化视觉分支/LLM 的 VLM 权重')
    parser.add_argument('--from_audiolm_weight', default='none', type=str, help='仅在不加载 omni 权重时，用于初始化音频分支的 AudioLM 权重')
    parser.add_argument('--from_resume', default=0, type=int, choices=[0, 1], help='是否自动检测并续训（0=否，1=是）')
    parser.add_argument('--freeze_llm', default=0, type=int, choices=[0, 1, 2], help='冻结策略（0=完全可训练，1=冻结+解冻第0层，2=完全冻结仅训练proj）')
    parser.add_argument('--use_compile', default=0, type=int, choices=[0, 1], help='是否使用torch.compile加速（0=否，1=是）')
    parser.add_argument('--use_wandb', action='store_true', help='是否使用wandb')
    parser.add_argument('--wandb_project', type=str, default='MiniMind-Omni-SFT', help='wandb项目名')
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
        wandb_run_name = f'MiniMind-Omni-SFT-Epoch-{args.epochs}-BatchSize-{args.batch_size}-LearningRate-{args.learning_rate}'
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
