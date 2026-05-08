import os
import torch
import warnings
from .model_minimind import *
from typing import Optional, Tuple, List, Union
from torch import nn
from transformers.modeling_outputs import MoeCausalLMOutputWithPast
from transformers import WhisperModel, WhisperProcessor
warnings.filterwarnings('ignore')


class AudioLMConfig(MiniMindConfig):
    """AudioLM 的配置类。

    这里在语言模型通用配置的基础上，补充了音频模态需要的几个字段：
    1. `audio_special_token`：文本中用于占位音频片段的特殊 token。
    2. `audio_ids`：上述特殊 token 在词表中的 id。
    3. `audio_hidden_size`：Whisper 编码器输出的隐藏维度。
    4. `audio_token_len`：每段音频最终压缩到多少个“伪 token”。
    """
    model_type = "minimind-audio"

    def __init__(self, audio_special_token='<|audio_pad|>', audio_ids=[16], **kwargs):
        self.audio_special_token = audio_special_token
        self.audio_ids = audio_ids
        # 本地 audio_model 是 whisper-base，默认隐藏维度为 512。
        self.audio_hidden_size = kwargs.get("audio_hidden_size", 512)
        # 一段音频最终映射成多少个 LLM token 位置。
        self.audio_token_len = kwargs.get("audio_token_len", 64)
        super().__init__(**kwargs)


class MMAudioProjector(nn.Module):
    """把 Whisper 编码后的时序特征压缩并映射到 LLM 隐空间。

    Whisper encoder 输出通常是较长的时间序列，而语言模型侧只预留了固定数量
    的 `<|audio_pad|>` 占位 token。所以这里做两步：
    1. 通过 `AdaptiveAvgPool1d` 把任意长度的时序压到 `target_tokens`。
    2. 用两层 MLP 把音频特征维度投影到语言模型 hidden size。
    """

    def __init__(self, in_dim, out_dim, target_tokens=64):
        super().__init__()
        # 沿时间维做自适应池化，让不同长度音频都能压成固定 token 数。
        self.pool = nn.AdaptiveAvgPool1d(target_tokens)
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.GELU(),
            nn.Linear(out_dim, out_dim),
        )

    def forward(self, x):
        # 输入形状: [B, T, C]
        # AdaptiveAvgPool1d 期望通道维在中间，因此先转成 [B, C, T]，池化后再转回。
        x = self.pool(x.transpose(1, 2)).transpose(1, 2)
        # 输出形状: [B, target_tokens, hidden_size]
        return self.mlp(x)


class MiniMindAudioLM(MiniMindForCausalLM):
    """MiniMind 的音频版多模态模型。

    整体思路和 VLM 类似：
    1. 文本 token 先正常走 LLM embedding。
    2. 如果输入里带了音频特征，就先送入 Whisper encoder。
    3. 再通过 `audio_proj` 压缩成固定数量的音频 token 表示。
    4. 最后用这些音频向量替换文本中的 `<|audio_pad|>` 占位位置。
    5. 替换后的序列继续走 MiniMind 主干做自回归建模。
    """
    config_class = AudioLMConfig

    def __init__(self, config: AudioLMConfig = None, audio_model_path="./model/audio_model"):
        self.config = config or AudioLMConfig()
        super().__init__(self.config)

        # 加载 Whisper 编码器和对应 processor。
        self.audio_encoder, self.processor = self.__class__.get_audio_model(audio_model_path)

        # 如果本地成功加载了 Whisper，就以真实配置为准覆盖 audio_hidden_size，
        # 避免 projector 输入维度和编码器输出维度不一致。
        if self.audio_encoder is not None:
            self.config.audio_hidden_size = getattr(self.audio_encoder.config, 'd_model', self.config.audio_hidden_size)

        # 把 Whisper 特征映射到 LLM hidden space。
        self.audio_proj = MMAudioProjector(
            self.config.audio_hidden_size,
            self.config.hidden_size,
            target_tokens=self.config.audio_token_len,
        )

    @staticmethod
    def get_audio_model(model_path: str):
        """加载本地 Whisper 模型，并冻结其参数。

        这里把 Whisper 当成固定特征提取器使用，只训练上层 projector / LLM。
        """
        from transformers import logging as hf_logging
        hf_logging.set_verbosity_error()
        if not os.path.exists(model_path):
            return None, None
        model = WhisperModel.from_pretrained(model_path)
        processor = WhisperProcessor.from_pretrained(model_path)

        # 冻结音频编码器参数，避免训练时更新 Whisper 权重。
        for param in model.parameters():
            param.requires_grad = False
        return model.eval(), processor

    @staticmethod
    def audio2tensor(audio, processor, sampling_rate=16000):
        """把原始音频波形转成 Whisper 需要的 `input_features`。"""
        return processor(audio=audio, sampling_rate=sampling_rate, return_tensors="pt")

    @staticmethod
    def get_audio_embeddings(audio_inputs, audio_model):
        """调用 Whisper encoder，提取最后一层隐藏状态。

        支持两种输入形式：
        1. dict：通常是 processor 直接返回的 `{'input_features': ...}`。
        2. 外部先整理过的同名字典。
        """
        if hasattr(audio_inputs, 'keys'):
            # 某些批处理场景下可能多出一层长度为 1 的维度，这里顺手压掉。
            audio_inputs = {
                k: v.squeeze(1) if v.ndim > 3 and v.shape[1] == 1 else v
                for k, v in audio_inputs.items()
            }
        with torch.no_grad():
            # WhisperModel 是 encoder-decoder 结构，这里只取 encoder 输出作为音频表示。
            outputs = audio_model.encoder(**audio_inputs)
        return outputs.last_hidden_state

    @torch.compiler.disable
    def count_audio_proj(self, tokens, hidden_states, audio_tensors=None, seqlen=512):
        """把音频向量替换到文本序列中的 `<|audio_pad|>` 位置。

        约定是：prompt 里会连续放若干个 `<|audio_pad|>`，数量通常等于
        `audio_token_len`。这里逐段扫描 token 序列，把连续占位区间替换成
        对应的音频特征。
        """
        if audio_tensors is None or not self.config.audio_ids:
            return hidden_states

        marker, audio_features = self.config.audio_ids[0], audio_tensors

        # 如果只传入单段音频，形状可能是 [B, T, C]，这里补成 [B, N, T, C]。
        if audio_features.dim() == 3:
            audio_features = audio_features.unsqueeze(1)

        out = []
        for batch_idx in range(hidden_states.size(0)):
            hidden_state = hidden_states[batch_idx]
            seq = tokens[batch_idx].tolist()
            feature_idx, token_idx = 0, 0

            while token_idx < len(seq):
                if seq[token_idx] == marker:
                    start = token_idx
                    while token_idx < len(seq) and seq[token_idx] == marker:
                        token_idx += 1

                    # 一段连续 `<|audio_pad|>` 对应一段音频特征。
                    if feature_idx < audio_features.size(1):
                        hidden_state = torch.cat(
                            (
                                hidden_state[:start],
                                audio_features[batch_idx][feature_idx][:token_idx - start],
                                hidden_state[token_idx:],
                            ),
                            dim=0,
                        )[:seqlen]
                        feature_idx += 1
                else:
                    token_idx += 1

            out.append(hidden_state)
        return torch.stack(out)

    def _encode_audio_inputs(self, input_features):
        """统一处理单段 / 多段音频输入，并输出投影后的音频 token 表示。

        支持的输入形状：
        1. dict，单段音频: `{'input_features': [B, 80, T]}`
        2. dict，多段音频: `{'input_features': [B, N, 80, T]}`
        3. tensor，单段音频: `[B, 80, T]`
        4. tensor，多段音频: `[B, N, 80, T]`
        """
        if self.audio_encoder is None:
            raise ValueError("audio_encoder is not initialized, but input_features were provided.")

        if hasattr(input_features, 'keys'):
            sample_val = next(iter(input_features.values()))

            if sample_val.ndim == 4:
                # 多段音频时先把 [B, N, ...] 拉平成 [B*N, ...] 一次性过 encoder，
                # 之后再 reshape 回去，减少 Python 循环。
                batch_size, num_audios = sample_val.shape[:2]
                flat_inputs = {k: v.flatten(0, 1) for k, v in input_features.items()}
                audio_tensors = self.audio_proj(self.get_audio_embeddings(flat_inputs, self.audio_encoder))
                return audio_tensors.view(batch_size, num_audios, self.config.audio_token_len, -1)

            return self.audio_proj(self.get_audio_embeddings(input_features, self.audio_encoder))

        if input_features.ndim == 4:
            batch_size, num_audios = input_features.shape[:2]
            audio_tensors = self.audio_proj(
                self.get_audio_embeddings({'input_features': input_features.flatten(0, 1)}, self.audio_encoder)
            )
            return audio_tensors.view(batch_size, num_audios, self.config.audio_token_len, -1)

        return self.audio_proj(self.get_audio_embeddings({'input_features': input_features}, self.audio_encoder))

    def forward(self,
                input_ids: Optional[torch.Tensor] = None,
                attention_mask: Optional[torch.Tensor] = None,
                past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
                use_cache: bool = False,
                logits_to_keep: Union[int, torch.Tensor] = 0,
                labels: Optional[torch.Tensor] = None,
                input_features: Optional[torch.FloatTensor] = None,
                **args):
        """前向计算。

        注意这里兼容了几种历史命名：
        1. 标准音频输入用 `input_features`
        2. 兼容 `audio_values`
        3. 再兼容之前误沿用的 `pixel_values`
        """
        seq_length = input_ids.shape[1]
        if hasattr(past_key_values, 'layers'):
            past_key_values = None
        past_key_values = past_key_values or [None] * len(self.model.layers)
        start_pos = past_key_values[0][0].shape[1] if past_key_values[0] is not None else 0

        # 先得到纯文本 embedding。
        hidden_states = self.model.dropout(self.model.embed_tokens(input_ids))

        # 兼容旧接口，优先级依次为 input_features > audio_values > pixel_values。
        input_features = input_features if input_features is not None else args.pop('audio_values', None)
        input_features = input_features if input_features is not None else args.pop('pixel_values', None)

        # 只有在 prefilling 阶段才需要把整段音频特征替换进 hidden_states。
        # 生成阶段的后续 step 会走 KV cache，不需要重复编码音频。
        if input_features is not None and start_pos == 0:
            audio_tensors = self._encode_audio_inputs(input_features)
            hidden_states = self.count_audio_proj(
                tokens=input_ids,
                hidden_states=hidden_states,
                audio_tensors=audio_tensors,
                seqlen=seq_length,
            )

        position_embeddings = (
            self.model.freqs_cos[start_pos:start_pos + seq_length],
            self.model.freqs_sin[start_pos:start_pos + seq_length]
        )

        presents = []
        for layer, past_key_value in zip(self.model.layers, past_key_values):
            hidden_states, present = layer(
                hidden_states,
                position_embeddings,
                past_key_value=past_key_value,
                use_cache=use_cache,
                attention_mask=attention_mask
            )
            presents.append(present)

        hidden_states = self.model.norm(hidden_states)

        # MoE 的辅助损失；后面的 dummy gradient 用于 DDP 下稳定保留 audio_proj 计算图。
        aux_loss = sum(
            [l.mlp.aux_loss for l in self.model.layers if isinstance(l.mlp, MOEFeedForward)],
            hidden_states.new_zeros(1).squeeze()
        )
        aux_loss = aux_loss + sum(p.sum() for p in self.audio_proj.parameters()) * 0

        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])

        loss = None
        if labels is not None:
            # 自回归语言模型标准的 shift-one-token loss。
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100
            )

        output = MoeCausalLMOutputWithPast(
            loss=loss,
            aux_loss=aux_loss,
            logits=logits,
            past_key_values=presents,
            hidden_states=hidden_states
        )
        return output

    def generate(self, *args, num_return_sequences=1, **kwargs):
        """生成接口。

        当 `num_return_sequences > 1` 时，需要把音频输入沿 batch 维复制，
        这样每条候选生成都能拿到同一份音频条件。
        """
        input_features = kwargs.get('input_features', kwargs.get('audio_values', kwargs.get('pixel_values')))
        input_key = 'input_features' if 'input_features' in kwargs else 'audio_values' if 'audio_values' in kwargs else 'pixel_values'

        if num_return_sequences > 1 and input_features is not None:
            if hasattr(input_features, 'keys'):
                kwargs[input_key] = {
                    k: v.repeat(num_return_sequences, *([1] * (v.ndim - 1)))
                    for k, v in input_features.items()
                }
            else:
                kwargs[input_key] = input_features.repeat(
                    num_return_sequences,
                    *([1] * (input_features.ndim - 1))
                )

        return super().generate(*args, num_return_sequences=num_return_sequences, **kwargs)
