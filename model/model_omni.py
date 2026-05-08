import os
import warnings
from typing import List, Optional, Tuple, Union

import torch
from torch import nn
from transformers import Siglip2ImageProcessor, Siglip2VisionModel, WhisperModel, WhisperProcessor
from transformers.modeling_outputs import MoeCausalLMOutputWithPast

from .model_minimind import *

warnings.filterwarnings('ignore')


class OmniConfig(MiniMindConfig):
    """Omni 模型配置。

    在 MiniMind 的语言模型配置之上，同时挂上图像模态和音频模态所需的字段。
    这样同一个 LLM 主干就能识别 `<|image_pad|>` 和 `<|audio_pad|>` 两种占位符。
    """

    model_type = "minimind-omni"

    def __init__(
        self,
        image_special_token='<|image_pad|>',
        image_ids=[12],
        audio_special_token='<|audio_pad|>',
        audio_ids=[16],
        **kwargs,
    ):
        self.image_special_token = image_special_token
        self.image_ids = image_ids
        self.image_hidden_size = kwargs.get('image_hidden_size', 768)
        self.image_token_len = kwargs.get('image_token_len', 64)

        self.audio_special_token = audio_special_token
        self.audio_ids = audio_ids
        self.audio_hidden_size = kwargs.get('audio_hidden_size', 512)
        self.audio_token_len = kwargs.get('audio_token_len', 64)
        super().__init__(**kwargs)


class MMVisionProjector(nn.Module):
    """把视觉编码器输出的 patch/token 特征压缩到固定数量的 LLM token。"""

    def __init__(self, in_dim, out_dim, source_tokens=256, target_tokens=64):
        super().__init__()
        self.target_tokens = target_tokens
        self.merge = max(source_tokens // target_tokens, 1)
        self.mlp = nn.Sequential(
            nn.Linear(in_dim * self.merge, out_dim),
            nn.GELU(),
            nn.Linear(out_dim, out_dim),
        )

    def forward(self, x):
        batch_size, seq_len, hidden_dim = x.shape
        expected = self.target_tokens * self.merge
        if seq_len != expected:
            # 当视觉编码器输出 token 数和默认配置不完全一致时，沿 token 维做自适应平均池化，
            # 统一压到 target_tokens * merge，再送入两层 MLP。
            x = torch.nn.functional.adaptive_avg_pool1d(
                x.transpose(1, 2), expected
            ).transpose(1, 2)
        x = x.reshape(batch_size, self.target_tokens, hidden_dim * self.merge)
        return self.mlp(x)


class MMAudioProjector(nn.Module):
    """把 Whisper 时序特征压缩到固定数量的音频 token。"""

    def __init__(self, in_dim, out_dim, target_tokens=64):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool1d(target_tokens)
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.GELU(),
            nn.Linear(out_dim, out_dim),
        )

    def forward(self, x):
        x = self.pool(x.transpose(1, 2)).transpose(1, 2)
        return self.mlp(x)


class MiniMindOmni(MiniMindForCausalLM):
    """同时支持图像和音频的 Omni 模型。

    整体思路非常直接：
    1. 文本 token 正常查表得到 embedding；
    2. 图像经过 vision_encoder + vision_proj 变成若干视觉 token；
    3. 音频经过 audio_encoder + audio_proj 变成若干音频 token；
    4. 用这些模态 token 替换输入序列中连续的 `<|image_pad|>` / `<|audio_pad|>`；
    5. 替换后的隐状态继续走同一个 LLM 主干。
    """

    config_class = OmniConfig

    def __init__(
        self,
        config: OmniConfig = None,
        vision_model_path='./model/vision_model/siglip2-base-p16-ve',
        audio_model_path='./model/audio_model',
    ):
        self.config = config or OmniConfig()
        super().__init__(self.config)

        self.vision_encoder, self.image_processor = self.__class__.get_vision_model(vision_model_path)
        self.audio_encoder, self.audio_processor = self.__class__.get_audio_model(audio_model_path)

        # 统一暴露一个 processor 字典，训练/推理脚本都可以从这里取到预处理器。
        self.processor = {
            'image': self.image_processor,
            'audio': self.audio_processor,
        }

        if self.vision_encoder is not None:
            self.config.image_hidden_size = getattr(
                self.vision_encoder.config, 'hidden_size', self.config.image_hidden_size
            )
        if self.audio_encoder is not None:
            self.config.audio_hidden_size = getattr(
                self.audio_encoder.config, 'd_model', self.config.audio_hidden_size
            )

        self.vision_proj = MMVisionProjector(
            self.config.image_hidden_size,
            self.config.hidden_size,
            target_tokens=self.config.image_token_len,
        )
        self.audio_proj = MMAudioProjector(
            self.config.audio_hidden_size,
            self.config.hidden_size,
            target_tokens=self.config.audio_token_len,
        )

    @staticmethod
    def get_vision_model(model_path: str):
        from transformers import logging as hf_logging

        hf_logging.set_verbosity_error()
        if not os.path.exists(model_path):
            return None, None
        model = Siglip2VisionModel.from_pretrained(model_path)
        processor = Siglip2ImageProcessor.from_pretrained(model_path)
        for param in model.parameters():
            param.requires_grad = False
        return model.eval(), processor

    @staticmethod
    def get_audio_model(model_path: str):
        from transformers import logging as hf_logging

        hf_logging.set_verbosity_error()
        if not os.path.exists(model_path):
            return None, None
        model = WhisperModel.from_pretrained(model_path)
        processor = WhisperProcessor.from_pretrained(model_path)
        for param in model.parameters():
            param.requires_grad = False
        return model.eval(), processor

    @staticmethod
    def image2tensor(image, processor):
        if image.mode in ['RGBA', 'LA']:
            image = image.convert('RGB')
        return processor(images=image, return_tensors='pt')

    @staticmethod
    def audio2tensor(audio, processor, sampling_rate=16000):
        return processor(audio=audio, sampling_rate=sampling_rate, return_tensors='pt')

    @staticmethod
    def get_image_embeddings(image_inputs, vision_model):
        if hasattr(image_inputs, 'keys'):
            image_inputs = {
                k: v.squeeze(1) if v.ndim > 2 and v.shape[1] == 1 else v
                for k, v in image_inputs.items()
            }
        with torch.no_grad():
            outputs = vision_model(**image_inputs)
        return outputs.last_hidden_state

    @staticmethod
    def get_audio_embeddings(audio_inputs, audio_model):
        if hasattr(audio_inputs, 'keys'):
            audio_inputs = {
                k: v.squeeze(1) if v.ndim > 3 and v.shape[1] == 1 else v
                for k, v in audio_inputs.items()
            }
        with torch.no_grad():
            outputs = audio_model.encoder(**audio_inputs)
        return outputs.last_hidden_state

    def _encode_single_vision_input(self, pixel_values):
        if pixel_values is None:
            return None
        if self.vision_encoder is None:
            raise ValueError('vision_encoder is not initialized, but pixel_values were provided.')

        if hasattr(pixel_values, 'keys'):
            sample_val = next(iter(pixel_values.values()))
            if sample_val.ndim == 5:
                num_images = sample_val.shape[0]
                flat_inputs = {k: v.flatten(0, 1) for k, v in pixel_values.items()}
                image_tensors = self.vision_proj(self.get_image_embeddings(flat_inputs, self.vision_encoder))
                return image_tensors.view(num_images, self.config.image_token_len, -1)
            return self.vision_proj(self.get_image_embeddings(pixel_values, self.vision_encoder))

        if pixel_values.ndim == 5:
            num_images = pixel_values.shape[0]
            image_tensors = self.vision_proj(
                self.get_image_embeddings({'pixel_values': pixel_values.flatten(0, 1)}, self.vision_encoder)
            )
            return image_tensors.view(num_images, self.config.image_token_len, -1)
        return self.vision_proj(self.get_image_embeddings({'pixel_values': pixel_values}, self.vision_encoder))

    def _encode_single_audio_input(self, input_features):
        if input_features is None:
            return None
        if self.audio_encoder is None:
            raise ValueError('audio_encoder is not initialized, but input_features were provided.')

        if hasattr(input_features, 'keys'):
            sample_val = next(iter(input_features.values()))
            if sample_val.ndim == 4:
                num_audios = sample_val.shape[0]
                flat_inputs = {k: v.flatten(0, 1) for k, v in input_features.items()}
                audio_tensors = self.audio_proj(self.get_audio_embeddings(flat_inputs, self.audio_encoder))
                return audio_tensors.view(num_audios, self.config.audio_token_len, -1)
            return self.audio_proj(self.get_audio_embeddings(input_features, self.audio_encoder))

        if input_features.ndim == 4:
            num_audios = input_features.shape[0]
            audio_tensors = self.audio_proj(
                self.get_audio_embeddings({'input_features': input_features.flatten(0, 1)}, self.audio_encoder)
            )
            return audio_tensors.view(num_audios, self.config.audio_token_len, -1)
        return self.audio_proj(self.get_audio_embeddings({'input_features': input_features}, self.audio_encoder))

    def _encode_vision_inputs(self, pixel_values):
        if pixel_values is None:
            return None
        if isinstance(pixel_values, list):
            return self._batch_encode_list_inputs(
                pixel_values,
                encode_single_fn=self._encode_single_vision_input,
                concat_encode_fn=self._encode_vision_inputs,
            )
        if hasattr(pixel_values, 'keys'):
            sample_val = next(iter(pixel_values.values()))
            if sample_val.ndim == 5:
                batch_size, num_images = sample_val.shape[:2]
                flat_inputs = {k: v.flatten(0, 1) for k, v in pixel_values.items()}
                image_tensors = self.vision_proj(self.get_image_embeddings(flat_inputs, self.vision_encoder))
                return image_tensors.view(batch_size, num_images, self.config.image_token_len, -1)
            return self.vision_proj(self.get_image_embeddings(pixel_values, self.vision_encoder))
        return self._encode_single_vision_input(pixel_values)

    def _encode_audio_inputs(self, input_features):
        if input_features is None:
            return None
        if isinstance(input_features, list):
            return self._batch_encode_list_inputs(
                input_features,
                encode_single_fn=self._encode_single_audio_input,
                concat_encode_fn=self._encode_audio_inputs,
            )
        if hasattr(input_features, 'keys'):
            sample_val = next(iter(input_features.values()))
            if sample_val.ndim == 4:
                batch_size, num_audios = sample_val.shape[:2]
                flat_inputs = {k: v.flatten(0, 1) for k, v in input_features.items()}
                audio_tensors = self.audio_proj(self.get_audio_embeddings(flat_inputs, self.audio_encoder))
                return audio_tensors.view(batch_size, num_audios, self.config.audio_token_len, -1)
            return self.audio_proj(self.get_audio_embeddings(input_features, self.audio_encoder))
        return self._encode_single_audio_input(input_features)

    @staticmethod
    def _batch_encode_list_inputs(modal_inputs, encode_single_fn, concat_encode_fn):
        if modal_inputs is None:
            return None
        if not isinstance(modal_inputs, list):
            return concat_encode_fn(modal_inputs)

        valid_indices = [idx for idx, item in enumerate(modal_inputs) if item is not None]
        if not valid_indices:
            return [None] * len(modal_inputs)

        valid_items = [modal_inputs[idx] for idx in valid_indices]
        first_item = valid_items[0]

        # 如果是 dict 结构，就把 batch 内有效样本沿第 0 维拼起来，一次性过 encoder。
        if hasattr(first_item, 'keys'):
            split_sizes = [next(iter(item.values())).shape[0] for item in valid_items]
            merged_inputs = {
                key: torch.cat([item[key] for item in valid_items], dim=0)
                for key in first_item.keys()
            }
            merged_outputs = concat_encode_fn(merged_inputs)
            split_outputs = list(torch.split(merged_outputs, split_sizes, dim=0))
        else:
            split_outputs = [encode_single_fn(item) for item in valid_items]

        restored_outputs = [None] * len(modal_inputs)
        for idx, output in zip(valid_indices, split_outputs):
            restored_outputs[idx] = output
        return restored_outputs

    @staticmethod
    def _select_sample_modal_tensors(modal_tensors, batch_idx):
        if modal_tensors is None:
            return None
        if isinstance(modal_tensors, list):
            return modal_tensors[batch_idx]
        return modal_tensors[batch_idx]

    @torch.compiler.disable
    def _replace_modal_embeddings(self, tokens, hidden_state, marker_id, modal_features, seqlen):
        if modal_features is None:
            return hidden_state

        if modal_features.dim() == 2:
            modal_features = modal_features.unsqueeze(0)

        seq = tokens.tolist()
        feature_idx = 0
        token_idx = 0
        while token_idx < len(seq):
            if seq[token_idx] == marker_id:
                start = token_idx
                while token_idx < len(seq) and seq[token_idx] == marker_id:
                    token_idx += 1
                if feature_idx < modal_features.size(0):
                    hidden_state = torch.cat(
                        (
                            hidden_state[:start],
                            modal_features[feature_idx][:token_idx - start],
                            hidden_state[token_idx:],
                        ),
                        dim=0,
                    )[:seqlen]
                    feature_idx += 1
            else:
                token_idx += 1
        return hidden_state

    @torch.compiler.disable
    def count_omni_proj(self, tokens, hidden_states, vision_tensors=None, audio_tensors=None, seqlen=512):
        out = []
        image_marker = self.config.image_ids[0] if self.config.image_ids else None
        audio_marker = self.config.audio_ids[0] if self.config.audio_ids else None

        for batch_idx in range(hidden_states.size(0)):
            hidden_state = hidden_states[batch_idx]
            seq = tokens[batch_idx]
            if image_marker is not None:
                hidden_state = self._replace_modal_embeddings(
                    seq,
                    hidden_state,
                    image_marker,
                    self._select_sample_modal_tensors(vision_tensors, batch_idx),
                    seqlen,
                )
            if audio_marker is not None:
                hidden_state = self._replace_modal_embeddings(
                    seq,
                    hidden_state,
                    audio_marker,
                    self._select_sample_modal_tensors(audio_tensors, batch_idx),
                    seqlen,
                )
            out.append(hidden_state)
        return torch.stack(out)

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
        use_cache: bool = False,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        labels: Optional[torch.Tensor] = None,
        pixel_values: Optional[torch.FloatTensor] = None,
        input_features: Optional[torch.FloatTensor] = None,
        **args,
    ):
        seq_length = input_ids.shape[1]
        if hasattr(past_key_values, 'layers'):
            past_key_values = None
        past_key_values = past_key_values or [None] * len(self.model.layers)
        start_pos = past_key_values[0][0].shape[1] if past_key_values[0] is not None else 0

        hidden_states = self.model.dropout(self.model.embed_tokens(input_ids))

        pixel_values = pixel_values if pixel_values is not None else args.pop('images', None)
        input_features = input_features if input_features is not None else args.pop('audio_values', None)
        input_features = input_features if input_features is not None else args.pop('pixel_audio_values', None)

        if start_pos == 0 and (pixel_values is not None or input_features is not None):
            vision_tensors = self._encode_vision_inputs(pixel_values) if pixel_values is not None else None
            audio_tensors = self._encode_audio_inputs(input_features) if input_features is not None else None
            hidden_states = self.count_omni_proj(
                tokens=input_ids,
                hidden_states=hidden_states,
                vision_tensors=vision_tensors,
                audio_tensors=audio_tensors,
                seqlen=seq_length,
            )

        position_embeddings = (
            self.model.freqs_cos[start_pos:start_pos + seq_length],
            self.model.freqs_sin[start_pos:start_pos + seq_length],
        )

        presents = []
        for layer, past_key_value in zip(self.model.layers, past_key_values):
            hidden_states, present = layer(
                hidden_states,
                position_embeddings,
                past_key_value=past_key_value,
                use_cache=use_cache,
                attention_mask=attention_mask,
            )
            presents.append(present)

        hidden_states = self.model.norm(hidden_states)
        aux_loss = sum(
            [layer.mlp.aux_loss for layer in self.model.layers if isinstance(layer.mlp, MOEFeedForward)],
            hidden_states.new_zeros(1).squeeze(),
        )
        aux_loss = aux_loss + sum(param.sum() for param in self.vision_proj.parameters()) * 0
        aux_loss = aux_loss + sum(param.sum() for param in self.audio_proj.parameters()) * 0

        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])

        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100,
            )

        return MoeCausalLMOutputWithPast(
            loss=loss,
            aux_loss=aux_loss,
            logits=logits,
            past_key_values=presents,
            hidden_states=hidden_states,
        )

    @staticmethod
    def _repeat_modal_inputs(modal_inputs, num_return_sequences):
        if modal_inputs is None:
            return None
        if isinstance(modal_inputs, list):
            repeated = []
            for item in modal_inputs:
                repeated.extend([item] * num_return_sequences)
            return repeated
        if hasattr(modal_inputs, 'keys'):
            return {
                k: v.repeat(num_return_sequences, *([1] * (v.ndim - 1)))
                for k, v in modal_inputs.items()
            }
        return modal_inputs.repeat(num_return_sequences, *([1] * (modal_inputs.ndim - 1)))

    def generate(self, *args, num_return_sequences=1, **kwargs):
        if num_return_sequences > 1 and 'pixel_values' in kwargs:
            kwargs['pixel_values'] = self._repeat_modal_inputs(kwargs['pixel_values'], num_return_sequences)
        if num_return_sequences > 1 and 'input_features' in kwargs:
            kwargs['input_features'] = self._repeat_modal_inputs(kwargs['input_features'], num_return_sequences)
        return super().generate(*args, num_return_sequences=num_return_sequences, **kwargs)
