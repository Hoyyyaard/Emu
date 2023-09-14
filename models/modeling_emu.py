from functools import partial
from typing import List, Optional

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.fully_sharded_data_parallel import (
    CPUOffload,
    BackwardPrefetch,
)
from torch.distributed.fsdp.wrap import (
    size_based_auto_wrap_policy,
    enable_wrap,
    wrap,
)
from torch import nn
from .causal_former import CausalFormer
from .model import MultimodalCfg, CLIPVisionCfg, VLadapterCfg, _build_vision_tower
from .transformer import LayerNorm

try:
    from transformers import BeamSearchScorer, LogitsProcessorList, MinLengthLogitsProcessor, StoppingCriteriaList, \
        MaxLengthCriteria
except ImportError as e:
    pass

from transformers.generation.configuration_utils import GenerationConfig
GENERATION_CONFIG = GenerationConfig(bos_token_id=1, eos_token_id=2, pad_token_id=32000)


class Emu(nn.Module):
    def __init__(
        self,
        embed_dim,
        multimodal_cfg: MultimodalCfg,
        vision_cfg: CLIPVisionCfg,
        vladapter_cfg: VLadapterCfg,
        quick_gelu: bool = False,
        cast_dtype: Optional[torch.dtype] = None,
        pad_id: int = 0,
        args=None,
        apply_lemmatizer=False,
        prompt=None
    ):
        super().__init__()

        self.args = args

        multimodal_cfg = MultimodalCfg(**multimodal_cfg) if isinstance(multimodal_cfg, dict) else multimodal_cfg
        vision_cfg = CLIPVisionCfg(**vision_cfg) if isinstance(vision_cfg, dict) else vision_cfg
        vladapter_cfg = VLadapterCfg(**vladapter_cfg) if isinstance(vladapter_cfg, dict) else vladapter_cfg

        self.visual = _build_vision_tower(
            embed_dim=embed_dim,
            vision_cfg=vision_cfg,
            cast_dtype=cast_dtype,
        )
        if vision_cfg.freeze:
            self.visual.requires_grad_(False)
            self.visual = self.visual.eval()

        norm_layer = partial(LayerNorm, eps=1e-6)
        
        self.ln_visual = norm_layer(vision_cfg.width)
        nn.init.constant_(self.ln_visual.bias, 0)
        nn.init.constant_(self.ln_visual.weight, 1.0)

        from models.modeling_llama import LLaMAForClsAndRegression
        self.decoder = LLaMAForClsAndRegression(args=args)

        if multimodal_cfg.freeze:
            self.decoder.requires_grad_(False)
            self.decoder = self.decoder.eval()

        self.cformer = CausalFormer(args=args,
                                  n_causal=vladapter_cfg.n_causal,
                                  vision_width=vision_cfg.width,
                                  output_dim=self.decoder.config.d_model)

        self.n_causal = vladapter_cfg.n_causal
        self.pad_id = pad_id

        self.prompt = prompt
        self._apply_lemmatizer = apply_lemmatizer
        self._lemmatizer = None

        self.image_placeholder = "[IMG]" + "<image>" * self.n_causal + "[/IMG]"
        
        # self.visual = self.visual.eval()
        # self.decoder = self.decoder.eval()
        # self.cformer = self.cformer.eval()
        # self.ln_visual =self.ln_visual.eval()

    def wrap_fsdp(self, wrapper_kwargs, addition_device_id):
        """
        Manually wraps submodules for FSDP and move other parameters to device_id.

        Why manually wrap?
        - all parameters within the FSDP wrapper must have the same requires_grad.
            We have a mix of frozen and unfrozen parameters.
        - model.vision_encoder.visual needs to be individually wrapped or encode_vision_x errors
            See: https://github.com/pytorch/pytorch/issues/82461#issuecomment-1269136344

        The rough wrapping structure is:
        - FlamingoModel
            - FSDP(FSDP(vision_encoder))
            - FSDP(FSDP(perceiver))
            - lang_encoder
                - FSDP(FSDP(input_embeddings))
                - FlamingoLayers
                    - FSDP(FSDP(gated_cross_attn_layer))
                    - FSDP(FSDP(decoder_layer))
                - FSDP(FSDP(output_embeddings))
                - other parameters

        Known issues:
        - Our FSDP strategy is not compatible with tied embeddings. If the LM embeddings are tied,
            train with DDP or set the --freeze_lm_embeddings flag to true.
        - With FSDP + gradient ckpting, one can increase the batch size with seemingly no upper bound.
            Although the training curves look okay, we found that downstream performance dramatically
            degrades if the batch size is unreasonably large (e.g., 100 MMC4 batch size for OPT-125M).

        FAQs about our FSDP wrapping strategy:
        Why double wrap?
        As of torch==2.0.1, FSDP's _post_forward_hook and _post_backward_hook
        only free gathered parameters if the module is NOT FSDP root.

        Why unfreeze the decoder_layers?
        See https://github.com/pytorch/pytorch/issues/95805
        As of torch==2.0.1, FSDP's _post_backward_hook is only registed if the flat param
        requires_grad=True. We need the postback to fire to avoid OOM.
        To effectively freeze the decoder layers, we exclude them from the optimizer.

        What is assumed to be frozen v. unfrozen?
        We assume that the model is being trained under normal Flamingo settings
        with these lines being called in factory.py:
            ```
            # Freeze all parameters
            model.requires_grad_(False)
            assert sum(p.numel() for p in model.parameters() if p.requires_grad) == 0

            # Unfreeze perceiver, gated_cross_attn_layers, and LM input embeddings
            model.perceiver.requires_grad_(True)
            model.lang_encoder.gated_cross_attn_layers.requires_grad_(True)
            [optional] model.lang_encoder.get_input_embeddings().requires_grad_(True)
            ```
        """
        from torch.distributed.fsdp.wrap import (
            enable_wrap,
            wrap,
        )
        from torch.distributed.fsdp import (
            FullyShardedDataParallel as FSDP,
        )
        # wrap in FSDP
        with enable_wrap(wrapper_cls=FSDP, **wrapper_kwargs):
            pass
            # self.visual = wrap(self.visual)
            # self.ln_visual = wrap(self.ln_visual)

            # self.cformer = wrap(wrap(self.cformer))
            # self.decoder = wrap(wrap(self.decoder))
            # self.cformer = nn.ModuleList(
            #     wrap(block) for block in self.cformer.modules()
            # )
        import functools
        
        self.visual = self.visual.to(torch.cuda.current_device())
        self.ln_visual = self.ln_visual.to(torch.cuda.current_device())
        self.cformer = self.cformer.to(torch.cuda.current_device())
        
        my_auto_wrap_policy = functools.partial(
            size_based_auto_wrap_policy, min_num_params=1
        )
        
        for block in self.decoder.modules():
            block.requires_grad_(False)
        
        # self.decoder = self.decoder.to(torch.cuda.current_device())
        # print("Load Model to cuda done")
        
        ignored_modules = []
        # for i in range(40):
        #     layer = getattr(self.decoder.lm.base_model.model.model.layers, f"{i}")
        #     ignored_modules.append(layer.input_layernorm)
        #     ignored_modules.append(layer.post_attention_layernorm)
        

            
        self.decoder = FSDP(self.decoder , 
            auto_wrap_policy=my_auto_wrap_policy,
            device_id=torch.cuda.current_device(),
            ignored_modules=ignored_modules,
            cpu_offload=CPUOffload(offload_params=False),
            use_orig_params=False)
        
        def apply_with_stopping_condition(module, apply_fn, apply_condition=None, stopping_condition=None, **other_args):
            if stopping_condition(module):
                return
            if apply_condition(module):
                apply_fn(module, **other_args)
                
            for name, child in module.named_children():
                apply_with_stopping_condition(
                    child,
                    apply_fn,
                    apply_condition=apply_condition,
                    stopping_condition=stopping_condition,
                    **other_args
                )
        apply_with_stopping_condition(
            module=self.decoder,
            apply_fn=lambda m: m.to(torch.cuda.current_device()),
            apply_condition=lambda m: len(list(m.children())) == 0,
            stopping_condition=lambda m: isinstance(m, FSDP),
        )

        print(f'Decoder params after fsdp {(sum(p.numel() for p in self.decoder.parameters()))*2/(1024**3):.3f} GB')
        print(f'torch.cuda.memory_reserved : {torch.cuda.memory_reserved(torch.cuda.current_device())/1024/1024.:2f}')
            
            # self.lang_encoder.old_decoder_blocks = nn.ModuleList(
            #     wrap(wrap(block)) for block in self.lang_encoder.old_decoder_blocks
            # )
            # self.lang_encoder.gated_cross_attn_layers = nn.ModuleList(
            #     wrap(wrap(layer)) if layer is not None else None
            #     for layer in self.lang_encoder.gated_cross_attn_layers
            # )
            # self.lang_encoder.init_flamingo_layers(self._use_gradient_checkpointing)
            # self.lang_encoder.set_input_embeddings(
            #     wrap(wrap(self.lang_encoder.get_input_embeddings()))
            # )
            # self.lang_encoder.set_output_embeddings(
            #     wrap(wrap(self.lang_encoder.get_output_embeddings()))
            # )
            # self.vision_encoder = wrap(wrap(self.vision_encoder))  # frozen
        
        
    @torch.jit.ignore
    def set_grad_checkpointing(self, enable=True):
        self.visual.set_grad_checkpointing(enable)
        self.cformer.set_grad_checkpointing()
        self.decoder.set_grad_checkpointing()

    def forward(self, image, text_input, input_mask, text_output=None, output_mask=None, image_latent=None,
                image_features=None):
        # [B, C, H, W] --> [B, n_patch, C_vis]
        if image_latent is None or image_features is None:
            image_features = self.visual.forward_features(image)
        # ln for visual features
        image_features = self.ln_visual(image_features)
        # [B, n_patch, C_vis] --> [B, n_causal, C_llm]
        image_features = self.cformer(image_features)
        # loss from hf lm model
        loss = self.decoder(image_features, text_input=text_input, text_output=text_output, text_mask=input_mask,
                            output_mask=output_mask)
        return loss

    @torch.no_grad()
    def generate(
        self,
        samples,
        do_sample=False,
        num_beams=5,
        max_new_tokens=50,
        min_length=1,
        top_p=0.9,
        repetition_penalty=1.0,
        length_penalty=0.0,
        num_captions=1,
        temperature=1,
        penalty_alpha=None,  # contrastive search
        top_k=None,
        no_repeat_ngram_size=None,
    ):
        GENERATION_CONFIG.pad_token_id = self.decoder.tokenizer.pad_token_id
        GENERATION_CONFIG.bos_token_id = self.decoder.tokenizer.bos_token_id
        GENERATION_CONFIG.eos_token_id = self.decoder.tokenizer.eos_token_id
        
        image = samples["image"]
        if image is not None:
            # image = image.to(dtype=torch.float16)
            image = image.to(torch.cuda.current_device()).to(dtype=torch.float16)
            image_features = self.ln_visual(self.visual.forward_features(image)).to(dtype=torch.float16)
            # image_features = image_features.cpu()
            # image_features = self.cformer(image_features).squeeze().to(dtype=torch.bfloat16)
            image_features = self.cformer(image_features).squeeze().to(dtype=torch.float16)

        prompt = samples["prompt"] if "prompt" in samples.keys() else self.prompt

        from models.modeling_llama import LLaMAForClsAndRegression
        if isinstance(self.decoder, LLaMAForClsAndRegression):
            self.decoder.tokenizer.padding_side = "left"

        input_tokens = self.decoder.tokenizer(
            prompt, 
            padding="longest", 
            return_tensors="pt",
            add_special_tokens=True,
        ).to(self.args.device)

        self.decoder.tokenizer.padding_side = "right"

        input_ids = input_tokens.input_ids[0]
        encoder_atts = input_tokens.attention_mask[0]

        img_token_id = self.decoder.tokenizer.convert_tokens_to_ids(["<image>"])[0]  # 32003
        img_token_idx_list = input_ids.eq(img_token_id).squeeze() 

        # with torch.amp.autocast(device_type=self.args.device.type, dtype=torch.bfloat16):
        with torch.amp.autocast(device_type=self.args.device.type, dtype=torch.float16):
            if self.args.instruct:
                inputs_embeds = self.decoder.lm.model.model.embed_tokens(input_ids)
            else:
                inputs_embeds = self.decoder.lm.model.embed_tokens(input_ids)

            if image is not None:
                image_features = image_features.reshape(-1, image_features.shape[-1])
                inputs_embeds[img_token_idx_list] = image_features

            inputs_embeds = inputs_embeds.unsqueeze(0)
            encoder_atts = encoder_atts.unsqueeze(0)

            outputs = self.decoder.lm.generate(
                generation_config=GENERATION_CONFIG,
                inputs_embeds=inputs_embeds,
                attention_mask=encoder_atts,
                do_sample=do_sample,
                top_p=top_p,
                temperature=temperature,
                num_beams=num_beams,
                max_new_tokens=max_new_tokens,
                min_length=min_length,
                repetition_penalty=repetition_penalty,
                length_penalty=length_penalty,
                num_return_sequences=num_captions,
                penalty_alpha=penalty_alpha,
                top_k=top_k,
                no_repeat_ngram_size=no_repeat_ngram_size,
            )

            output_text = self.decoder.tokenizer.batch_decode(
                outputs, skip_special_tokens=True
            ) 

        return output_text

    @torch.no_grad()
    def generate_image(
        self,
        text: List[str],
        image: Optional[torch.Tensor] = None,
        placeholder: str = "[<IMG_PLH>]",
    ) -> torch.Tensor:
        IMAGE, BOI = self.decoder.tokenizer.convert_tokens_to_ids(["<image>", "[IMG]"])
        device = self.ln_visual.weight.device

        if image is not None:
            # image placeholder is already injected into text prompt
            prompt_image_embeds = self.visual.forward_features(image)
            prompt_image_embeds = self.ln_visual(prompt_image_embeds)
            prompt_image_embeds = self.cformer(prompt_image_embeds)
            prompt_image_embeds = prompt_image_embeds.view(-1, prompt_image_embeds.shape[-1])

        text = [t.replace(placeholder, self.image_placeholder) for t in text]

        target_image_embeds = None
        for num_img_token in range(self.n_causal):
            if num_img_token == 0:
                text = [f"{t}[IMG]" for t in text]
            else:
                text = [f"{t}<image>" for t in text]

            inputs = self.decoder.tokenizer(text, padding="longest", return_tensors="pt")
            attention_mask = inputs.attention_mask.to(device)
            input_ids = inputs.input_ids.to(device)

            text_embeds = self.decoder.lm.model.embed_tokens(input_ids)

            image_idx = (input_ids == IMAGE)
            cumsum_idx = torch.flip(torch.cumsum(torch.flip(image_idx, dims=[1]), dim=1), dims=[1])
            if image is not None:
                prompt_idx = torch.logical_and(image_idx, cumsum_idx > num_img_token)
                text_embeds[prompt_idx] = prompt_image_embeds

            if target_image_embeds is not None:
                target_idx = torch.logical_and(image_idx, torch.logical_and(cumsum_idx > 0, cumsum_idx <= num_img_token))
                text_embeds[target_idx] = target_image_embeds

            outputs = self.decoder.lm.model(
                inputs_embeds=text_embeds,
                attention_mask=attention_mask,
                output_hidden_states=True,
                return_dict=True,
            )

            image_idx = (input_ids == IMAGE) + (input_ids == BOI)
            cumsum_idx = torch.flip(torch.cumsum(torch.flip(image_idx, dims=[1]), dim=1), dims=[1])
            target_idx = torch.logical_and(image_idx, torch.logical_and(cumsum_idx > 0, cumsum_idx <= num_img_token+1))

            hidden_states = outputs.hidden_states[-1]
            target_image_embeds = hidden_states[target_idx]
            target_image_embeds = target_image_embeds.view(-1, target_image_embeds.shape[-1])
            target_image_embeds = self.decoder.lm.stu_regress_head(target_image_embeds)

        _, C = target_image_embeds.shape
        B = hidden_states.shape[0]
        target_image_embeds = target_image_embeds.view(B, -1, C)

        return target_image_embeds
