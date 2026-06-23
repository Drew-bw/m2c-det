import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import T5TokenizerFast
from mmdet.models.language_models.modeling_t5 import T5Config, T5ForConditionalGeneration
from mmdet.registry import MODELS

class VLAlign(nn.Module):
    def __init__(self,
                 v_dim: int = 256,
                 t_dim: int = 768,
                 log_scale: float = 0.0,
                 ):
        super().__init__()
        prior_prob = 0.01
        bias_value = -math.log((1 - prior_prob) / prior_prob)

        # dot product soft token head
        self.dot_product_projection_image = nn.Linear(v_dim, t_dim, bias=True)
        self.dot_product_projection_text = nn.Identity()
        self.log_scale = nn.Parameter(torch.Tensor([log_scale]), requires_grad=True)
        self.bias_lang = nn.Parameter(torch.zeros(t_dim), requires_grad=True)  # (768，)
        self.bias0 = nn.Parameter(torch.Tensor([bias_value]), requires_grad=True)  # size (1,)

    def forward(self, x, embedding):
        """
        x: visual features (bs, num_query, 256)
        embedding: language features (bs, max_num_object, 768)
        """
        dot_product_proj_tokens = self.dot_product_projection_text(embedding)  # 768 -> 256
        dot_product_proj_queries = self.dot_product_projection_image(x)  # (bs, num_query, 256)
        dot_product_logit = torch.matmul(dot_product_proj_queries, dot_product_proj_tokens.transpose(-1, -2))
        if self.cfg.MODEL.DYHEAD.FUSE_CONFIG.CLAMP_DOT_PRODUCT:
            dot_product_logit = torch.clamp(dot_product_logit, max=50000)
            dot_product_logit = torch.clamp(dot_product_logit, min=-50000)
        return dot_product_logit

class StillClassifier(nn.Module):
    def __init__(self, hidden_dim, num_class=1):
        super().__init__()
        self.body = nn.Linear(hidden_dim, num_class)

    def forward(self, x, lang_feat=None):
        return self.body(x)

@MODELS.register_module()
class GenerateWithT5(nn.Module):

    def __init__(
            self,
            name: str ='/home/lenovo/mmProjects/pretrain/flan-t5-base',
            fix_text_decoder: bool = False,
            max_txt_len: int = 32,
            generate_loss_weight: float = 1.0,
            use_focal_loss: bool = True,
    ):
        super().__init__()
        self.max_txt_len = max_txt_len
        self.generate_loss_weight = generate_loss_weight
        self.use_focal_loss =use_focal_loss
        self.use_all_negative = True

        self.t5_tokenizer = T5TokenizerFast.from_pretrained(name)
        t5_config = T5Config.from_pretrained(name)
        t5_config.dense_act_fn = "gelu"
        self.t5_model = T5ForConditionalGeneration.from_pretrained(name, config=t5_config)
        if fix_text_decoder:
            for name, param in self.t5_model.named_parameters():
                param.requires_grad = False
        self.t5_proj = nn.Linear(256, self.t5_model.config.hidden_size)

    def forward(self, object_features, object_descriptions, object_features_att_mask):
        inputs_t5 = self.t5_proj(object_features)
        atts_t5 = object_features_att_mask

        output_tokens = self.t5_tokenizer(
            object_descriptions,
            padding="longest",
            truncation=True,
            max_length=self.max_txt_len,
            return_tensors="pt",
        ).to(object_features.device)

        encoder_atts = atts_t5

        targets = output_tokens.input_ids.masked_fill(
            output_tokens.input_ids == self.t5_tokenizer.pad_token_id, -100
        )
        inputs_embeds = inputs_t5
        loss = {}
        outputs = self.t5_model(
            inputs_embeds=inputs_embeds,
            attention_mask=encoder_atts,
            decoder_attention_mask=output_tokens.attention_mask,
            return_dict=True,
            labels=targets,
            use_focal_loss=self.use_focal_loss,
        )
        t5_loss = {'t5_loss' :outputs.loss * self.generate_loss_weight}
        loss.update(t5_loss)

        return loss

    @torch.no_grad()
    def text_decoder(
            self,
            text_decoder_inputs,
            use_nucleus_sampling=False,
            num_beams=5,
            max_length=30,
            min_length=1,
            top_p=0.9,
            repetition_penalty=1.0,
            length_penalty=1.0,
            num_captions=1,
            temperature=1,
    ):
        object_features = text_decoder_inputs['object_features']

        inputs_t5 = self.t5_proj(object_features)
        atts_t5 = text_decoder_inputs['atts_t5']


        encoder_atts = atts_t5
        inputs_embeds = inputs_t5

        outputs = self.t5_model.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=encoder_atts,
            do_sample=use_nucleus_sampling,
            top_p=top_p,
            temperature=temperature,
            num_beams=num_beams,
            max_new_tokens=max_length,
            min_length=min_length,
            repetition_penalty=repetition_penalty,
            length_penalty=length_penalty,
            num_return_sequences=num_beams, # num_captions,
            output_scores=True,
            return_dict_in_generate=True,
        )
        output_text = self.t5_tokenizer.batch_decode(
            outputs.sequences, skip_special_tokens=True
        )
        if num_beams >1:
            output_sequences_scores = outputs.sequences_scores.sigmoid()
        else:
            scores = torch.stack(list(outputs.scores) ,dim=0) # [30, 900, 32128]
            log_probs = F.log_softmax(scores, dim=-1)
            top_logprobs, predicted_classes = log_probs.topk(1)
            top_logprobs = top_logprobs.transpose(1 ,0)
            indexes = outputs.sequences > 0
            sum_top_logprobs = []
            for top_logprob, index in zip(top_logprobs, indexes):
                sum_top_logprobs.append(torch.sum(top_logprob[index[1:]], dim=0))
            output_sequences_scores = torch.tensor(sum_top_logprobs).to(log_probs.device) # [900]

        output_dict = {
            'pred_object_descriptions': output_text,
            'logprobs': output_sequences_scores,
        }

        return output_dict