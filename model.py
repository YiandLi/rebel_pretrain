import random

import numpy as np
import torch
from torch import nn
from transformers import T5ForConditionalGeneration
from transformers.modeling_outputs import BaseModelOutputWithCrossAttentions

"""
直接将单个 编码 给decoder 的版本，结果不好，输出全固定
"""


def multilabel_categorical_crossentropy(y_pred, y_true):
    """
    y_true: (batch_size, ent_type_size, seq_len, seq_len)
    y_pred: (batch_size, ent_type_size, seq_len, seq_len)
    """
    batch_size, ent_type_size = y_pred.shape[:2]
    y_true = y_true.reshape(batch_size * ent_type_size, -1)
    y_pred = y_pred.reshape(batch_size * ent_type_size, -1)
    
    y_pred = (1 - 2 * y_true) * y_pred  # -1 -> pos classes, 1 -> neg classes
    y_pred_neg = y_pred - y_true * 1e12  # mask the pred outputs of pos classes
    y_pred_pos = y_pred - (1 - y_true) * 1e12  # mask the pred outputs of neg classes
    zeros = torch.zeros_like(y_pred[..., :1])  # (batch_size * ent_type_size, 1)
    # 都加1列 0 ，exp(0)=1 表示 1
    y_pred_neg = torch.cat([y_pred_neg, zeros], dim=-1)
    y_pred_pos = torch.cat([y_pred_pos, zeros], dim=-1)
    neg_loss = torch.logsumexp(y_pred_neg, dim=-1)
    pos_loss = torch.logsumexp(y_pred_pos, dim=-1)
    return (neg_loss + pos_loss).mean()


def add_mask_tril(logits, mask):
    """
    logits: [ b, t_n, s, s ]
    mask: [ b , s ]
    """
    
    def sequence_masking(x, mask, value='-inf', axis=None):
        if mask is None:
            return x
        else:
            if value == '-inf':
                value = -1e12
            elif value == 'inf':
                value = 1e12
            assert axis > 0, 'axis must be greater than 0'
            # mask拓展axis个维度
            for _ in range(axis - 1):
                mask = torch.unsqueeze(mask, 1)  # [b, 1, 1, s]
            # 如果还是不对齐，则在mask后面拓展维度
            for _ in range(x.ndim - mask.ndim):
                mask = torch.unsqueeze(mask, mask.ndim)
            return x * mask + value * (1 - mask)
    
    if mask.dtype != logits.dtype:
        mask = mask.type(logits.dtype)
    logits = sequence_masking(logits, mask, '-inf', logits.ndim - 2)
    logits = sequence_masking(logits, mask, '-inf', logits.ndim - 1)
    # 排除下三角
    mask = torch.tril(torch.ones_like(logits), diagonal=-1)
    logits = logits - mask * 1e12
    return logits


class CustomT5ForConditionalGeneration(T5ForConditionalGeneration):
    
    def prepare_inputs_for_generation(self,
                                      input_ids,
                                      past_key_values=None,
                                      attention_mask=None,
                                      head_mask=None,
                                      decoder_head_mask=None,
                                      decoder_attention_mask=None,
                                      cross_attn_head_mask=None,
                                      use_cache=None,
                                      encoder_outputs=None,
                                      **kwargs
                                      ):
        
        res = {
            "past_key_values": past_key_values,
            "encoder_outputs": encoder_outputs,
            "attention_mask": attention_mask,
            "head_mask": head_mask,
            "decoder_head_mask": decoder_head_mask,
            "decoder_attention_mask": decoder_attention_mask,
            "cross_attn_head_mask": cross_attn_head_mask,
            "use_cache": use_cache,
        }
        
        if past_key_values is not None:
            input_ids = input_ids[:, -1:]
        if "decoder_inputs_embeds" in kwargs and past_key_values is None:  # we only want to use them in the 1st generation step
            model_inputs = {"decoder_inputs_embeds": kwargs['decoder_inputs_embeds']}
        else:
            model_inputs = {"decoder_input_ids": input_ids}
        
        res.update(model_inputs)
        
        # this solution reference: https://github.com/huggingface/transformers/issues/6535
        #   tested by transformers version 4.27.0 and  4.30.2
        # maybe another solution :https://github.com/huggingface/transformers/pull/21671
        
        return res


class ScFreeModel(nn.Module):
    
    def __init__(self, args, encoder, ent_decoder, rel_decoder):
        super(ScFreeModel, self).__init__()
        self.args = args
        
        self.encoder = encoder
        
        self.rel_decoder = rel_decoder
        self.ent_decoder = ent_decoder
        
        # self.span_length_emb = nn.Embedding(self.args.max_seq_len * 2, 64)
        
        self.agg_method = args.agg_method
        if args.agg_method == "2_d":
            og_dim = encoder.config.hidden_size * 2
        elif args.agg_method == "4_d":
            og_dim = encoder.config.hidden_size * 4
        else:
            print("The aggregation method should be specified !! Exit ...")
            exit()
        
        self.ent_down_map = torch.nn.Linear(og_dim, encoder.config.hidden_size)
        self.rel_down_map = torch.nn.Linear(og_dim, encoder.config.hidden_size)
        
        self.span_classifier = torch.nn.Sequential(
            torch.nn.Linear(og_dim, encoder.config.hidden_size),
            torch.nn.Linear(encoder.config.hidden_size, 1)
        )
        
        assert rel_decoder.config.hidden_size == encoder.config.hidden_size, \
            f"The dimension of rel_decoder({rel_decoder.config.hidden_size}) should be the same as the one of encoder({encoder.config.hidden_size}) "
        assert ent_decoder.config.hidden_size == encoder.config.hidden_size, \
            f"The dimension of ent_decoder({ent_decoder.config.hidden_size}) should be the same as the one of encoder({encoder.config.hidden_size}) "
    
    def get_loss(self, all_input_ids, all_input_mask,
                 ent_labels=None, non_ent_labels=None, rel_labels=None, non_rel_labels=None,
                 ent_inputs=None, non_ent_inputs=None, rel_inputs=None, non_rel_inputs=None,
                 ent_pos=None, ent_pos_neg=None, rel_pos_p=None, rel_pos_f=None,  # 位置信息，指导从哪里进行聚合
                 agg_idx_prefix_rel_p=None, agg_idx_prefix_rel_f=None,
                 agg_idx_prefix_ent=None, agg_idx_prefix_ent_f=None,  # 记录 aggregation 的索引位置
                 valid_ent_loss_flags=[], span_loss_only=False
                 ):
        
        assert ent_pos != None, f"[Model] entity position is not provided"
        assert rel_pos_p != None, f"[Model] relation position is not provided"
        
        # device = all_input_ids.device
        
        batch_outputs = self.encoder(input_ids=all_input_ids,
                                     attention_mask=all_input_mask)
        last_hidden_state = batch_outputs.last_hidden_state
        
        span_loss = 0
        gold_ent_loss, other_ent_loss = 0, 0
        gold_rel_loss, other_rel_loss = 0, 0
        neg_rel_nums = 0
        
        # TODO: 得到 span identification loss
        if self.args.train_span_iden:
            span_loss = self.get_span_logistics_loss(all_input_mask, last_hidden_state, ent_pos)
        
        # TODO：得到所有实体向量和对应 id
        vector_dict = self.get_ents_and_rels(last_hidden_state, ent_pos, rel_pos_p, rel_pos_f,
                                             ent_pos_neg_list=ent_pos_neg)
        
        # TODO：计算损失
        if self.args.trian_ent and not span_loss_only and vector_dict["gold_ent_doc_ids"]:
            sent_vectors = last_hidden_state[vector_dict["gold_ent_doc_ids"]]
            sent_attn_mask = all_input_mask[vector_dict["gold_ent_doc_ids"]]
            input_vector = self.ent_decoder.get_input_embeddings()(ent_inputs)  # 先得到原始 ent inputs ids 的 embedding，然后替换
            if agg_idx_prefix_ent:
                input_vector[range(len(agg_idx_prefix_ent)), agg_idx_prefix_ent] = vector_dict["gold_ent_vector"]
            
            assert len(input_vector) == len(ent_labels) == len(sent_vectors), \
                f"[Model] |encoder vectors|{len(input_vector)} != |encoder labels|{len(ent_labels)} != |encoder sentence vectors|{len(sent_vectors)}"
            
            if any(valid_ent_loss_flags):
                gold_ent_loss = self.ent_decoder(
                    encoder_outputs=BaseModelOutputWithCrossAttentions(
                        last_hidden_state=sent_vectors[valid_ent_loss_flags]),
                    attention_mask=sent_attn_mask[valid_ent_loss_flags],
                    decoder_inputs_embeds=input_vector[valid_ent_loss_flags],
                    labels=ent_labels[valid_ent_loss_flags]).loss
            
            # ----- loss for negative entities
            if self.args.ent_negative_sample.lower() != "none":  # and any(ent_pos_neg):
                
                train_gene_batch_size = self.args.batch_size * 2
                for i in range(0, len(non_ent_inputs), train_gene_batch_size):
                    j = i + train_gene_batch_size
                    
                    other_sent_vectors = last_hidden_state[vector_dict["other_ent_doc_ids"][i:j]]
                    other_sent_attn_mask = all_input_mask[vector_dict["other_ent_doc_ids"][i:j]]
                    other_input_vector = self.ent_decoder.get_input_embeddings()(non_ent_inputs[i:j])
                    if agg_idx_prefix_ent_f:
                        other_input_vector[range(len(other_input_vector)), agg_idx_prefix_ent_f[i:j]] \
                            = vector_dict["other_ent_vector"][i:j]
                    
                    this_loss = self.ent_decoder(
                        encoder_outputs=BaseModelOutputWithCrossAttentions(last_hidden_state=other_sent_vectors),
                        attention_mask=other_sent_attn_mask,
                        decoder_inputs_embeds=other_input_vector,
                        labels=non_ent_labels[i:j]).loss
                    
                    if other_ent_loss == 0:
                        other_ent_loss = this_loss
                    else:
                        other_ent_loss += this_loss
        
        if self.args.train_rel and not span_loss_only and vector_dict["gold_rel_doc_ids"]:
            sent_vectors = last_hidden_state[vector_dict["gold_rel_doc_ids"]]
            sent_attn_mask = all_input_mask[vector_dict["gold_rel_doc_ids"]]
            input_vector = self.rel_decoder.get_input_embeddings()(rel_inputs)
            
            if agg_idx_prefix_rel_p:  # 填充 aggregation vector
                input_vector[range(len(agg_idx_prefix_rel_p)), agg_idx_prefix_rel_p] = vector_dict["gold_rel_vector"]
                # input_vector[:, 0, :] = vector_dict["gold_rel_vector"]  # 替换 bos 的嵌入表示
            
            assert len(input_vector) == len(rel_labels) == len(rel_labels)
            gold_rel_loss = self.rel_decoder(
                encoder_outputs=BaseModelOutputWithCrossAttentions(last_hidden_state=sent_vectors),
                attention_mask=sent_attn_mask,
                decoder_inputs_embeds=input_vector,
                labels=rel_labels).loss
            
            sent_vectors = last_hidden_state[vector_dict["other_rel_doc_ids"]]
            sent_attn_mask = all_input_mask[vector_dict["other_rel_doc_ids"]]
            
            if len(non_rel_inputs) != len(sent_attn_mask):
                non_rel_inputs = non_rel_inputs[0][None, :].expand((len(sent_attn_mask), -1)).contiguous()
                non_rel_labels = non_rel_labels[0][None, :].expand((len(sent_attn_mask), -1)).contiguous()
                agg_idx_prefix_rel_f = [0] * len(sent_attn_mask)
            neg_rel_nums += len(sent_attn_mask)
            
            input_vector = self.rel_decoder.get_input_embeddings()(non_rel_inputs)
            
            if agg_idx_prefix_rel_f:  # 填充 aggregation vector
                input_vector[range(len(agg_idx_prefix_rel_f)), agg_idx_prefix_rel_f] = vector_dict["other_rel_vector"]
                # input_vector[:, 0, :] = vector_dict["other_rel_vector"]  # 替换 bos 的嵌入表示
            
            assert len(input_vector) == len(sent_vectors)
            other_rel_loss = self.rel_decoder(
                encoder_outputs=BaseModelOutputWithCrossAttentions(last_hidden_state=sent_vectors),
                attention_mask=sent_attn_mask,
                decoder_inputs_embeds=input_vector,
                labels=non_rel_labels).loss
        
        return span_loss, \
               gold_ent_loss, other_ent_loss, \
               gold_rel_loss, other_rel_loss, neg_rel_nums
    
    def get_span_logistics_loss(self, all_input_mask, last_hidden_state, ent_pos_list=None):
        
        batch_size, seq_len, hidden_size = last_hidden_state.shape
        raw_extend = last_hidden_state.unsqueeze(2).expand(-1, -1, seq_len, -1)
        col_extend = last_hidden_state.unsqueeze(1).expand(-1, seq_len, -1, -1)
        
        # pos_ = torch.arange(seq_len).expand(batch_size, -1)
        # pos_raw = pos_.unsqueeze(2).expand(-1, -1, seq_len)
        # pos_col = pos_.unsqueeze(1).expand(-1, seq_len, -1)
        # pos_ = (pos_col - pos_raw + seq_len).to(last_hidden_state.device)
        # pos_embedding = self.span_length_emb(pos_)  # batch_size, seq_len, seq_len, hidden
        
        if self.agg_method == "2_d":
            span_vector = torch.concat((raw_extend, col_extend), -1)
        elif self.agg_method == "4_d":
            span_vector = torch.concat(
                (raw_extend, col_extend, raw_extend * col_extend, torch.abs(raw_extend - col_extend)), -1)
        
        span_logits = self.span_classifier(span_vector).squeeze()[:, None]  # batch_size, 1, seq_len, seq_len
        
        span_logits = add_mask_tril(span_logits, mask=all_input_mask)  # mask 操作， batch_size, 1, seq_len, seq_len
        
        if not ent_pos_list:  # inference 时候直接输出概率矩阵
            return span_logits.squeeze()
        
        else:  # 训练时候，构造 0 - neg ； 1 - positive
            label = torch.zeros((batch_size, seq_len, seq_len), requires_grad=False)
            label = label.float().to(span_logits.device)  # float 类型，用作标签
            for i, span_list in enumerate(ent_pos_list):  # instance 维度
                for j, k in span_list:
                    label[i, j, k] = 1  # label 先计作2
            
            # Circle Loss
            loss = multilabel_categorical_crossentropy(span_logits, label[:, None])
            
            return loss
    
    def embed_and_span_identify(self, all_input_ids, all_input_mask):
        # 首先 span identification
        span_batch_outputs = self.encoder(input_ids=all_input_ids, attention_mask=all_input_mask)
        span_last_hidden_state = span_batch_outputs.last_hidden_state
        
        span_logistics = self.get_span_logistics_loss(all_input_mask,
                                                      span_last_hidden_state).cpu().numpy()  # batch_size, seq_len, seq_len
        
        span_idxs = [list(zip(*np.where(span_logistics[i] > self.args.span_logistic_threshold)))
                     for i in range(len(span_logistics))]  # sentence level spans
        
        return span_last_hidden_state, span_idxs, all_input_mask
    
    def inference_ent(self, last_hidden_state, all_input_mask,
                      ent_inputs, ent_mask, ent_pos_list, agg_idx_prefix_ent):
        
        ent_dicts = self.get_ents(last_hidden_state, ent_pos_list)
        
        # TODO: ent 预测分类部分
        ent_preds_ = []
        generate_batch_size = self.args.batch_size * 5
        for i in range(0, len(ent_inputs), generate_batch_size):
            j = i + generate_batch_size
            input_vector = self.ent_decoder.get_input_embeddings()(ent_inputs[i:j])
            if agg_idx_prefix_ent:
                input_vector[range(len(input_vector)), agg_idx_prefix_ent[i:j]] = ent_dicts["ent_vector"][i:j]
            
            gen_ = self.ent_decoder.generate(
                decoder_input_ids=ent_inputs[i:j],  # input_id
                # decoder 部分
                decoder_attention_mask=ent_mask[i:j],
                decoder_inputs_embeds=input_vector,
                
                # encoder 部分
                encoder_outputs=BaseModelOutputWithCrossAttentions(
                    last_hidden_state=last_hidden_state[ent_dicts["ent_doc_ids"][i:j]]),
                # 存在则不需要输入默认的 input_id 了
                attention_mask=all_input_mask[ent_dicts["ent_doc_ids"][i:j]],
                # mask 4 queried encoded input side
                max_length=self.args.max_seq_len,
            
            ).cpu()
            
            # pad 一下
            pad_ = self.ent_decoder.config.pad_token_id * torch.ones(
                (gen_.shape[0], self.args.max_seq_len - gen_.shape[1]), dtype=gen_.dtype)
            ent_preds_.append(torch.hstack((gen_, pad_)))
        
        ent_preds_ = torch.vstack(ent_preds_)
        assert len(ent_preds_) == len(ent_inputs)
        return ent_preds_, ent_dicts
    
    def inference_rel(self, last_hidden_state, all_input_mask,
                      rel_inputs, rel_mask, agg_idx_prefix_rel, rel_vectors, document_id_lists):
        rel_preds_ = []
        generate_batch_size = self.args.batch_size * 10
        for i in range(0, len(rel_inputs), generate_batch_size):
            j = i + generate_batch_size
            # print(f"inference_rel: {i} / {j} ---- {len(rel_inputs)}")
            input_vector = self.ent_decoder.get_input_embeddings()(rel_inputs[i:j])
            if agg_idx_prefix_rel:  # 填充 aggregation vector
                input_vector[range(len(input_vector)), agg_idx_prefix_rel[i:j]] = rel_vectors[i:j]
                # input_vector[:, 0, :] = vector_dict["gold_rel_vector"]  # 替换 bos 的嵌入表示
            
            gen_ = self.rel_decoder.generate(
                decoder_input_ids=rel_inputs[i:j],
                decoder_attention_mask=rel_mask[i:j],
                decoder_inputs_embeds=input_vector,
                
                encoder_outputs=BaseModelOutputWithCrossAttentions(
                    last_hidden_state=last_hidden_state[document_id_lists[i:j]]),
                attention_mask=all_input_mask[document_id_lists[i:j]],
                max_length=self.args.max_seq_len
            ).cpu()
            
            pad_ = self.rel_decoder.config.pad_token_id * torch.ones(
                (gen_.shape[0], self.args.max_seq_len - gen_.shape[1]), dtype=gen_.dtype)
            rel_preds_.append(torch.hstack((gen_, pad_)))
        
        rel_preds_ = torch.vstack(rel_preds_)
        assert len(rel_preds_) == len(rel_inputs)
        del gen_, pad_
        
        return rel_preds_
    
    def get_ents_and_rels(self, last_hidden_state, ent_pos_list, rel_pos_p, rel_pos_f,
                          ent_pos_neg_list=None):
        
        """
        得到所有合法的实体 vector 和对应的 golden_entity_id
        """
        return_dict = {
            "gold_ent_vector": None,
            "other_ent_vector": None,
            "gold_rel_vector": None,
            "other_rel_vector": None,
            
            "gold_ent_doc_ids": None,  # 每一个 ent vector 对应的 document id within batch
            "other_ent_doc_ids": None,
            "gold_rel_doc_ids": None,
            "other_rel_doc_ids": None
        }
        
        batch_size, seq_len, hidden_size = last_hidden_state.shape
        raw_extend = last_hidden_state.unsqueeze(2).expand(-1, -1, seq_len, -1)
        col_extend = last_hidden_state.unsqueeze(1).expand(-1, seq_len, -1, -1)
        
        # construct entity embedding
        if self.agg_method == "2_d":
            span_vector = torch.concat((raw_extend, col_extend), -1)
        elif self.agg_method == "4_d":
            span_vector = torch.concat(
                (raw_extend, col_extend, raw_extend * col_extend, torch.abs(raw_extend - col_extend)), -1)
        
        span_vector = self.ent_down_map(span_vector)
        
        # TODO: gold_ent_vector -> train
        gold_entity_vector = torch.stack(
            [span_vector[i, j, k] for i, span_list in enumerate(ent_pos_list) for j, k in span_list])
        gold_ent_doc_ids = [i for i, span_list in enumerate(ent_pos_list) for _ in span_list]
        
        return_dict["gold_ent_vector"] = gold_entity_vector
        return_dict["gold_ent_doc_ids"] = gold_ent_doc_ids
        
        # -------- negative entities
        if self.args.ent_negative_sample.lower() != "none":
            
            if any(ent_pos_neg_list):  # when inference, no needed
                other_entity_vector = torch.stack(
                    [span_vector[i, j, k] for i, span_list in enumerate(ent_pos_neg_list) for j, k in span_list])
                other_ent_doc_ids = [i for i, span_list in enumerate(ent_pos_neg_list) for _ in span_list]
                
                return_dict["other_ent_vector"] = other_entity_vector
                return_dict["other_ent_doc_ids"] = other_ent_doc_ids
        
        # TODO: gold_rel_vector & other_rel_vector , all defined by given golden entities
        gold_rel_vector, other_rel_vector = [], []
        gold_rel_doc_ids, other_rel_doc_ids = [], []
        
        e_s = 0
        for i, (ent_pos, ent_pos_f, _rel_pos_p, _rel_pos_f) in enumerate(
                zip(ent_pos_list, ent_pos_neg_list, rel_pos_p, rel_pos_f)):
            
            if not ent_pos: continue  # in case of not entity is identified
            
            ent_vectors = torch.stack([span_vector[i, j, k]
                                       for j, k in
                                       ent_pos + random.sample(ent_pos_f,
                                                               min(len(ent_pos_f), self.args.rel_neg_sample_num))])
            # ent_pos  ])
            
            # get rel representations
            raw_ent_vectors = ent_vectors.unsqueeze(1).expand(-1, len(ent_vectors), -1)
            col_ent_vectors = ent_vectors.unsqueeze(0).expand(len(ent_vectors), -1, -1)
            
            # aggregation
            if self.agg_method == "2_d":
                rel_vector = torch.concat((raw_ent_vectors, col_ent_vectors), -1)
            elif self.agg_method == "4_d":
                rel_vector = torch.concat(
                    (raw_ent_vectors, col_ent_vectors,
                     raw_ent_vectors * col_ent_vectors,
                     torch.abs(raw_ent_vectors - col_ent_vectors)), -1)
            
            rel_vector = self.rel_down_map(rel_vector)
            
            for s_id, o_id in _rel_pos_p:
                gold_rel_doc_ids.append(i)
                gold_rel_vector.append(rel_vector[s_id, o_id])
            
            # for s_id, o_id in _rel_pos_f:
            for s_id in range(len(rel_vector)):  # 全采样
                for o_id in range(len(rel_vector)):
                    if (s_id, o_id) in _rel_pos_p: continue
                    other_rel_doc_ids.append(i)
                    other_rel_vector.append(rel_vector[s_id, o_id])
            
            e_s += len(ent_pos)
            assert e_s != 0
        
        return_dict["gold_rel_vector"] = torch.stack(gold_rel_vector) if gold_rel_vector else None
        return_dict["other_rel_vector"] = torch.stack(other_rel_vector) if other_rel_vector else None
        return_dict["gold_rel_doc_ids"] = gold_rel_doc_ids
        return_dict["other_rel_doc_ids"] = other_rel_doc_ids
        
        # print(gold_rel_vector, "\n")
        
        return return_dict
    
    def get_ents(self, last_hidden_state, ent_pos_list):
        batch_size, seq_len, hidden_size = last_hidden_state.shape
        raw_extend = last_hidden_state.unsqueeze(2).expand(-1, -1, seq_len, -1)
        col_extend = last_hidden_state.unsqueeze(1).expand(-1, seq_len, -1, -1)
        
        # construct entity embedding
        if self.agg_method == "2_d":
            span_vector = torch.concat((raw_extend, col_extend), -1)
        elif self.agg_method == "4_d":
            span_vector = torch.concat(
                (raw_extend, col_extend, raw_extend * col_extend, torch.abs(raw_extend - col_extend)), -1)
        
        span_vector = self.ent_down_map(span_vector)
        
        # TODO: gold_ent_vector -> train
        gold_entity_vector = torch.stack(
            [span_vector[i, j, k] for i, span_list in enumerate(ent_pos_list) for j, k in span_list])
        gold_ent_doc_ids = [i for i, span_list in enumerate(ent_pos_list) for _ in span_list]
        
        return {
            "ent_vector": gold_entity_vector,
            "ent_doc_ids": gold_ent_doc_ids
        }
