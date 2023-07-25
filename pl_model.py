import logging

import numpy as np
import pytorch_lightning as pl
import torch

logging.basicConfig(format='%(message)s',
                    # format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    # datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


def sequence_padding(inputs, length=None, value=0, seq_dims=1, mode='post'):
    """Numpy函数，将序列padding到同一长度
    seq_dims : 控制维度
    """
    if length is None:
        length = np.max([np.shape(x)[:seq_dims] for x in inputs],
                        axis=0)  # label_num, max_seq_len, max_seq_len，注意这里 max_seq_len 是同batch内最长句子的长度
    elif not hasattr(length, '__getitem__'):
        length = [length]
    
    slices = [np.s_[:length[i]] for i in range(seq_dims)]
    slices = tuple(slices) if len(slices) > 1 else slices[0]
    pad_width = [(0, 0) for _ in np.shape(inputs[0])]
    
    outputs = []
    for x in inputs:
        x = x[slices]
        for i in range(seq_dims):
            if mode == 'post':
                pad_width[i] = (0, length[i] - np.shape(x)[i])
            elif mode == 'pre':
                pad_width[i] = (length[i] - np.shape(x)[i], 0)
            else:
                raise ValueError('"mode" argument must be "post" or "pre".')
        
        # pad_width是在各维度的各个方向上想要填补的长度,如（（1，2），（2，2））
        # 表示在第一个维度上水平方向上padding=1,垂直方向上padding=2
        # 在第二个维度上水平方向上padding=2,垂直方向上padding=2。
        # 如果直接输入一个整数，则说明各个维度和各个方向所填补的长度都一样。
        x = np.pad(x, pad_width, 'constant', constant_values=value)
        outputs.append(x)
    
    return np.array(outputs)


class LitModel(pl.LightningModule):
    def __init__(self, base_model, metric, ent_tokenizer, rel_tokenizer, config):
        super().__init__()
        
        self.model = base_model
        self.metric = metric
        self.ent_tokenizer = ent_tokenizer
        self.rel_tokenizer = rel_tokenizer
        self.config = config
        self.best_rel_f1 = -float('inf')
        self.save_hyperparameters()
    
    def get_ent_result(self, ent_id_preds, ent_pos_list, ent_prefix_lens):
        ent_preds = self.metric.get_labels(ent_id_preds, self.ent_tokenizer, ent_prefix_lens, "ent")
        
        ent_s = 0
        ent_predictions_by_sentence = []
        for pred_ent_pos in ent_pos_list:
            pred_ent_tags = ent_preds[ent_s: ent_s + len(pred_ent_pos)]
            ent_predictions_by_sentence.append(pred_ent_tags)
            
            ent_s += len(pred_ent_pos)
        
        return ent_preds, ent_predictions_by_sentence
    
    def generate_ent_inference_batch_by_spans(self, pred_span_tuples, batch_input_ids):
        
        # Decoder - Prompt 部分：构造模版，保存相对应的位置信息
        ent_prompt = self.config.ent_prompt.replace("[agg_ent_vector]", self.ent_tokenizer.bos_token)
        
        ent_prefixes = []
        
        for i, span_by_sentence in enumerate(pred_span_tuples):
            spans, ent_pos_, rel_pos_ = [], [], []
            for (j, k) in span_by_sentence:
                span_text = self.ent_tokenizer.decode(batch_input_ids[i, j:k + 1])
                spans.append(span_text)
                ent_prefixes += [ent_prompt.replace("{ent}", span_text)]
        
        ent_prefixes = self.ent_tokenizer.batch_encode_plus(ent_prefixes, add_special_tokens=False).input_ids
        ent_inputs = torch.tensor(sequence_padding(ent_prefixes, value=self.ent_tokenizer.pad_token_id))
        
        ent_mask = torch.ones_like(ent_inputs).float()
        ent_mask[ent_inputs == self.ent_tokenizer.pad_token_id] = 0
        
        ent_prefix_lens = [len(i) for i in ent_prefixes]
        ent_prefix_lens = max(ent_prefix_lens)
        
        # 得到 aggregation vector id
        agg_idx_prefix_ent = [i.index(self.ent_tokenizer.bos_token_id)
                              for i in ent_prefixes] if "[agg_ent_vector]" in self.config.ent_prompt else None
        
        return ent_inputs, ent_mask, ent_prefix_lens, agg_idx_prefix_ent
    
    def generate_rel_inference_batch_by_ents(self, all_ent_vectors, ent_pred_labels_by_sentence):
        rel_prompt = self.config.rel_prompt.replace("[agg_rel_vector]", self.rel_tokenizer.bos_token)
        rel_prefixes, rel_pos_list, rel_vectors, document_id_lists = [], [], [], []
        
        e_s = 0
        for i, ent_pred_labels in enumerate(ent_pred_labels_by_sentence):
            
            ent_vectors_this_sentence = all_ent_vectors[e_s: e_s + len(ent_pred_labels)]
            
            # 有 label 则为 True ， 否则 False ( 还这里面计算了 default，但是他没有被训练，无伤大雅
            # valid_ent_flags = [True if i else False for i in ent_pred_labels]
            
            # 因为没有计算 default 的损失，所以模型会胡乱生成 ent label，所以这里设置为全部 True
            # 即认定第一步 span 全都是对的，相当于并行构造 rel 了
            valid_ent_flags = [True for i in ent_pred_labels]
            
            ent_vectors_this_sentence = ent_vectors_this_sentence[valid_ent_flags]
            
            # fuse ent label emb accordingly
            ture_ent_pred_labels = [i for i in valid_ent_flags if i]
            
            if not ture_ent_pred_labels:
                rel_pos_list.append([])
                e_s += len(ent_pred_labels)
                continue
            
            rel_pos_ = []
            for h in range(len(ent_vectors_this_sentence)):
                for t in range(len(ent_vectors_this_sentence)):
                    rel_prefixes.append(rel_prompt)
                    rel_pos_.append((h, t))
                    document_id_lists.append(i)
            
            if rel_pos_:
                # ent_num_this_sent, hidden_state ->
                raw_ent_vectors = ent_vectors_this_sentence.unsqueeze(1).expand(-1, len(ent_vectors_this_sentence), -1)
                col_ent_vectors = ent_vectors_this_sentence.unsqueeze(0).expand(len(ent_vectors_this_sentence), -1, -1)
                
                # aggregation
                if self.config.agg_method == "2_d":
                    rel_vector = torch.concat((raw_ent_vectors, col_ent_vectors), -1)
                elif self.config.agg_method == "4_d":
                    rel_vector = torch.concat(
                        (raw_ent_vectors, col_ent_vectors,
                         raw_ent_vectors * col_ent_vectors,
                         torch.abs(raw_ent_vectors - col_ent_vectors)), -1)
                
                rel_vector = self.model.rel_down_map(rel_vector)
                
                rel_vectors.extend([rel_vector[i, j] for i, j in rel_pos_])
            
            rel_pos_list.append(rel_pos_)
            
            e_s += len(ent_pred_labels)
        
        assert e_s == len(all_ent_vectors)  # assure till the end
        
        rel_vectors = torch.stack(rel_vectors)
        rel_prefixes = self.rel_tokenizer.batch_encode_plus(rel_prefixes, add_special_tokens=False).input_ids
        rel_inputs = torch.tensor(sequence_padding(rel_prefixes, value=self.rel_tokenizer.pad_token_id))
        
        rel_mask = torch.ones_like(rel_inputs).float()
        rel_mask[rel_inputs == self.rel_tokenizer.pad_token_id] = 0
        
        rel_prefix_lens = [len(i) for i in rel_prefixes]
        rel_prefix_lens = max(rel_prefix_lens)
        
        agg_idx_prefix_rel = [i.index(self.rel_tokenizer.bos_token_id)
                              for i in rel_prefixes] if "[agg_rel_vector]" in self.config.rel_prompt else None
        
        return rel_inputs, rel_mask, rel_pos_list, rel_prefix_lens, agg_idx_prefix_rel, rel_vectors, document_id_lists
    
    def training_step(self, batch_train_data, batch_idx):  # TODO 定义 train 过程
        # 是否同时开始训练
        span_loss_only = bool(self.current_epoch < self.config.complete_train_begin_epoch)
        
        span_loss, gold_ent_loss, other_ent_loss, gold_rel_loss, other_rel_loss, neg_rel_nums \
            = self.model.get_loss(*batch_train_data, span_loss_only)
        
        ent_loss = gold_ent_loss + other_ent_loss
        rel_loss = gold_rel_loss + other_rel_loss
        
        rel_cur_epoch_ratio = 1 - self.config.span_loss_ratio
        if span_loss_only:
            loss = span_loss
        
        else:
            # loss = (ent_loss + rel_loss) * (1 - self.config.span_loss_ratio) + \
            #        (span_loss * self.config.span_loss_ratio)
            
            loss = (rel_loss) * (1 - self.config.span_loss_ratio) + \
                   ((ent_loss + span_loss) * self.config.span_loss_ratio)
            
            # rel_step_accumulate_ratio = (0.9 - 0.1) / (self.config.epochs - self.config.complete_train_begin_epoch)
            # rel_cur_epoch_ratio = 0.1 + (
            #         self.current_epoch - self.config.complete_train_begin_epoch) * rel_step_accumulate_ratio
            # loss = (rel_loss) * rel_cur_epoch_ratio + \
            #        (ent_loss + span_loss) * (1 - rel_cur_epoch_ratio)
        
        for name, param in self.model.named_parameters():
            # if param.grad == None:
            loss += (param * 0).sum().to(param.device)
        
        if batch_idx <= 3 or batch_idx % 50 == 0:
            logging.info(
                f"train {self.current_epoch}-{batch_idx}/{self.config.train_batch_num} -- "
                f"Span loss: {span_loss.item() if type(span_loss) == torch.Tensor else span_loss:.2f}, "
                f"Entity loss: {ent_loss.item() if type(ent_loss) == torch.Tensor else ent_loss:.2f}, "
                f"Relation loss: {rel_loss.item() if type(rel_loss) == torch.Tensor else rel_loss:.2f}")
        
        return {'loss': loss,
                'rel_cur_epoch_ratio': rel_cur_epoch_ratio,
                'train_ent_p_num': len(batch_train_data[6]),
                'train_ent_n_num': len(batch_train_data[7]),
                'train_rel_p_num': len(batch_train_data[8]),
                'train_rel_n_num': neg_rel_nums,
                'span_loss': span_loss.item() if type(span_loss) == torch.Tensor else span_loss,
                'ent_loss': ent_loss.item() if type(ent_loss) == torch.Tensor else ent_loss,
                'gold_ent_loss': gold_ent_loss.item() if type(gold_ent_loss) == torch.Tensor else gold_ent_loss,
                'other_ent_loss': other_ent_loss.item() if type(other_ent_loss) == torch.Tensor else other_ent_loss,
                'rel_loss': rel_loss.item() if type(rel_loss) == torch.Tensor else rel_loss,
                'gold_rel_loss': gold_rel_loss.item() if type(gold_rel_loss) == torch.Tensor else gold_rel_loss,
                'other_rel_loss': other_rel_loss.item() if type(other_rel_loss) == torch.Tensor else other_rel_loss}
    
    def training_epoch_end(self, outputs):
        
        train_ent_p_num = sum(i['train_ent_p_num'] for i in outputs) / self.config.train_ins_num
        train_ent_n_num = sum(i['train_ent_n_num'] for i in outputs) / self.config.train_ins_num
        train_rel_p_num = sum(i['train_rel_p_num'] for i in outputs) / self.config.train_ins_num
        train_rel_n_num = sum(i['train_rel_n_num'] for i in outputs) / self.config.train_ins_num
        
        span_loss = sum(i['span_loss'] for i in outputs) / len(outputs)
        ent_loss = sum(i['ent_loss'] for i in outputs) / len(outputs)
        gold_ent_loss = sum(i['gold_ent_loss'] for i in outputs) / len(outputs)
        other_ent_loss = sum(i['other_ent_loss'] for i in outputs) / len(outputs)
        rel_loss = sum(i['rel_loss'] for i in outputs) / len(outputs)
        gold_rel_loss = sum(i['gold_rel_loss'] for i in outputs) / len(outputs)
        other_rel_loss = sum(i['other_rel_loss'] for i in outputs) / len(outputs)
        
        content = f"Epoch-{self.current_epoch}-train-end:\n" \
                  f"\tmean_span_loss: {span_loss:.3f}\n"
        
        if bool(self.current_epoch >= self.config.complete_train_begin_epoch):
            content += f"\trelation loss ratio: {outputs[0]['rel_cur_epoch_ratio']:.4f}\n" \
                       f"\tmean_ent_loss: {ent_loss:.3f} " \
                       f"( mean_p_ent_loss: {gold_ent_loss:.3f}, mean_n_ent_loss: {other_ent_loss:.3f}\n" \
                       f"\tmean_rel_loss: {rel_loss:.3f} " \
                       f"( mean_p_rel_loss: {gold_rel_loss:.3f}, mean_n_rel_loss: {other_rel_loss:.3f}\n" \
                       f"\tavg_ent_positive_num: {train_ent_p_num:.2f}, avg_ent_negative_num: {train_ent_n_num:.2f}\n" \
                       f"\tavg_rel_positive_num: {train_rel_p_num:.2f}, avg_rel_negative_num: {train_rel_n_num:.2f}\n"
        logging.info(content + "\n" + "---" * 30 + "\n")
    
    def validation_step(self, batch_valid_data, batch_idx):  # TODO 定义 eval 过程
        batch_input_ids, batch_input_mask, ent_labels, rel_labels, texts, sentence_ent_default_flags = batch_valid_data
        
        device = batch_input_ids.device
        
        with torch.no_grad():
            last_hidden_state, span_idxs, all_input_mask \
                = self.model.embed_and_span_identify(batch_input_ids, batch_input_mask)
            
            span_num, ent_num = sum([len(i) for i in span_idxs]), 0
            
            if self.current_epoch < self.config.complete_eval_begin_epoch:  # 前几轮不需要 eval， 没有拟合太慢了
                self.metric.update_span_eval(ent_labels, span_idxs)  # 只 eval span 即可
                return {'span_num': span_num, 'ent_num': ent_num}
            
            if span_num:
                if not self.config.trian_ent:
                    span_idxs = [[i for i, j in b.keys()] for b in ent_labels]  # for test only
                
                ent_inputs, ent_mask, ent_prefix_lens, agg_idx_prefix_ent \
                    = self.generate_ent_inference_batch_by_spans(span_idxs, batch_input_ids)
                
                ent_inputs, ent_mask = ent_inputs.to(device), ent_mask.to(device)
                
                ent_id_preds, ent_dict = self.model.inference_ent(
                    last_hidden_state, all_input_mask,
                    ent_inputs, ent_mask, span_idxs, agg_idx_prefix_ent
                )
                
                ent_preds, ent_pred_by_sent = self.get_ent_result(ent_id_preds,
                                                                  span_idxs,
                                                                  ent_prefix_lens)  # sentence level 进行整理
                
                ent_num = sum([1 for i in ent_pred_by_sent for j in i if j]) + int(
                    any(sentence_ent_default_flags))  # valid ent with label 的数量
                
                logging.info(
                    f"eval {batch_idx} - ent_num: {ent_num}, sentence_ent_default_flags:{int(any(sentence_ent_default_flags))}")
                
                if ent_num:
                    rel_inputs, rel_mask, rel_pos_list, rel_prefix_lens, agg_idx_prefix_rel, rel_vectors, document_id_lists \
                        = self.generate_rel_inference_batch_by_ents(ent_dict["ent_vector"],
                                                                    ent_pred_by_sent,
                                                                    )
                    
                    rel_inputs, rel_mask, rel_vectors = rel_inputs.to(device), rel_mask.to(device), rel_vectors.to(
                        device)
                    
                    rel_id_preds = self.model.inference_rel(last_hidden_state, batch_input_mask,
                                                            rel_inputs, rel_mask, agg_idx_prefix_rel,
                                                            rel_vectors, document_id_lists)
                    
                    self.metric.update_span_eval(ent_labels, span_idxs)
                    self.metric.update_unified_eval(ent_preds, rel_id_preds, ent_labels, rel_labels,
                                                    self.ent_tokenizer, self.rel_tokenizer,
                                                    ent_prefix_lens, rel_prefix_lens, span_idxs, rel_pos_list, texts,
                                                    sentence_ent_default_flags)
            
            if ent_num == 0 or span_num == 0:  # bad case, no spans or entities is predicted
                logging.info(f"!! eval {batch_idx} identify none entities !! " \
                             f"span_num:{span_num}, ent_num:{ent_num},  any(sentence_ent_default_flags):{int(any(sentence_ent_default_flags))}")
                for gold_ents, gold_rels in zip(ent_labels, rel_labels):
                    gold_ent_tuples = [
                        (*i[0], j.replace("</s>", "").lower().replace(" ", "").split(self.ent_tokenizer.sep_token))
                        for i, j in gold_ents.items()]
                    gold_ent_tuples = self.metric.help_split(gold_ent_tuples)
                    gold_rel_tuples = [
                        (*i, j.replace("</s>", "").lower().replace(" ", "").split(self.rel_tokenizer.sep_token))
                        for i, j in gold_rels.items()]
                    gold_rel_tuples = self.metric.help_split(gold_rel_tuples)
                    
                    self.metric.ent_gold_num += len(gold_ent_tuples)
                    self.metric.span_gold_num += len(gold_ent_tuples)
                    self.metric.rel_gold_num += len(gold_rel_tuples)
            
            return {'span_num': span_num, 'ent_num': ent_num}
    
    def on_validation_epoch_start(self) -> None:
        open(self.config.bad_case_output_path, "w")  # 只保存一次，后续用 "a"
    
    def validation_epoch_end(self, outputs) -> None:
        
        total_span_sum = sum(i['span_num'] for i in outputs)
        total_ent_sum = sum(i['ent_num'] for i in outputs)
        
        
        if self.current_epoch >= self.config.complete_eval_begin_epoch:
            total_ent_sum = torch.tensor(total_ent_sum)
            total_span_sum = torch.tensor(total_span_sum)
            
            _total_ent_sum = [torch.zeros(1, dtype=torch.int64)
                              for _ in range(torch.distributed.get_world_size())]
            _total_span_sum = [torch.zeros(1, dtype=torch.int64)
                               for _ in range(torch.distributed.get_world_size())]
            
            if torch.distributions.get_rank() == 0:
                torch.distributed.all_gather(_total_ent_sum, total_ent_sum)  # 整合全部卡上的结果
                torch.distributed.all_gather(_total_span_sum, total_span_sum)  # 整合全部卡上的结果
                
                total_ent_sum = sum([i.item() for i in _total_ent_sum])
                total_span_sum = sum([i.item() for i in _total_span_sum])

                # train_instance_log = f"\tspan identification num is : {total_span_sum}, avg: {total_span_sum / self.config.dev_ins_num:.2f}\n"

                train_instance_log += f"\tentity identification num is : {total_ent_sum}, avg: {total_ent_sum / self.config.dev_ins_num:.2f}\n"
        
        if self.current_epoch < self.config.complete_eval_begin_epoch:
            logging.info(f"Epoch-{self.current_epoch}-eval-end:\n" + train_instance_log + self.metric.get_span_result())
            self.log("eval_epoch_relation_f1", 0.0000001 * self.current_epoch)  # 占位用
            self.metric.refresh()
            return
        
        (ent_precision, ent_recall, ent_f1_score), (
            rel_precision, rel_recall, rel_f1_score), log_content = self.metric.get_result()
        
        if self.best_rel_f1 < rel_f1_score:
            self.best_rel_f1 = rel_f1_score
            logging.info(f"Epoch-{self.current_epoch}-eval-end:\n" + train_instance_log + log_content)
        else:
            logging.info(f"Epoch-{self.current_epoch}-eval-end:" + "eval_epoch_relation_f1" + rel_f1_score)
        
        self.metric.refresh()
        self.log("eval_epoch_relation_f1", rel_f1_score)
    
    def test_step(self, batch_test_data, batch_idx):  # TODO 定义 eval 过程
        batch_input_ids, batch_input_mask, ent_labels, rel_labels, texts, valid_ent_loss_flags = batch_test_data
        
        device = batch_input_ids.device
        
        with torch.no_grad():
            last_hidden_state, span_idxs, all_input_mask \
                = self.model.embed_and_span_identify(batch_input_ids, batch_input_mask)
            
            span_num, ent_num = sum([len(i) for i in span_idxs]), 0
            
            if span_num:
                if not self.config.trian_ent:
                    span_idxs = [[i for i, j in b.keys()] for b in ent_labels]  # for test only
                
                ent_inputs, ent_mask, ent_prefix_lens, agg_idx_prefix_ent \
                    = self.generate_ent_inference_batch_by_spans(span_idxs, batch_input_ids)
                
                ent_inputs, ent_mask = ent_inputs.to(device), ent_mask.to(device)
                
                ent_id_preds, ent_dict = self.model.inference_ent(
                    last_hidden_state, all_input_mask,
                    ent_inputs, ent_mask, span_idxs, agg_idx_prefix_ent
                )
                
                ent_preds, ent_pred_by_sent = self.get_ent_result(ent_id_preds,
                                                                  span_idxs,
                                                                  ent_prefix_lens)  # sentence level 进行整理
                
                ent_num = sum([1 for i in ent_pred_by_sent for j in i if j]) + int(
                    not any(valid_ent_loss_flags))  # 被预测出来的 valid ent with label 的数量
                
                if ent_num:
                    rel_inputs, rel_mask, rel_pos_list, rel_prefix_lens, agg_idx_prefix_rel, rel_vectors, document_id_lists \
                        = self.generate_rel_inference_batch_by_ents(ent_dict["ent_vector"],
                                                                    ent_pred_by_sent,
                                                                    )
                    
                    rel_inputs, rel_mask, rel_vectors = rel_inputs.to(device), rel_mask.to(device), rel_vectors.to(
                        device)
                    
                    rel_id_preds = self.model.inference_rel(last_hidden_state, batch_input_mask,
                                                            rel_inputs, rel_mask, agg_idx_prefix_rel,
                                                            rel_vectors, document_id_lists)
                    
                    self.metric.update_span_eval(ent_labels, span_idxs)
                    self.metric.update_unified_eval(ent_preds, rel_id_preds, ent_labels, rel_labels,
                                                    self.ent_tokenizer, self.rel_tokenizer,
                                                    ent_prefix_lens, rel_prefix_lens, span_idxs, rel_pos_list, texts,
                                                    valid_ent_loss_flags)
            
            if ent_num == 0 or span_num == 0:  # bad case, no spans or entities is predicted
                logging.info(f"!! eval {batch_idx} identify none entities !!")
                for gold_ents, gold_rels in zip(ent_labels, rel_labels):
                    gold_ent_tuples = [
                        (*i[0], j.replace("</s>", "").lower().replace(" ", "").split(self.ent_tokenizer.sep_token))
                        for i, j in gold_ents.items()]
                    gold_ent_tuples = self.metric.help_split(gold_ent_tuples)
                    gold_rel_tuples = [
                        (*i, j.replace("</s>", "").lower().replace(" ", "").split(self.rel_tokenizer.sep_token))
                        for i, j in gold_rels.items()]
                    gold_rel_tuples = self.metric.help_split(gold_rel_tuples)
                    
                    self.metric.ent_gold_num += len(gold_ent_tuples)
                    self.metric.span_gold_num += len(gold_ent_tuples)
                    self.metric.rel_gold_num += len(gold_rel_tuples)
            
            return {'span_num': span_num, 'ent_num': ent_num}
    
    def test_epoch_end(self, outputs) -> None:
        (ent_precision, ent_recall, ent_f1_score), (
            rel_precision, rel_recall, rel_f1_score), log_content = self.metric.get_result()
        
        logging.info(f"Test-end:\n" + log_content + "\n\n")
    
    def configure_optimizers(self):
        lr = float(self.config["lr"])
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_parameters = [
            {'params': [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
             'lr': lr, 'weight_decay': 1e-8},
            {'params': [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)],
             'lr': lr, 'weight_decay': 0.0}
        ]
        optimizer = torch.optim.AdamW(optimizer_parameters, lr=lr, eps=1e-6, betas=(0.9, 0.999))
        
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer,
                                                                         self.config.train_batch_num * 2,
                                                                         1)
        
        # return ([optimizer],[scheduler])
        return {"optimizer": optimizer, "lr_scheduler": scheduler}
