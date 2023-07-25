import torch

from transformers import BertTokenizer


class Metric():
    def __init__(self, ent_set, rel_set, output_path=None, bad_case_output_path=None):
        super().__init__()
        self.ent_set = set(i.replace(" ", "").lower() for i in ent_set)
        self.rel_set = set(i.replace(" ", "").lower() for i in rel_set)
        self.output_path = output_path
        self.bad_case_output_path = bad_case_output_path
        self.rel_correct_num, self.rel_predict_num, self.rel_gold_num = 1e-10, 1e-10, 1e-10
        self.strict_rel_correct_num = 1e-10
        self.rel_TN_pred, self.rel_TN_num = 1e-10, 1e-10
        self.ent_correct_num, self.ent_predict_num, self.ent_gold_num = 1e-10, 1e-10, 1e-10
        self.span_correct_num, self.span_predict_num, self.span_gold_num = 1e-10, 1e-10, 1e-10
    
    def get_labels(self, _ids, tokenizer, prefix_len=0, ent_or_rel="rel"):  # vector_num, seq_len
        if type(_ids) == torch.Tensor:
            _ids = _ids.cpu().numpy().tolist()
        
        # 先整理 generate/golden token ids ， 删 cls，eos ， 切分 pad
        _ids = [i[prefix_len:] for i in _ids]  # delete [cls] / [bos]
        
        _ids = [i + [tokenizer.eos_token_id] for i in _ids]  # if [eos] not in list, avoid exception
        _ids = [i[:i.index(tokenizer.eos_token_id)] for i in _ids]  # truncate [eos] + [pad]
        
        _tokens = tokenizer.batch_decode(_ids)
        _tokens = [i.split(tokenizer.sep_token) for i in _tokens]
        _tokens = [[i.strip() for i in g] for g in _tokens]
        
        target_set = self.rel_set if ent_or_rel == "rel" else self.ent_set
        _tokens = [[i.replace(" ", "").lower() for i in g if i.replace(" ", "").lower() in target_set]
                   for g in _tokens]  # 过滤 空白 和 invalid rel
        
        return _tokens  # vector_num, label_num
    
    def help_split(self, ele_tuples):
        res = set()  # 该 token 是否有合法输出
        for t in ele_tuples:
            if not t[-1]:
                continue
            for tag in t[-1]:
                res.add((*t[:-1], tag))
        
        return res
    
    def update_span_eval(self, ent_labels, span_idxs):
        for l_, pre_ in zip(ent_labels, span_idxs):
            l_ = {i for i, j in l_.keys()}
            pre_ = set(pre_)
            self.span_gold_num += len(l_)
            self.span_correct_num += len(l_ & pre_)
            self.span_predict_num += len(pre_)
    
    def get_span_result(self):
        span_precision = self.span_correct_num / self.span_predict_num
        span_recall = self.span_correct_num / self.span_gold_num
        span_f1_score = 2 * span_precision * span_recall / (span_precision + span_recall)
        output = f'Span:\n' \
                 f'\tcorrect_num: {self.span_correct_num:.0f}, predict_num: {self.span_predict_num:.0f}, gold_num: {self.span_gold_num:.0f}\n' \
                 f'\tprecision:{span_precision:.3f}, recall:{span_recall:.3f}, f1_score:{span_f1_score:.3f}'
        return output
    
    def update_unified_eval(self, ent_preds, _rel_preds, ent_labels, rel_labels,
                            ent_tokenizer, rel_tokenizer,
                            ent_prefix_lens, rel_prefix_lens,
                            ent_pos_list, rel_pos_list,
                            texts, sentence_ent_default_flags):
        writer = open(self.bad_case_output_path, "a")
        # ent_preds = self.get_labels(inference_preds['ent_preds'], ent_tokenizer, ent_prefix_lens, "ent")
        rel_preds = self.get_labels(_rel_preds, rel_tokenizer, rel_prefix_lens, "rel")
        
        ent_s, rel_s = 0, 0
        
        assert len(ent_pos_list) == len(rel_pos_list) == len(ent_labels) == len(rel_labels) \
               == len(texts) == len(sentence_ent_default_flags)
        
        for pred_ent_pos, pred_rel_pos, gold_ents, gold_rels, t, sentence_ent_default_flag \
                in zip(ent_pos_list, rel_pos_list, ent_labels, rel_labels, texts, sentence_ent_default_flags):
            
            if not sentence_ent_default_flag:
                pred_ent_tags = ent_preds[ent_s: ent_s + len(pred_ent_pos)]
                pred_ent_tuples = [(*i, j) for i, j in zip(pred_ent_pos, pred_ent_tags)]
                gold_ent_tuples = [
                    (*i[0], j.replace("</s>", "").lower().replace(" ", "").split(ent_tokenizer.sep_token))
                    for i, j in gold_ents.items()]
                
                pred_ent_tuples = self.help_split(pred_ent_tuples)
                gold_ent_tuples = self.help_split(gold_ent_tuples)
                
                # 更新评价指标
                self.ent_correct_num += len(gold_ent_tuples & pred_ent_tuples)
                self.ent_gold_num += len(gold_ent_tuples)
                self.ent_predict_num += len(pred_ent_tuples)
            
            pred_rel_tags = rel_preds[rel_s: rel_s + len(pred_rel_pos)]
            pred_rel_pos = [[*pred_ent_pos[i], *pred_ent_pos[j]] for (i, j) in pred_rel_pos]
            pred_rel_tuples = [(*i, j) for i, j in zip(pred_rel_pos, pred_rel_tags)]
            gold_rel_tuples = [(*i, j.replace("</s>", "").lower().replace(" ", "").split(rel_tokenizer.sep_token))
                               for i, j in gold_rels.items()]
            pred_rel_tuples = self.help_split(pred_rel_tuples)
            gold_rel_tuples = self.help_split(gold_rel_tuples)
            
            # 增加 strict relation 标准
            # pred_right_ents = {(i, j) for (i, j, k) in pred_ent_tuples & gold_ent_tuples}
            # strict_pred_rel_tuples = {(s_h, s_t, o_h, o_t, rel_) for (s_h, s_t, o_h, o_t, rel_) in pred_rel_tuples if
            #                           (s_h, s_t) in pred_right_ents and (o_h, o_t) in pred_right_ents}
            
            ent_s += len(pred_ent_pos)
            rel_s += len(pred_rel_pos)
            
            self.rel_correct_num += len(gold_rel_tuples & pred_rel_tuples)
            self.rel_gold_num += len(gold_rel_tuples)
            self.rel_predict_num += len(pred_rel_tuples)
            
            # self.strict_rel_correct_num += len(gold_rel_tuples & strict_pred_rel_tuples)
            
            # # 输出 bad case
            # cont = ""
            # if len(gold_ent_tuples - pred_ent_tuples):
            #     cont += f"\tgold ent that not predicted:\t {gold_ent_tuples - pred_ent_tuples}\n"
            # if len(pred_ent_tuples - gold_ent_tuples):
            #     cont += f"\tfalse ent that predicted:\t\t {pred_ent_tuples - gold_ent_tuples}\n"
            # if len(gold_rel_tuples - pred_rel_tuples):
            #     cont += f"\tgold rel that not predicted:\t {gold_rel_tuples - pred_rel_tuples}\n"
            # if len(pred_rel_tuples - gold_rel_tuples):
            #     cont += f"\tfalse rel that predicted:\t\t {pred_rel_tuples - gold_rel_tuples}\n"
            # if cont: writer.write(f"{t}\n{cont}\n")
    
    def get_result(self):
        # rel
        rel_precision = self.rel_correct_num / self.rel_predict_num
        rel_recall = self.rel_correct_num / self.rel_gold_num
        rel_f1_score = 2 * rel_precision * rel_recall / (rel_precision + rel_recall)
        
        # strict_rel_precision = self.strict_rel_correct_num / self.rel_predict_num
        # strict_rel_recall = self.strict_rel_correct_num / self.rel_gold_num
        # strict_rel_f1_score = 2 * strict_rel_precision * strict_rel_recall / (strict_rel_precision + strict_rel_recall)
        
        # f'{datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}\n' \
        output = f'Relation: \n' \
                 f'\t        f1_score: {rel_f1_score:.3f}, precision: {rel_precision:.3f}, recall: {rel_recall:.3f} ( ' \
                 f'correct_num: {self.rel_correct_num:.0f}, predict_num: {self.rel_predict_num:.0f}, gold_num: {self.rel_gold_num:.0f}\n'
        # f'\tstrict- f1_score: {strict_rel_f1_score:.3f}, precision: {strict_rel_precision:.3f}, recall: {strict_rel_recall:.3f} ( ' \
        # f'strict_correct_num: {self.strict_rel_correct_num:.0f}, strict_predict_num: {self.rel_predict_num:.0f}, strict_gold_num: {self.rel_gold_num:.0f}\n' \
        
        # ent
        ent_precision = self.ent_correct_num / self.ent_predict_num
        ent_recall = self.ent_correct_num / self.ent_gold_num
        ent_f1_score = 2 * ent_precision * ent_recall / (ent_precision + ent_recall)
        
        output += f'Entity:\n' \
                  f'\tf1_score: {ent_f1_score:.3f}, precision: {ent_precision:.3f}, recall: {ent_recall:.3f} ( ' \
                  f'correct_num: {self.ent_correct_num:.0f}, predict_num: {self.ent_predict_num:.0f}, gold_num: {self.ent_gold_num:.0f}\n'
        
        span_precision = self.span_correct_num / self.span_predict_num
        span_recall = self.span_correct_num / self.span_gold_num
        span_f1_score = 2 * span_precision * span_recall / (span_precision + span_recall)
        
        output += f'Span:\n' \
                  f'\tf1_score: {span_f1_score:.3f}, precision: {span_precision:.3f}, recall: {span_recall:.3f} ( ' \
                  f'correct_num: {self.span_correct_num:.0f}, predict_num: {self.span_predict_num:.0f}, gold_num: {self.span_gold_num:.0f}'
        
        # open(self.output_path, "a").write(output)
        # print(output)
        return (ent_precision, ent_recall, ent_f1_score), (rel_precision, rel_recall, rel_f1_score), output
    
    def refresh(self):
        self.rel_correct_num, self.rel_predict_num, self.rel_gold_num = 1e-10, 1e-10, 1e-10
        self.strict_rel_correct_num = 1e-10
        self.ent_correct_num, self.ent_predict_num, self.ent_gold_num = 1e-10, 1e-10, 1e-10
        self.span_correct_num, self.span_predict_num, self.span_gold_num = 1e-10, 1e-10, 1e-10
        self.rel_TN_pred, self.rel_TN_num = 1e-10, 1e-10


if __name__ == "__main__":
    def get_labels(_ids, tokenizer, rel_set):  # vector_num, seq_len
        # 先整理 generate/golden token ids ， 删 cls，eos ， 切分 pad
        _ids = [i[1:] for i in _ids]  # delete [cls] / [bos]
        _ids = [i + [tokenizer.eos_token_id] for i in _ids]  # if [eos] not in list, avoid exception
        _ids = [i[:i.index(tokenizer.eos_token_id)] for i in _ids]  # truncate [eos] + [pad]
        _tokens = tokenizer.batch_decode(_ids)
        _tokens = [i.split(tokenizer.sep_token) for i in _tokens]
        _tokens = [[i.strip() for i in g] for g in _tokens]
        
        _tokens = [[i for i in g if i in rel_set] for g in _tokens]  # 过滤 空白 和 invalid rel
        
        return _tokens  # vector_num, label_num
    
    
    import json
    
    rel_set = set(
        json.load(
            open("/Users/liuyilin/Downloads/2023上海人工智能中心/SchemeFree_RE/data/webnlg_star/rel2id.json", "r")).keys())
    
    print("do not limit the tokenizer vocab size, but need to set [eos], [bos], [pad] ...")
    decoder_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    decoder_tokenizer.add_special_tokens({"eos_token": '[EOS]', "bos_token": '[CLS]', "pad_token": '[EOS]'})
    
    for context, token in decoder_tokenizer.special_tokens_map.items():
        print(f"\t\t{context} -- {token}  -- {decoder_tokenizer.encode(token, add_special_tokens=False)}")
    
    decoder_tokenizer.mask_token = None
    
    _ids = [
        [101, 5636, 17058, 30522, -100, -100, -100, -100],
        [101, 15032, 22542, 5332, 4371, 30522, -100, -100],
        [101, 3003, 18442, 30522, -100, -100, -100, -100],
        [101, 3831, 3597, 16671, 2100, 30522, -100, -100],
    ]
    
    a = get_labels(_ids, decoder_tokenizer, rel_set)
    print(a)
