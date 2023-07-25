import copy
import json
import logging
import os
from collections import defaultdict
from multiprocessing import Pool

import numpy as np
import torch
from tokenizers import AddedToken
from torch.utils.data import Dataset
from tqdm import tqdm
from transformers import T5Tokenizer, T5TokenizerFast

# from omegaconf import OmegaConf, open_dict

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


# class InputFeatures(object):
#     """A single set of features of data."""
#
#     def __init__(self, text, ent_infos, rel_infos):
#         self.text = text
#         self.ent_infos = ent_infos
#         self.rel_infos = rel_infos
#
#     def to_dict(self):
#         return {
#             'text': self.text,
#             'ent_infos': self.ent_infos,
#             'rel_infos': self.rel_infos
#         }
#
#     def from_dict(self, dic):
#         self.text = dic.get('text')
#         self.ent_infos = dic.get('ent_infos')
#         self.rel_infos = dic.get('rel_infos')


def get_decoder_tokenizer(args):  # mode: ent / rel
    decoder_tokenizer = T5Tokenizer.from_pretrained(args.model_name)
    decoder_tokenizer.add_special_tokens(
        {"additional_special_tokens": [AddedToken("."), AddedToken("-"), AddedToken(";"), AddedToken(","),
                                       AddedToken("("), AddedToken(")"), AddedToken("<"), AddedToken(">"),
                                       AddedToken("*")]})
    
    decoder_tokenizer.add_special_tokens(
        {"bos_token": '<extra_id_0>', "sep_token": '<extra_id_1>'})
    
    for context, token in decoder_tokenizer.special_tokens_map.items():
        print(f"\t\t{context} -- {token}  -- {decoder_tokenizer.encode(token, add_special_tokens=False)}")
    
    decoder_tokenizer.mask_token = None
    return decoder_tokenizer


def split_into_short_samples(tokenizer, sample, max_seq_len, sliding_len=50, encoder="t5"):
    """    sample 为 instance dict ， 输出为对应的 list 字典   """
    
    max_seq_len -= 2
    
    text = sample["text"] = sample["text"].replace("\n", " ")
    tokens = tokenizer.tokenize(text, truncation=False)
    tok2char_span = \
        tokenizer.encode_plus(text, truncation=False, return_offsets_mapping=True, add_special_tokens=False)[
            "offset_mapping"]
    
    if len(tokens) <= max_seq_len - 1: return [sample]
    
    # sliding at token level
    split_sample_list = []
    for start_ind in range(0, len(tokens), sliding_len):
        if encoder == "t5":  # if use bert, do not split a word into two samples
            while not tokens[start_ind].startswith('▁'):
                start_ind += 1  # bad case
                
                if start_ind >= len(tokens): return split_sample_list
        
        end_ind = start_ind + max_seq_len
        
        char_span_list = tok2char_span[start_ind:end_ind]
        char_level_span = [char_span_list[0][0], char_span_list[-1][1]]
        sub_text = text[char_level_span[0]:char_level_span[1]]
        
        new_sample = {
            "text": sub_text,
            "tok_offset": start_ind,
            "char_offset": char_level_span[0],
        }
        
        # relation
        sub_rel_list = []
        for rel in sample.get("relation_list", []):
            subj_tok_span = rel["subj_char_span"]
            obj_tok_span = rel["obj_char_span"]
            # if subject and object are both in this subtext, add this spo to new sample
            if subj_tok_span[0] >= char_level_span[0] and subj_tok_span[1] <= char_level_span[1] \
                    and obj_tok_span[0] >= char_level_span[0] and obj_tok_span[1] <= char_level_span[1]:
                new_rel = copy.deepcopy(rel)
                new_rel["subj_char_span"][0] -= char_level_span[0]  # char level offset
                new_rel["subj_char_span"][1] -= char_level_span[0]
                new_rel["obj_char_span"][0] -= char_level_span[0]
                new_rel["obj_char_span"][1] -= char_level_span[0]
                
                assert new_rel['subject'] == sub_text[new_rel["subj_char_span"][0]: new_rel["subj_char_span"][1]]
                assert new_rel['object'] == sub_text[new_rel["obj_char_span"][0]: new_rel["obj_char_span"][1]]
                
                sub_rel_list.append(new_rel)
        
        # entity
        sub_ent_list = []
        for ent in sample.get("entity_list", []):
            tok_char_span = ent["char_span"]
            # if entity in this subtext, add the entity to new sample
            if tok_char_span[0] >= char_level_span[0] and tok_char_span[1] <= char_level_span[1]:
                new_ent = copy.deepcopy(ent)
                # new_ent["tok_span"] = [tok_span[0] - start_ind, tok_span[1] - start_ind]
                
                new_ent["char_span"][0] -= char_level_span[0]
                new_ent["char_span"][1] -= char_level_span[0]
                assert new_ent['text'] == sub_text[new_ent["char_span"][0]: new_ent["char_span"][1]], sample
                
                sub_ent_list.append(new_ent)
        
        new_sample["entity_list"] = sub_ent_list  # maybe empty
        new_sample["relation_list"] = sub_rel_list  # maybe empty
        
        split_sample_list.append(new_sample)
        
        # all segments covered, no need to continue
        if end_ind > len(tokens):
            break
    
    return split_sample_list


def process(input):
    encoder_tokenizer, instance_dict, args, ent_tokenizer, rel_tokenizer = input
    
    #  TODO: 首先进行切分，将长度大于 max seq len 的数据拆分成为多个其他数据
    split_sample_list = split_into_short_samples(encoder_tokenizer,
                                                 instance_dict,
                                                 max_seq_len=args.max_seq_len - 2,
                                                 sliding_len=50)
    
    this_ins_rel = []
    for split_sample in split_sample_list:
        # TODO：整合数据特征
        fea = convert_example_to_features(split_sample, encoder_tokenizer, args, ent_tokenizer, rel_tokenizer)
        if type(fea) == dict:
            this_ins_rel.append(fea)
    return this_ins_rel


def read_and_load_data(og_path, args, mode, encoder_tokenizer, ent_tokenizer, rel_tokenizer):
    # if os.path.exists(f"cache/{mode}_{args.way_num}_{args.shot_num}.pth"):
    #     return torch.load(f"cache/{mode}_{args.way_num}_{args.shot_num}.pth")
    
    cache_path = os.path.join(og_path, args.dataset_path,
                              f"{encoder_tokenizer.name_or_path.split('/')[-1]}_{args.max_seq_len}_cache_{mode}_data.txt")
    cache_num_path = os.path.join(og_path, args.dataset_path, f"num_cache_data.txt")
    
    cache_num_dict = json.load(open(cache_num_path, "r")) if os.path.exists(cache_num_path) else {}
    
    if os.path.exists(cache_path):
        logging.info(f"Reload data from cache {cache_path} , instances number is {cache_num_dict[mode]}")
        return cache_path
    
    wf = open(cache_path, "w")
    
    data = json.load(open(os.path.join(og_path, args.dataset_path, f"{mode}_data.json"), "r"))
    # data = json.load(open(os.path.join(og_path, args.dataset_path, f"{mode}_data.json"), "r"))[:20]  # for test
    
    input_f = 0
    
    p = Pool(30) if torch.cuda.is_available() else Pool(2)
    logger.info(
        f"torch.cuda.is_available() == {torch.cuda.is_available()}  ; Using {p._processes} processes  / of {os.cpu_count()} cores")
    
    contents = p.imap_unordered(process,
                                iterable=[(encoder_tokenizer, i, args, ent_tokenizer, rel_tokenizer) for i in data])
    
    for fea in tqdm(contents, desc=f"{mode}_data.json : ( {len(data)} items )", ncols=10):
        
        for f in fea:
            input_f += 1
            wf.write(json.dumps(f) + "\n")
    
    p.close()
    
    cache_num_dict[mode] = input_f
    json.dump(cache_num_dict, open(cache_num_path, "w"))
    logger.info(
        f"Get totally {input_f} instances from file : {mode}_data.json , original instances num is : {len(data)} ")
    
    return cache_path


def convert_example_to_features(
        instance_dict, tokenizer, args, ent_tokenizer, rel_tokenizer
):
    ent_return, rel_return = defaultdict(set), defaultdict(set)
    
    _res = tokenizer(instance_dict['text'],
                     return_offsets_mapping=True,
                     max_length=args.max_seq_len,
                     truncation=True
                     )
    token2char_span_mapping = _res["offset_mapping"]
    # input_ids = _res['input_ids']
    assert len(token2char_span_mapping) <= args.max_seq_len
    
    if not token2char_span_mapping:  # ! 有特殊字符报错("\u2063") 导致 word_tokens = []
        return "bad case token"
    
    # { 每个token的开始字符的索引: 第几个token } and { end_index : token_index }
    start_mapping = {j[0]: i for i, j in enumerate(token2char_span_mapping) if j != (0, 0)}
    end_mapping = {j[-1]: i for i, j in enumerate(token2char_span_mapping) if j != (0, 0)}
    
    # 将raw_text的下标 与 token的start和end下标对应
    for ent_info in instance_dict.get("entity_list", []):  # 全都是index索引，label不用额外转换
        span, [start, end], type = ent_info['text'], ent_info['char_span'], ent_info['type']
        assert instance_dict['text'][start: end] == span
        
        # GPT2 / flan-t5 特殊编码，前置空格，所以 start -1
        if 'flan-t5' in tokenizer.name_or_path or 'gpt' in tokenizer.name_or_path or 'v1' in tokenizer.name_or_path:
            if start != 0: start -= 1
        
        if start in start_mapping:
            while end < len(instance_dict['text']) and end not in end_mapping:
                end += 1
        
        if start in start_mapping and end in end_mapping:
            # ent_return[((start, end), span)].add(type)
            
            assert "\t" not in span, span
            key = "\t".join([str(start_mapping[start]), str(end_mapping[end]), span])
            ent_return[f"{key}"].add(type)
            # assert tokenizer.decode(input_ids[start: end + 1]).lower().strip() == span.lower()
        
        else:
            print(f"\tEntity ''{span}'' out of max seq_len {args.max_seq_len}, "
                  f"text {instance_dict['text'][:50]} ...")
    
    for rel_info in instance_dict.get("relation_list", []):
        sub_start, sub_end = rel_info['subj_char_span']
        obj_start, obj_end = rel_info['obj_char_span']
        
        #  GPT2 / flan-t5 特殊编码，前置空格，所以 start -1
        if 'flan-t5' in tokenizer.name_or_path or 'gpt' in tokenizer.name_or_path or 'v1' in tokenizer.name_or_path:
            if sub_start != 0: sub_start -= 1
            if obj_start != 0: obj_start -= 1
        
        if sub_start in start_mapping:
            while sub_end < len(instance_dict['text']) and sub_end not in end_mapping:
                sub_end += 1
        
        if obj_start in start_mapping:
            while obj_end < len(instance_dict['text']) and obj_end not in end_mapping:
                obj_end += 1
        
        if sub_start in start_mapping and sub_end in end_mapping and obj_start in start_mapping and obj_end in end_mapping:
            sub_start, sub_end = start_mapping[sub_start], end_mapping[sub_end]
            obj_start, obj_end = start_mapping[obj_start], end_mapping[obj_end]
            type = rel_info['predicate']
            key = "\t".join([str(sub_start), str(sub_end), str(obj_start), str(obj_end)])
            rel_return[key].add(type)
        # else:
        #     print(
        #         f"\tRelation ({rel_info['subject']}, {rel_info['predicate']}, {rel_info['object']}) out of max seq_len {args.max_seq_len}, "
        #         f"text {instance_dict['text'][:50]} ...")
    
    ent_bos, ent_sep, ent_eos = ent_tokenizer.bos_token, ent_tokenizer.sep_token, ent_tokenizer.eos_token
    rel_bos, rel_sep, rel_eos = rel_tokenizer.bos_token, rel_tokenizer.sep_token, rel_tokenizer.eos_token
    
    if ent_return or rel_return:  # 直接整理成为 decoder 输出格式 先不加 cls， eos
        for i in ent_return: ent_return[i] = f" {ent_sep} ".join(ent_return[i]) + f" {ent_eos}"
        
        for i in rel_return:
            if len(rel_return[i]) > 2:
                logging.info(f"multiple relation: {i} -- {rel_return[i]}")
            rel_return[i] = f" {rel_sep} ".join(rel_return[i]) + f" {ent_eos}"
        
        # if len(set(list(rel_return.values()))) != len(rel_return.values()):
        #     print(rel_return)
        
        return {
            'text': instance_dict['text'],
            'ent_infos': ent_return,
            'rel_infos': rel_return
        }
    
    else:
        return "no ent / no rel"


class MyDataset(Dataset):
    def __init__(self, data, args, encoder_tokenizer, ent_tokenizer, rel_tokenizer, mode):
        self.data = data
        self.args = args
        self.mode = mode
        self.encoder_tokenizer = encoder_tokenizer
        self.ent_tokenizer = ent_tokenizer
        self.rel_tokenizer = rel_tokenizer
    
    def __getitem__(self, index):
        return self.data[index]
    
    def __len__(self):
        return len(self.data)
    
    def sequence_padding(self, inputs, length=None, value=0, seq_dims=1, mode='post'):
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
    
    def generate_inference_batch_of_inputs(self, features):
        batch_input_ids, batch_input_mask = [], []
        ent_labels, rel_labels, texts = [], [], []
        for idx, _feature in enumerate(features):
            encoder_txt = self.encoder_tokenizer.encode_plus(_feature['text'], max_length=self.args.max_seq_len,
                                                             truncation=True)
            batch_input_ids.append(encoder_txt["input_ids"])
            batch_input_mask.append(encoder_txt["attention_mask"])
            
            ent_labels.append(_feature['ent_infos'])
            rel_labels.append(_feature['rel_infos'])
            texts.append(_feature['text'])
        
        batch_input_ids = torch.tensor(self.sequence_padding(batch_input_ids)).long()
        batch_input_mask = torch.tensor(self.sequence_padding(batch_input_mask)).float()
        
        return batch_input_ids, batch_input_mask, ent_labels, rel_labels, texts
    
    def negative_sample_ent(self, input_ids, ent_pos_tuples, ent_negative_sample):
        sample_ratio = torch.ones(len(input_ids) * len(input_ids)).reshape((len(input_ids), len(input_ids)))
        
        if ent_negative_sample.lower() == 'bernoulli':
            # 过滤不合法span，根据长度进行偏移符合分布
            target_num = len(ent_pos_tuples) * 2
            max_span_len = max(j - i for i, j in ent_pos_tuples)
            for i in range(len(input_ids)):
                for j in range(len(input_ids)):
                    if i > j: sample_ratio[i][j] = 0
                    # if j - i > max_span_len: sample_ratio[i][j] = 0
            
            sample_ratio = target_num * sample_ratio / sample_ratio.sum()
            sample_ratio[sample_ratio > 1] = 1
            sample_ratio = torch.bernoulli(sample_ratio)
        
        # elif ent_negative_sample == "all":
        
        # sample_ratio[:, -1] = 0  # 删除最后的 [sep] token
        # sample_ratio[-1, :] = 0  # 删除最后的 [sep] token
        
        for i, j in ent_pos_tuples:  # 删除 gold 部分
            sample_ratio[i, j] = 0
        
        rels_pos = list(zip(*np.where(sample_ratio > 0)))
        
        rels_texts = [self.encoder_tokenizer.decode(input_ids[i:j + 1]) for i, j in rels_pos]
        return rels_pos, rels_texts
    
    def generate_batch(self, features):
        """
            inputs = {"input_ids": torch.tensor([input_ids])}
            outputs = model(**inputs, labels=torch.tensor([input_ids]))
        """
        batch_input_ids, batch_input_mask = [], []  # for encoder
        instance_txt, rel_pos_p, rel_pos_f, ent_labels, rel_labels = [], [], [], [], []  # for decoder
        ent_pos, ent_txt, ent_pos_neg, ent_txt_neg = [], [], [], []
        rel_labels_p, rel_labels_f = [], []
        ent_inputs_p, ent_inputs_f, rel_inputs_p, rel_inputs_f = [], [], [], []
        agg_idx_prefix_rel_p, agg_idx_prefix_rel_f, agg_idx_prefix_ent_p, agg_idx_prefix_ent_f = [], [], [], []
        # Decoder - Prompt 部分：构造模版，保存相对应的位置信息
        ent_prompt = self.args.ent_prompt.replace("[agg_ent_vector]", self.ent_tokenizer.bos_token)
        rel_prompt = self.args.rel_prompt.replace("[agg_rel_vector]", self.rel_tokenizer.bos_token)
        rel_prefixes_p, rel_prefixes_f, ent_prefixes_p, ent_prefixes_f = [], [], [], []
        
        for idx, _feature in enumerate(features):
            encoder_txt = self.encoder_tokenizer.encode_plus(_feature['text'],
                                                             max_length=self.args.max_seq_len,
                                                             truncation=True)
            batch_input_ids.append(encoder_txt["input_ids"])
            batch_input_mask.append(encoder_txt["attention_mask"])
            
            ent_tuples = list(_feature['ent_infos'].keys())
            ent_tuples = [((int(i), int(j)), k) for i, j, k in ent_tuples.split("\t")]
            ent_pos_tuples = [i[0] for i in ent_tuples]
            ent_txt_tuples = [i[1] for i in ent_tuples]
            ent_pos_tuples2id = {k: i for i, k in enumerate(ent_pos_tuples)}
            
            instance_txt.append(_feature['text'])
            ent_pos.append(ent_pos_tuples)  # instance level
            ent_txt.append(ent_txt_tuples)
            
            ent_pos_tuples_neg, ent_txt_tuples_neg = [], []
            if self.args.ent_negative_sample.lower() != "none":
                ent_pos_tuples_neg, ent_txt_tuples_neg = self.negative_sample_ent(encoder_txt["input_ids"],
                                                                                  ent_pos_tuples,
                                                                                  self.args.ent_negative_sample)
            
            ent_pos_neg.append(ent_pos_tuples_neg)
            ent_txt_neg.append(ent_txt_tuples_neg)
            
            rel_infos_keys = [i.split("\t") for i in list(_feature['rel_infos'].keys())]
            this_instance_rel_pos_p = [(ent_pos_tuples2id[(s_s, s_e)], ent_pos_tuples2id[(o_s, o_e)])
                                       for s_s, s_e, o_s, o_e
                                       in rel_infos_keys]
            rel_pos_p.append(this_instance_rel_pos_p)
            
            this_instance_rel_pos_f = [(i, j)
                                       for i in range(len(_feature['ent_infos']))
                                       for j in range(len(_feature['ent_infos']))
                                       if (i, j) not in this_instance_rel_pos_p]
            
            # assert len(this_instance_rel_pos_p) > 0
            if self.mode != "test" \
                    and self.args.rel_negative_sample == 'bernoulli' \
                    and len(this_instance_rel_pos_f) > 0:
                # and self.mode == "train"
                
                if len(this_instance_rel_pos_p) > len(this_instance_rel_pos_f):
                    sample_id = torch.ones(len(this_instance_rel_pos_f))
                else:
                    sample_id = torch.bernoulli(
                        torch.ones(len(this_instance_rel_pos_f))
                        * len(this_instance_rel_pos_p) / len(this_instance_rel_pos_f)
                    )  # bernoulli sample
                # if sum(sample_id) == 0:  # 如果采样全部为 0，则强制采样一个
                #     sample_id = torch.zeros(len(this_instance_rel_pos_f))
                #     sample_id[random.randint(0, len(this_instance_rel_pos_f) - 1)] = 1
                this_instance_rel_pos_f = [i for i, j in zip(this_instance_rel_pos_f, sample_id) if j == 1]
            # else: 'complete' sample by default
            rel_pos_f.append(this_instance_rel_pos_f)
            
            # TODO: 存入该数据对应的所有的单个的ent/rel 的输出的 text 标签，和前缀信息
            ent_labels.extend(list(_feature['ent_infos'].values()))  # 不是句子level的了，而是 aggregation vector 纬度的
            ent_prefixes_p += [ent_prompt.replace("{ent}", i) for i in ent_txt_tuples]
            ent_prefixes_f += [ent_prompt.replace("{ent}", i) for i in ent_txt_tuples_neg]
            
            rel_labels.extend(list(_feature['rel_infos'].values()))
            rel_prefixes_p += [
                rel_prompt.replace("{sub}", ent_txt_tuples[sub_]).replace("{obj}", ent_txt_tuples[obj_])
                for (sub_, obj_) in this_instance_rel_pos_p]
            rel_prefixes_f += [
                rel_prompt.replace("{sub}", ent_txt_tuples[sub_]).replace("{obj}", ent_txt_tuples[obj_])
                for (sub_, obj_) in this_instance_rel_pos_f]
        
        non_rel_labels = self.rel_tokenizer.encode(
            f"{self.args.none_label_prompt} {self.rel_tokenizer.eos_token}",
            add_special_tokens=False)
        non_ent_labels = self.ent_tokenizer.encode(
            f"{self.args.none_ent_prompt} {self.ent_tokenizer.eos_token}",
            add_special_tokens=False)
        
        # padding
        batch_input_ids = torch.tensor(self.sequence_padding(batch_input_ids)).long()
        batch_input_mask = torch.tensor(self.sequence_padding(batch_input_mask)).float()
        
        # 将 prefix 和 label 进行拼接得到 input  ，并且设置 label
        assert len(rel_labels) == len(rel_prefixes_p)
        if ent_labels:
            ent_labels = self.ent_tokenizer.batch_encode_plus(ent_labels, add_special_tokens=False).input_ids
            ent_prefixes_p = self.ent_tokenizer.batch_encode_plus(ent_prefixes_p, add_special_tokens=False).input_ids
            ent_prefixes_f = self.ent_tokenizer.batch_encode_plus(ent_prefixes_f,
                                                                  add_special_tokens=False).input_ids if ent_prefixes_f else []
            ent_prefix_lens_p = [len(i) for i in ent_prefixes_p]
            ent_prefix_lens_f = [len(i) for i in ent_prefixes_f]
            if "[agg_ent_vector]" in self.args.ent_prompt:
                agg_idx_prefix_ent_p = [i.index(self.ent_tokenizer.bos_token_id) for i in ent_prefixes_p]
                agg_idx_prefix_ent_f = [i.index(self.ent_tokenizer.bos_token_id) for i in ent_prefixes_f]
            else:
                agg_idx_prefix_ent_p, agg_idx_prefix_ent_f = None, None
            ent_labels_p = [prefixes_p + label for prefixes_p, label in zip(ent_prefixes_p, ent_labels)]
            ent_labels_f = [prefixes_f + non_ent_labels for prefixes_f in ent_prefixes_f] if ent_prefixes_f else []
            
            # gpt2 不计算 -100 的损失
            ent_inputs_p = torch.tensor(self.sequence_padding(ent_labels_p, value=self.ent_tokenizer.pad_token_id))
            ent_inputs_f = torch.tensor(
                self.sequence_padding(ent_labels_f, value=self.ent_tokenizer.pad_token_id)) if ent_labels_f else []
            ent_labels_p = torch.tensor(self.sequence_padding(ent_labels_p, value=-100))
            ent_labels_f = torch.tensor(self.sequence_padding(ent_labels_f, value=-100)) if ent_labels_f else []
            for i in range(len(ent_labels_p)):  # prefix 部分不计算 loss，所以设置为 -100
                ent_labels_p[i][:ent_prefix_lens_p[i]] = -100
            for i in range(len(ent_labels_f)):  # prefix 部分不计算 loss，所以设置为 -100
                ent_labels_f[i][:ent_prefix_lens_f[i]] = -100
            
            # For T5 only, prepare label
            ent_labels_p = torch.cat(
                (ent_labels_p[:, 1:], torch.ones((ent_labels_p.shape[0], 1), dtype=ent_labels_p.dtype) * -100),
                dim=-1)
            ent_labels_f = torch.cat(
                (ent_labels_f[:, 1:], torch.ones((ent_labels_f.shape[0], 1), dtype=ent_labels_f.dtype) * -100),
                dim=-1) \
                if ent_labels_f != [] else []
        
        if rel_labels:
            rel_labels = self.rel_tokenizer.batch_encode_plus(rel_labels, add_special_tokens=False).input_ids
            rel_prefixes_p = self.rel_tokenizer.batch_encode_plus(rel_prefixes_p, add_special_tokens=False).input_ids
            rel_prefixes_f = self.rel_tokenizer.batch_encode_plus(rel_prefixes_f, add_special_tokens=False).input_ids
            
            rel_prefix_lens_p = [len(i) for i in rel_prefixes_p]
            rel_prefix_lens_f = [len(i) for i in rel_prefixes_f]
            if "[agg_rel_vector]" in self.args.rel_prompt:
                agg_idx_prefix_rel_p = [i.index(self.rel_tokenizer.bos_token_id) for i in rel_prefixes_p]
                agg_idx_prefix_rel_f = [i.index(self.rel_tokenizer.bos_token_id) for i in rel_prefixes_f]
            else:
                agg_idx_prefix_rel_p, agg_idx_prefix_rel_f = None, None
            rel_labels_p = [prefixes_p + label for prefixes_p, label in zip(rel_prefixes_p, rel_labels)]
            rel_labels_f = [prefixes_f + non_rel_labels for prefixes_f in rel_prefixes_f]
            
            rel_inputs_p = torch.tensor(self.sequence_padding(rel_labels_p, value=self.rel_tokenizer.pad_token_id))
            rel_inputs_f = torch.tensor(self.sequence_padding(rel_labels_f, value=self.rel_tokenizer.pad_token_id))
            rel_labels_p = torch.tensor(self.sequence_padding(rel_labels_p, value=-100))
            rel_labels_f = torch.tensor(self.sequence_padding(rel_labels_f, value=-100))
            
            for i in range(len(rel_labels_p)):
                rel_labels_p[i][:rel_prefix_lens_p[i]] = -100
            for i in range(len(rel_labels_f)):
                rel_labels_f[i][:rel_prefix_lens_f[i]] = -100
            rel_labels_p = torch.cat(
                (rel_labels_p[:, 1:], torch.ones((rel_labels_p.shape[0], 1), dtype=ent_labels_p.dtype) * -100), dim=-1)
            rel_labels_f = torch.cat(
                (rel_labels_f[:, 1:], torch.ones((rel_labels_f.shape[0], 1), dtype=ent_labels_p.dtype) * -100), dim=-1)
        
        return batch_input_ids, batch_input_mask, \
               ent_labels_p, ent_labels_f, rel_labels_p, rel_labels_f, ent_inputs_p, ent_inputs_f, rel_inputs_p, rel_inputs_f, \
               ent_pos, ent_pos_neg, rel_pos_p, rel_pos_f, \
               agg_idx_prefix_rel_p, agg_idx_prefix_rel_f, agg_idx_prefix_ent_p, agg_idx_prefix_ent_f
    
    """
     ent_pos, rel_pos_p, rel_pos_f  # 位置信息，用于模型聚合，提供位置信息
     ent_prefix_lens_p, rel_prefix_lens_p, rel_prefix_lens_f  # 前缀的长度，在 inference 时，根据这个数值进行截断输入，并且 matrix 时用于截断输出序列
    """


if __name__ == "__main__":
    a = {
        "text": "Cassytha pubescens is a native Australian hemiparasitic vine species, in the Laurel family. Common names for the species include devils twine, dodder-laurel, spilled devil\u2019s twine, snotty gobble or downy dodder-laurel. It is a widespread and common species in south eastern Australia. The species was first formally described in 1810 by the Scottish botanist Robert Brown in \"Prodromus Flora Novae Hollandiae et Insulae Van Diemen\" (Prodromus of the Flora of New Holland and Van Diemen\u2019s Land). Leaves are reduced to scales and photosynthesis is achieved through chlorophyll contained in the plants stems. Stems are between 0.5mm and 1.5mm in diameter and the haustoria are between 2 and 3\u00a0mm long.\n\n\n\"Cassytha pubescens\" is often compared with the genus Cuscuta (Convolvulaceae) due to similarities in their morphology and herbaceous parasitic habit.",
        "relation_list": [{"subject": "rodromus Flora Novae Hollandiae et Insulae Van Diemen", "object": "Robert Brown",
                           "subj_char_span": [377, 430], "obj_char_span": [359, 371], "predicate": "author"},
                          {"subject": "rodromus Flora Novae Hollandiae et Insulae Van Diemen", "object": "1810",
                           "subj_char_span": [377, 430], "obj_char_span": [329, 333], "predicate": "inception"}],
        "entity_list": [{"text": "hemiparasitic", "type": "DEFAULT", "char_span": [42, 55]},
                        {"text": "Laurel family", "type": "DEFAULT", "char_span": [77, 90]},
                        {"text": "Robert Brown", "type": "DEFAULT", "char_span": [359, 371]},
                        {"text": "rodromus Flora Novae Hollandiae et Insulae Van Diemen", "type": "DEFAULT",
                         "char_span": [377, 430]}, {"text": "Cuscuta", "type": "DEFAULT", "char_span": [755, 762]},
                        {"text": "Cassytha pubescens", "type": "DEFAULT", "char_span": [0, 18]},
                        {"text": "Cassytha pubescens", "type": "DEFAULT", "char_span": [702, 720]},
                        {"text": "1810", "type": "DEFAULT", "char_span": [329, 333]},
                        {"text": "0.5", "type": "DEFAULT", "char_span": [624, 627]},
                        {"text": "1.5", "type": "DEFAULT", "char_span": [634, 637]},
                        {"text": "2", "type": "DEFAULT", "char_span": [682, 683]},
                        {"text": "3", "type": "DEFAULT", "char_span": [688, 689]}]}
    
    tknz = T5TokenizerFast.from_pretrained('t5-small')
    
    tknz.add_special_tokens(
        {"additional_special_tokens": [AddedToken("."), AddedToken("-"), AddedToken(";"), AddedToken(","),
                                       AddedToken("("), AddedToken(")"), AddedToken("<"), AddedToken(">"),
                                       AddedToken("*")]}
    )
    split_sample_list = split_into_short_samples(tknz, a, 128, sliding_len=50, encoder="t5")
    
    import argparse
    
    ns = argparse.Namespace(**{"max_seq_len": 128})
    
    for i in split_sample_list:
        convert_example_to_features(i, tknz, ns, tknz, tknz)

"""


def read_and_load_data(og_path, args, mode, encoder_tokenizer, ent_tokenizer, rel_tokenizer):
    # if os.path.exists(f"cache/{mode}_{args.way_num}_{args.shot_num}.pth"):
    #     return torch.load(f"cache/{mode}_{args.way_num}_{args.shot_num}.pth")
    
    cache_path = os.path.join(og_path, args.dataset_path,
                              f"{encoder_tokenizer.name_or_path.split('/')[-1]}_{args.max_seq_len}_cache_{mode}_data.txt")
    cache_num_path = os.path.join(og_path, args.dataset_path, f"num_cache_data.txt")
    
    cache_num_dict = json.load(open(cache_num_path, "r")) if os.path.exists(cache_num_path) else {}
    
    if os.path.exists(cache_path):
        logging.info(f"Reload data from cache {cache_path} , instances number is {cache_num_dict[mode]}")
        return open(cache_path, "r")
    
    cache_path_writer = open(cache_path, "w")
    data = json.load(open(os.path.join(og_path, args.dataset_path, f"{mode}_data.json"), "r"))
    # data = json.load(open(os.path.join(og_path, args.dataset_path, f"{mode}_data.json"), "r"))[:20]  # for test
    
    spit_data_num = 0
    new_sample_list = []
    total_n = len(data)
    
    bad_token_ins_num = 0
    out_boundary_ins_num = 0
    input_f = 0
    
    for i, instance_dict in enumerate(data):
        
        if i % 10000 == 0: logger.info(f" {i} / {total_n}")
        
        #  TODO: 首先进行切分，将长度大于 max seq len 的数据拆分成为多个其他数据
        split_sample_list = split_into_short_samples(encoder_tokenizer, instance_dict, max_seq_len=args.max_seq_len - 2,
                                                     sliding_len=100)
        if len(split_sample_list) > 1: spit_data_num += 1
        
        for split_sample in split_sample_list:
            # TODO：整合数据特征
            fea = convert_example_to_features(split_sample, encoder_tokenizer, args, ent_tokenizer, rel_tokenizer)
            if type(fea) == dict:
                input_f += 1
                cache_path_writer.write(json.dumps(fea) + "\n")
                # input_features.append(fea)  # 越界情况跳过
            elif fea == "bad case token":
                bad_token_ins_num += 1
            elif fea == "no ent / no rel":
                out_boundary_ins_num += 1
    
    cache_num_dict[mode] = input_f
    json.dump(cache_num_dict, open(cache_num_path, "w"))
    logger.info(
        f"Get totally {input_f} instances from file : {mode}_data.json \n"
        f"\t original instances num is : {len(data)} , after split is: {len(new_sample_list)}, split instance number is: {spit_data_num}\n"
        f"\t omit cases {bad_token_ins_num}, oo-boudnary cases {out_boundary_ins_num}]")
    
    return open(cache_path, "r")


"""
