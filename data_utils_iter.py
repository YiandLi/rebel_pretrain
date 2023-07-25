import json
from itertools import islice

import numpy as np
import torch
from torch.distributed import get_rank, get_world_size
from torch.utils.data import IterableDataset, get_worker_info


def print_on_rank(string):
    rank = get_rank()
    print(f"[rank={rank}]{string}")


class MyIterableDataset(IterableDataset):
    """
    https://colab.research.google.com/drive/1OFLZnX9y5QUFNONuvFsxOizq4M-tFvk-?usp=sharing
    https://github.com/Lightning-AI/lightning/issues/15734
    """
    
    def __init__(self, path, args, encoder_tokenizer, ent_tokenizer, rel_tokenizer, ins_num, mode):
        
        super(MyIterableDataset, self).__init__()
        self.path = path  # 这里设置所有待读取的文件的目录
        self.args = args
        self.mode = mode
        self.encoder_tokenizer = encoder_tokenizer
        self.ent_tokenizer = ent_tokenizer
        self.rel_tokenizer = rel_tokenizer
        self.ins_num = ins_num
        
    # def parse_file(self):
    #     with open(self.path, "r") as file_obj:
    #         for line in file_obj:  # 逐行读取
    #             line = json.loads(line.strip())
    #             yield line
    
    def __iter__(self):
        world_size = get_world_size()
        process_rank = get_rank()
        worker_info = get_worker_info()
        num_workers = worker_info.num_workers if worker_info is not None else 1
        worker_id = worker_info.id if worker_info is not None else 0
        
        file_itr = open(self.path)
        mapped_itr = map(lambda x: json.loads(x.strip()), file_itr)
        # Solution  : https://stackoverflow.com/questions/69778356/iterable-pytorch-dataset-with-multiple-workers
        step_itr = islice(mapped_itr, worker_id + (num_workers * process_rank),
                          None, (num_workers * world_size))  # wrap the iterator
        return step_itr
        
        # sampler = DistributedSampler(self, num_replicas=(num_workers * world_size),
        #                              rank=(process_rank * num_workers + worker_id), shuffle=False)
        # for i in iter(sampler):
        #     yield i
    
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
        sentence_ent_default_flags = []  # sentence level!
        
        for idx, _feature in enumerate(features):
            encoder_txt = self.encoder_tokenizer.encode_plus(_feature['text'], max_length=self.args.max_seq_len,
                                                             truncation=True)
            batch_input_ids.append(encoder_txt["input_ids"])
            batch_input_mask.append(encoder_txt["attention_mask"])
            
            ent_tuples = list(_feature['ent_infos'].keys())
            ent_values = list(_feature['ent_infos'].values())
            
            sentence_ent_default_flags.append(any([bool('DEFAULT' in i) for i in ent_values]))
            
            ent_tuples = [m.split("\t") for m in ent_tuples]
            ent_tuples = [((int(i), int(j)), k) for i, j, k in ent_tuples]
            
            ent_labels.append(dict(zip(ent_tuples, ent_values)))
            
            rel_infos_keys = [i.split("\t") for i in list(_feature['rel_infos'].keys())]
            rel_infos_keys = [(int(j) for j in i) for i in rel_infos_keys]
            
            rel_labels.append(dict(zip(rel_infos_keys, list(_feature['rel_infos'].values()))))
            texts.append(_feature['text'])
        
        # = [not bool('DEFAULT' in i) for j in ent_labels for i in j.values()]  # 是否有实体类型的 标记
        batch_input_ids = torch.tensor(self.sequence_padding(batch_input_ids)).long()
        batch_input_mask = torch.tensor(self.sequence_padding(batch_input_mask)).float()
        
        return batch_input_ids, batch_input_mask, ent_labels, rel_labels, texts, sentence_ent_default_flags
    
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
            ent_tuples = [m.split("\t") for m in ent_tuples]
            ent_tuples = [((int(i), int(j)), k) for i, j, k in ent_tuples]
            ent_pos_tuples = [i[0] for i in ent_tuples]
            ent_txt_tuples = [i[1] for i in ent_tuples]
            ent_pos_tuples2id = {k: i for i, k in enumerate(ent_pos_tuples)}
            
            instance_txt.append(_feature['text'])
            ent_pos.append(ent_pos_tuples)  # instance level
            ent_txt.append(ent_txt_tuples)
            
            ent_pos_tuples_neg, ent_txt_tuples_neg = [], []
            if self.args.ent_negative_sample.lower() != "none" and \
                    not "DEFAULT </s>" in _feature['ent_infos'].values():  # 没有实体类型的样本，不需要计算实体 loss
                
                ent_pos_tuples_neg, ent_txt_tuples_neg = self.negative_sample_ent(encoder_txt["input_ids"],
                                                                                  ent_pos_tuples,
                                                                                  self.args.ent_negative_sample)
            
            ent_pos_neg.append(ent_pos_tuples_neg)
            
            rel_infos_keys = [i.split("\t") for i in list(_feature['rel_infos'].keys())]
            rel_infos_keys = [(int(j) for j in i) for i in rel_infos_keys]
            
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
            
            valid_ent_loss_flags = [not bool('DEFAULT' in i) for i in ent_labels]  # 是否有实体类型的 标记
            
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
               agg_idx_prefix_rel_p, agg_idx_prefix_rel_f, agg_idx_prefix_ent_p, agg_idx_prefix_ent_f, valid_ent_loss_flags
    
    """
     ent_pos, rel_pos_p, rel_pos_f  # 位置信息，用于模型聚合，提供位置信息
     ent_prefix_lens_p, rel_prefix_lens_p, rel_prefix_lens_f  # 前缀的长度，在 inference 时，根据这个数值进行截断输入，并且 matrix 时用于截断输出序列
    """
