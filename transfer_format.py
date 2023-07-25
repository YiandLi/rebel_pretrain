import os, json
from tqdm import tqdm
from collections import defaultdict

file_map = [
    ('en_test.jsonl', 'test_data.json'),
    ('en_train.jsonl', 'train_data.json'),
    ('en_val.jsonl', 'valid_data.json')
]

dir = "rebel_star"

if not os.path.exists(dir): os.mkdir(dir)

rel_labels = {}

for (file, out) in file_map:
    writer = open(os.path.join(dir, out), "w")
    output_data = []
    rel_label_set = defaultdict(int)
    
    reader = open(file, "r")
    
    total_seq_len = 0
    
    for line in tqdm(reader, desc=file + ":"):
        line = json.loads(line)
        
        entity_list = []
        for ent in line['entities']:
            entity_list.append(
                {"text": ent['surfaceform'],
                 "type": 'DEFAULT',
                 "char_span": ent['boundaries'],
                 }
            )
        
        relation_list = []
        for rel in line['triples']:
            subject = rel['subject']['surfaceform']
            object = rel['object']['surfaceform']
            subj_char_span = rel['subject']['boundaries']
            obj_char_span = rel['object']['boundaries']
            predicate = rel['predicate']['surfaceform']
            rel_label_set[predicate] += 1
            
            relation_list.append({
                "subject": subject,
                "object": object,
                "subj_char_span": subj_char_span,
                "obj_char_span": obj_char_span,
                "predicate": predicate
            })
        
        # print(line)
        
        if relation_list:
            output_data.append({
                "text": line["text"],
                "relation_list": relation_list,
                "entity_list": entity_list
            })
        
            total_seq_len += len(line["text"])
    
    rel_labels[file] = rel_label_set
    print(
        f"file: {file} , instance num : {len(output_data)} , average sequence length : {total_seq_len / len(output_data) : .3f}")
    json.dump(output_data, writer)

json.dump(rel_labels, open(os.path.join(dir, "rel_labels_count.json"), "w"))
