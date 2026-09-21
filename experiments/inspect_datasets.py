import gzip
import json
import os
import fastparquet

print("=== 1. OASST1 ===")
p_oasst = r'C:\Users\elijk\.cache\huggingface\hub\datasets--OpenAssistant--oasst1\snapshots\fdf72ae0827c1cda404aff25b6603abec9e3399b\2023-04-12_oasst_ready.messages.jsonl.gz'
en_pairs = 0
trees = set()
with gzip.open(p_oasst, 'rt', encoding='utf-8') as f:
    for line in f:
        m = json.loads(line)
        if m.get('lang') == 'en':
            trees.add(m['message_tree_id'])
            if m.get('role') == 'assistant':
                en_pairs += 1
print(f"OASST1 EN assistant replies: {en_pairs}, distinct trees: {len(trees)}")

print("=== 2. WildChat Shard 0 ===")
p_wild = r'C:\Users\elijk\.cache\huggingface\hub\datasets--allenai--WildChat\snapshots\f66566ceaaeb619dd98ffb0f3bf3ce1f86775ac4\data\train-00000-of-00006.parquet'
pf = fastparquet.ParquetFile(p_wild)
df = pf.to_pandas(columns=['conversation_id', 'model', 'language', 'conversation'])
en_wild = df[df['language'] == 'English']
print(f"WildChat shard 0 total: {len(df)}, English: {len(en_wild)}")
sample_conv = en_wild.iloc[0]['conversation']
roles = [x.get('role') for x in sample_conv]
print(f"Sample WildChat conv turns: {len(sample_conv)}, roles: {roles}")

print("=== 3. CodeAlpaca 20k ===")
p_code = r'C:\Users\elijk\.cache\huggingface\hub\datasets--sahil2801--CodeAlpaca-20k\snapshots\06b24d77641031b2ecda7cfca3b567d12f3bc173\code_alpaca_20k.json'
with open(p_code, 'r', encoding='utf-8') as f:
    code_data = json.load(f)
print(f"CodeAlpaca total items: {len(code_data)}")

print("=== 4. GSM8K ===")
p_gsm_test = r'C:\Users\elijk\.cache\huggingface\hub\datasets--openai--gsm8k\snapshots\cd9e71e21b764268e370a241e17d23d853e34b7f\main\test-00000-of-00001.parquet'
p_gsm_train = r'C:\Users\elijk\.cache\huggingface\hub\datasets--openai--gsm8k\snapshots\cd9e71e21b764268e370a241e17d23d853e34b7f\main\train-00000-of-00001.parquet'
pf_test = fastparquet.ParquetFile(p_gsm_test)
pf_train = fastparquet.ParquetFile(p_gsm_train)
print(f"GSM8K train: {pf_train.count()}, test: {pf_test.count()}, total: {pf_train.count() + pf_test.count()}")

print("=== 5. MBPP ===")
mbpp_count = sum(1 for line in open('data/train.jsonl', encoding='utf-8') if json.loads(line).get('domain') == 'code')
mbpp_test_count = sum(1 for line in open('data/test.jsonl', encoding='utf-8') if json.loads(line).get('domain') == 'code')
mbpp_val_count = sum(1 for line in open('data/val.jsonl', encoding='utf-8') if json.loads(line).get('domain') == 'code')
print(f"MBPP existing across splits: {mbpp_count + mbpp_test_count + mbpp_val_count}")
