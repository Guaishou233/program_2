from datasets import load_dataset

# 下载并保存数据集
dataset = load_dataset("Salesforce/wikitext","wikitext-103-v1")
dataset.save_to_disk('/home/tangqiansong/program_2/data/wikitext')