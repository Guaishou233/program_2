from datasets import load_dataset

# 下载并保存数据集
dataset = load_dataset("EdinburghNLP/xsum")
dataset.save_to_disk('/home/tangqiansong/program_2/data/xsum_dataset')