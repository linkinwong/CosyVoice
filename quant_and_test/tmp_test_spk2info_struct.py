import torch
import matplotlib.pyplot as plt
import numpy as np
import os
from cosyvoice.utils.file_utils import load_wav

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
# 加载spk2info文件
spk2info_path = os.path.join(ROOT_DIR, 'pretrained_models/CosyVoice2-0.5B/spk2info.pt')
spk2info = torch.load(spk2info_path)

# 打印基本信息
print(f"文件类型: {type(spk2info)}")
print(f"说话人数量: {len(spk2info)}")
print(f"说话人ID列表: {list(spk2info.keys())}")

# 查看第一个说话人的详细信息
first_spk = next(iter(spk2info.keys()))
print(f"\n第一个说话人ID: {first_spk}")
print(f"包含的数据类型: {[k for k in spk2info[first_spk].keys()]}")

# 打印嵌入向量的形状
for spk_id in spk2info:
    print(f"\n说话人 {spk_id} 的信息:")
    for k, v in spk2info[spk_id].items():
        if isinstance(v, torch.Tensor):
            print(f"  - {k} 形状: {v.shape}, 类型: {v.dtype}")

# # 可视化embedding向量 (可选)
# plt.figure(figsize=(12, 8))
# for i, spk_id in enumerate(spk2info):
#     emb = spk2info[spk_id]['embedding'].cpu().numpy().flatten()
#     plt.plot(emb, label=spk_id)
# plt.legend()
# plt.title("不同说话人的Embedding向量对比")
# plt.savefig('speaker_embeddings.png')
# print("\n已保存说话人embedding可视化图像到'speaker_embeddings.png'")


prompt_speech_16k = load_wav("./asset/HarryPorter.wav", 16000)

# 打印prompt_speech_16k的形状
print(f"prompt_speech_16k 的形状: {prompt_speech_16k.shape}")

# 打印prompt_speech_16k的类型
print(f"prompt_speech_16k 的类型: {type(prompt_speech_16k)}")

