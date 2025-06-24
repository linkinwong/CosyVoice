import sys
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "5,6"
import logging
from cosyvoice.cli.cosyvoice import CosyVoice, CosyVoice2
from cosyvoice.utils.file_utils import load_wav
import torchaudio
import torch
import argparse
import time
import numpy as np
from stream_player import StreamPlayer
from funasr import AutoModel
import glob
from pathlib import Path

# 设置根目录并添加第三方库路径
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append("{}/third_party/Matcha-TTS".format(ROOT_DIR))


# 设置日志级别为 DEBUG
logging.basicConfig(
    level=logging.DEBUG, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logging.getLogger().setLevel(logging.DEBUG)

# 确保设置影响所有模块
for name in logging.root.manager.loggerDict:
    logging.getLogger(name).setLevel(logging.DEBUG)


# 添加命令行参数解析
parser = argparse.ArgumentParser(description="CosyVoice2 for wake word")
parser.add_argument(
    "--model_dir",
    type=str,
    default="pretrained_models/CosyVoice2-0.5B",
    help="模型目录路径",
)
parser.add_argument(
    "--fp16", action="store_true", default=False, help="是否使用半精度(fp16)推理"
)
parser.add_argument(
    "--use_flow_cache", action="store_true", default=False, help="是否使用流式缓存"
)
parser.add_argument(
    "--log_level", type=str, default="INFO", help="日志级别"
)

args = parser.parse_args()

print(f"使用模型目录: {args.model_dir}")
cosyvoice = CosyVoice2(
    args.model_dir,
    load_jit=False,
    load_trt=True,
    fp16=args.fp16,
    use_flow_cache=args.use_flow_cache,
)

logging.info(f"cosyvoice.list_available_spks(): {cosyvoice.list_available_spks()}")


model_dir = "iic/SenseVoiceSmall"
asr_model = AutoModel(
    model=model_dir,
    disable_update=True,
    log_level=args.log_level,
    device="cuda")
    # device="cuda:0")

def prompt_wav_recognition(prompt_wav):
    """音频识别文本"""
    if prompt_wav is None:
        return ''
        
    try:
        res = asr_model.generate(
            input=prompt_wav,
            language="auto",  # "zn", "en", "yue", "ja", "ko", "nospeech"
            use_itn=True,
        )
        return res[0]["text"].split('|>')[-1]
    except Exception as e:
        logging.error(f"音频识别文本失败: {e}")
        return ''

wake_words = ["你好小灵","你好小七", "嗨, 小七", "嗨,小灵", "嘿,小七",  "嘿,小灵", "小七小七", "小灵小灵"] 

def process_audio_files_and_save_speakers():
    """
    功能1: 从audios文件夹中的音频文件提取音色并保存
    """
    print("开始处理音频文件并保存音色...")
    
    # 获取audios文件夹中的所有音频文件
    audio_folder = "./audios"
    if not os.path.exists(audio_folder):
        print(f"音频文件夹 {audio_folder} 不存在，使用asset文件夹作为替代")
        audio_folder = "./asset"
    
    # 支持的音频格式
    audio_extensions = ['*.wav', '*.mp3', '*.flac', '*.m4a']
    audio_files = []
    for ext in audio_extensions:
        audio_files.extend(glob.glob(os.path.join(audio_folder, ext)))
    
    if not audio_files:
        print(f"在 {audio_folder} 文件夹中没有找到音频文件")
        return []
    
    print(f"找到 {len(audio_files)} 个音频文件")
    
    processed_speakers = []
    
    for i, audio_file in enumerate(audio_files, 1):
        try:
            print(f"处理第 {i} 个音频文件: {audio_file}")
            
            # 加载音频文件
            prompt_speech_16k = load_wav(audio_file, 16000)
            
            # 使用ASR识别文本
            recognized_text = prompt_wav_recognition(audio_file)
            
            if not recognized_text.strip():
                print(f"警告: 无法识别音频文件 {audio_file} 的文本，跳过...")
                continue
            
            print(f"识别到的文本: {recognized_text}")
            
            # 创建speaker_id
            speaker_id = str(i)
            
            # 添加zero shot speaker
            success = cosyvoice.add_zero_shot_spk(
                recognized_text,
                prompt_speech_16k,
                speaker_id
            )
            
            if success:
                print(f"成功添加音色 {speaker_id}")
                processed_speakers.append({
                    'speaker_id': speaker_id,
                    'audio_file': audio_file,
                    'text': recognized_text
                })
            else:
                print(f"添加音色 {speaker_id} 失败")
                
        except Exception as e:
            print(f"处理音频文件 {audio_file} 时出错: {e}")
            continue
    
    # 保存speaker信息
    if processed_speakers:
        cosyvoice.save_spkinfo()
        print(f"成功处理了 {len(processed_speakers)} 个音色")
    
    return processed_speakers


    
def generate_wake_word_audios(processed_speakers):
    """
    功能2: 使用保存的音色为每个唤醒词生成音频（非流式生成）
    """
    print("开始生成唤醒词音频...")
    
    # 创建输出目录
    output_dir = "./wake_word_audio"
    os.makedirs(output_dir, exist_ok=True)
    
    total_generated = 0
    
    for speaker_info in processed_speakers:
        speaker_id = speaker_info['speaker_id']
        print(f"使用音色 {speaker_id} 生成唤醒词音频...")
        
        for wake_word in wake_words:
            try:
                print(f"生成: {speaker_id} + {wake_word}")
                
                # 使用非流式方式生成音频（一次性生成）
                result = next(cosyvoice.inference_sft(
                    wake_word,
                    speaker_id,
                    stream=False  # 设置为非流式生成
                ))
                
                # 获取生成的音频
                audio = result["tts_speech"]
                
                # 生成文件名
                filename = f"{speaker_id}_{wake_word.replace(',', '').replace(' ', '_')}.wav"
                filepath = os.path.join(output_dir, filename)
                
                # 保存音频文件
                torchaudio.save(filepath, audio, cosyvoice.sample_rate)
                print(f"保存音频: {filepath}")
                total_generated += 1
                
            except Exception as e:
                print(f"生成唤醒词音频失败 {speaker_id} + {wake_word}: {e}")
                continue
    
    print(f"总共生成了 {total_generated} 个唤醒词音频文件")


def generate_wake_word_audios_for_existing_speakers():
    """
    功能3: 使用预训练模型中已有的音色为每个唤醒词生成音频
    """
    print("开始为预训练模型中的已有音色生成唤醒词音频...")
    
    # 创建输出目录
    output_dir = "./wake_word_audio"
    os.makedirs(output_dir, exist_ok=True)
    
    total_generated = 0
    
    try:
        # 加载预训练模型中的音色信息
        spk_info_path = os.path.join(cosyvoice.model_dir, "spk2info.pt")
        if not os.path.exists(spk_info_path):
            print(f"找不到预训练音色信息文件: {spk_info_path}")
            return 0
            
        spk_info = torch.load(spk_info_path)
        
        # 获取所有非数字开头的音色ID
        existing_speakers = [spk_id for spk_id in spk_info.keys() if not spk_id[0].isdigit()]
        
        print(f"找到 {len(existing_speakers)} 个预训练音色")
        
        for speaker_id in existing_speakers:
            print(f"使用预训练音色 {speaker_id} 生成唤醒词音频...")
            
            for wake_word in wake_words:
                try:
                    print(f"生成: {speaker_id} + {wake_word}")
                    
                    # 使用非流式方式生成音频（一次性生成）
                    result = next(cosyvoice.inference_sft(
                        wake_word,
                        speaker_id,
                        stream=False  # 设置为非流式生成
                    ))
                    
                    # 获取生成的音频
                    audio = result["tts_speech"]
                    
                    # 生成文件名
                    filename = f"{speaker_id}_{wake_word.replace(',', '').replace(' ', '_')}.wav"
                    filepath = os.path.join(output_dir, filename)
                    
                    # 保存音频文件
                    torchaudio.save(filepath, audio, cosyvoice.sample_rate)
                    print(f"保存音频: {filepath}")
                    total_generated += 1
                    
                except Exception as e:
                    print(f"生成唤醒词音频失败 {speaker_id} + {wake_word}: {e}")
                    continue
                    
    except Exception as e:
        print(f"处理预训练音色时出错: {e}")
    
    print(f"总共为预训练音色生成了 {total_generated} 个唤醒词音频文件")
    return total_generated


def main():
    """主函数"""
    print("开始音色提取和唤醒词音频生成流程...")
    
    # # 功能1: 处理音频文件并保存音色
    # processed_speakers = process_audio_files_and_save_speakers()
    
    # if not processed_speakers:
    #     print("没有成功处理任何音色，程序退出")
    #     return
    
    # # 功能2: 生成唤醒词音频
    # generate_wake_word_audios(processed_speakers)
    
    # 功能3: 使用预训练模型中已有的音色为每个唤醒词生成音频
    generate_wake_word_audios_for_existing_speakers()

    print("所有任务完成!")

# 运行主函数
if __name__ == "__main__":
    main()



# def generate_wake_word_audios_stream(processed_speakers):
#     """
#     功能2: 使用保存的音色为每个唤醒词生成音频
#     """
#     print("开始生成唤醒词音频...")
    
#     # 创建输出目录
#     output_dir = "./wake_word_audio"
#     os.makedirs(output_dir, exist_ok=True)
    
#     total_generated = 0
    
#     for speaker_info in processed_speakers:
#         speaker_id = speaker_info['speaker_id']
#         print(f"使用音色 {speaker_id} 生成唤醒词音频...")
        
#         for wake_word in wake_words:
#             try:
#                 print(f"生成: {speaker_id} + {wake_word}")
                
#                 # 使用inference_sft生成音频
#                 audio_chunks = []
#                 for i, j in enumerate(
#                     cosyvoice.inference_sft(
#                         wake_word,
#                         speaker_id,
#                         stream=args.use_flow_cache,
#                     )
#                 ):
#                     audio_chunks.append(j["tts_speech"])
                
#                 # 合并所有音频块
#                 if audio_chunks:
#                     full_audio = torch.cat(audio_chunks, dim=-1)
                    
#                     # 生成文件名
#                     filename = f"{speaker_id}_{wake_word.replace(',', '').replace(' ', '_')}.wav"
#                     filepath = os.path.join(output_dir, filename)
                    
#                     # 保存音频文件
#                     torchaudio.save(filepath, full_audio, cosyvoice.sample_rate)
#                     print(f"保存音频: {filepath}")
#                     total_generated += 1
                
#             except Exception as e:
#                 print(f"生成唤醒词音频失败 {speaker_id} + {wake_word}: {e}")
#                 continue
    
#     print(f"总共生成了 {total_generated} 个唤醒词音频文件")



# ================================= 注释掉原有的代码 =================================
# save zero_shot spk for future usage
# 修复：添加prompt_speech_16k变量定义
# prompt_speech_16k = load_wav("./asset/harry_potter_snape_injured.wav", 16000)
# assert (
#     cosyvoice.add_zero_shot_spk(
#         # "声纹识别能力，多测一些", prompt_speech_16k, "wll"
#         # '明天是星期六啦，我要去上果粒课啦，你们知道吗？', prompt_speech_16k, "wzy"
#         # "啊这个也能理解啊，因为七牛毕竟，是国内最早做云存储的公司。嗯，所以我想，就是和云存储相关的交流，可以在这个这个会之后自由讨论的时候，我们只管沟通啊。知无不言，言无不尽，哼哼。", prompt_speech_16k, "laoxu"
#         # "我最喜欢夏天，满地的鲜花，这里一朵，那里一朵， 真比天上的星星还多。 夜晚，我数着天上的星星，真比地上的花儿还要多。那里一颗，真比天上的花还，花儿还多。",
#         "I'm not hungry. That explains the blood. Listen. Last night, I'm guessing Snape let the troll in as a diversion, so he could get past that dog. But he got bit, that's why he's limping. The day I was at Gringotts, Hagrid took something out of the vault. Said it was Hogwarts business, very secret. That's what the dog's guarding. That's want Snape wants.  I never get mail.",
#         prompt_speech_16k,
#         "hp",
#     )
#     is True
# )
# cosyvoice.save_spkinfo()


# player = StreamPlayer(sample_rate=cosyvoice.sample_rate, channels=1, block_size=18048)
# player.start()


# print(
#     "\n按回车使用默认文本，输入新文本后回车使用新文本，输入q后回车退出, 输入@后回车使用新指令\n"
# )


# while True:
#     # 交互式循环，可以反复输入文本生成语音
#     # speaker = "xiaoluo_mandarin"
#     # speaker = "Donald J. Trump"
#     # default_tts_text = "CosyVoice迎来全面升级，提供更准、更稳、更快、 更好的语音生成能力。make america great again. "
#     default_speaker = "hp"
#     default_tts_text = "从此每当害怕时，他就想起那个和伙伴共同编织星光的夜晚 [noise] ，勇气便像萤火虫般在心底亮起来。"
#     default_instruct_text = "用很慢的语速读这个故事"
#     speaker = default_speaker
#     tts_text = default_tts_text
#     instruct_text = default_instruct_text
#     # 获取用户输入
#     user_input = input(
#         f"请输入文本 (格式: ' speaker @ tts_text @ instruct_text')  退出: q "
#     )

#     # 检查是否退出
#     if user_input.strip() == "q":
#         print("退出语音生成循环")
#         break

#     if len(user_input) > 1:
#         speaker = user_input.split("@")[0]
#     if len(user_input.split("@")) > 1:
#         speaker = user_input.split("@")[0]
#         tts_text = user_input.split("@")[1]
#     if len(user_input.split("@")) > 2:
#         speaker = user_input.split("@")[0]
#         tts_text = user_input.split("@")[1]
#         instruct_text = user_input.split("@")[2]

#     print(f"SPEAKER 是： {speaker}， tts_text 是： {tts_text}")
#     start_time = time.time()
#     for i, j in enumerate(
#         # cosyvoice.inference_instruct2(
#         #     tts_text,
#         #     instruct_text,
#         #     prompt_speech_16k,
#         #     stream=True,
#         #     speed=0.8,
#         #     text_frontend=True,
#         # )
#         cosyvoice.inference_sft(
#             tts_text,
#             speaker,
#             stream=args.use_flow_cache,
#         )
#     ):
#         current_time = time.time()
#         # logging.info(f"第 {i} 次生成耗时: {current_time - start_time:.2f} 秒")
#         start_time = current_time

#         # torchaudio.save(
#         #     "sft_{}.wav".format(i), j["tts_speech"], cosyvoice.sample_rate
#         # )
#         player.play(j["tts_speech"].numpy().T)

# # 停止播放器
# player.stop()

# # 最后一个示例，保存到文件而不是播放
# start_time = time.time()
# for i, j in enumerate(
#     cosyvoice.inference_zero_shot(
#         # "这句话里面到底在使用了谁的语音呢？",
#         "CosyVoice迎来全面升级，提供更准、更稳、更快、 更好的语音生成能力。make america great again. ",
#         "我会把三段话切成3段，用来做",
#         prompt_speech_16k,
#         stream=True,
#     )
# ):
#     current_time = time.time()
#     logging.info(f"第 {i} 次生成耗时: {current_time - start_time:.2f} 秒")
#     start_time = current_time
#     torchaudio.save(
#         "zero_shot_{}.wav".format(i), j["tts_speech"], cosyvoice.sample_rate
#     )

# # instruct usage
# for i, j in enumerate(
#     cosyvoice.inference_instruct2(
#         "收到好友从远方寄来的生日礼物，那份意外的惊喜与深深的祝福让我心中充满了甜蜜的快乐，笑容如花儿般绽放。",
#         "用四川话说这句话",
#         prompt_speech_16k,
#         stream=True,
#     )
# ):
#     torchaudio.save("instruct_{}.wav".format(i), j["tts_speech"], cosyvoice.sample_rate)

# # 流式生成并添加到播放队列
# for i, j in enumerate(
#     cosyvoice.inference_zero_shot(
#         # "这句话里面到底在使用了谁的语音呢？",
#         "CosyVoice迎来全面升级，提供更准、更稳、更快、 更好的语音生成能力。make america great again. ",
#         "我会把三段话切成3段，用来做",
#         prompt_speech_16k,
#         stream=True,
#     )
# ):
#     current_time = time.time()
#     logging.info(f"第 {i} 次生成耗时: {current_time - start_time:.2f} 秒")
#     start_time = current_time

#     player.play(j["tts_speech"].numpy().T)


# # prompt_speech_16k = load_wav("./asset/sqr3.wav", 16000)
# # prompt_speech_16k = load_wav("./asset/wll3.wav", 16000)
# # prompt_speech_16k = load_wav("./asset/wzy_read_poet_27s.wav", 16000)
# prompt_speech_16k = load_wav("./asset/harry_potter_snape_injured.wav", 16000)
# # prompt_speech_16k = load_wav("./asset/laoxu.wav", 16000)
# for i, j in enumerate(
#     cosyvoice.inference_zero_shot(
#         "收到好友从远方寄来的生日礼物，那份意外的惊喜与深深的祝福让我心中充满了甜蜜的快乐，笑容如花儿般绽放。",
#         # "声纹识别能力，多测一些",
#         # '明天是星期六啦，我要去上果粒课啦，你们知道吗？',
#         "I'm not hungry. That explains the blood. Listen. Last night, I'm guessing Snape let the troll in as a diversion, so he could get past that dog. But he got bit, that's why he's limping. The day I was at Gringotts, Hagrid took something out of the vault. Said it was Hogwarts business, very secret. That's what the dog's guarding. That's what Snape wants.  I never get mail.",
#         # "啊这个也能理解啊，因为七牛毕竟，是国内最早做云存储的公司。嗯，所以我想，就是和云存储相关的交流，可以在这个这个会之后自由讨论的时候，我们只管沟通啊。知无不言，言无不尽，哼哼。",
#         # "我最喜欢夏天，满地的鲜花，这里一朵，那里一朵， 真比天上的星星还多。 夜晚，我数着天上的星星，真比地上的花儿还要多。那里一颗，真比天上的花还，花儿还多。",
#         prompt_speech_16k,
#         stream=args.use_flow_cache,
#     )
# ):
#     torchaudio.save(
#         "zero_shot_{}.wav".format(i), j["tts_speech"], cosyvoice.sample_rate
#     )