import onnxruntime as ort

# 检查CUDA是否可用
providers = ort.get_available_providers()
print("Available providers:", providers)

# 检查CUDA版本
if 'CUDAExecutionProvider' in providers:
    sess_options = ort.SessionOptions()
    cuda_provider_options = {
        'device_id': 0,
        'arena_extend_strategy': 'kNextPowerOfTwo',
        'cudnn_conv_algo_search': 'EXHAUSTIVE'
    }
    
    # 创建一个使用CUDA的会话
    try:
        sess = ort.InferenceSession("/disk1/wanglinlin/ros/voice_user_interface/speech_synthesis/CosyVoice/pretrained_models/CosyVoice2-0.5B/flow.decoder.estimator.fp32.onnx", 
                                  sess_options=sess_options, 
                                  providers=[('CUDAExecutionProvider', cuda_provider_options)])
        print("CUDA兼容性测试通过")
    except Exception as e:
        print(f"CUDA兼容性测试失败: {e}")
else:
    print("CUDAExecutionProvider 不可用")
