import json
import time
import numpy as np
import soundfile as sf
from tqdm import trange
import onnxruntime as ort
from pydub import AudioSegment
from vachana_g2p import th2ipa
from utils import chunk_text, load_model

DEVICE = "cuda" if "CUDAExecutionProvider" in ort.get_available_providers() else "cpu"
print(f"Device : {DEVICE}")

ROOT = "checkpoints"
MODELS = [f"{ROOT}/Preprocess.onnx", f"{ROOT}/Transformer.onnx", f"{ROOT}/Vocoder.onnx"]
SAMPLE_RATE, HOP_LENGTH, NFE_STEP = 24000, 256, 20

load_model()

with open(f"{ROOT}/vocab.json", "r", encoding="utf-8") as f:
    vocab_char_map = {k: int(v) for k, v in json.load(f)["vocab"].items()}

def list_str_to_idx(text, vocab_char_map):
    seqs = [np.array([vocab_char_map.get(c, 0) for c in t], dtype=np.int32) for t in text]
    max_len = max(len(s) for s in seqs)
    out = np.full((len(seqs), max_len), -1, dtype=np.int32)
    for i, s in enumerate(seqs):
        out[i, :len(s)] = s
    return out

def normalize_to_int16(audio):
    max_val = np.max(np.abs(audio))
    return (audio * (32767.0 / max_val if max_val > 0 else 1.0)).astype(np.int16)

session_opts = ort.SessionOptions()
session_opts.enable_cpu_mem_arena = True
session_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

providers = [{'device_id': 0, 'gpu_mem_limit': 4*1024**3}] if "CUDAExecutionProvider" else None
sessions = [
    ort.InferenceSession(MODELS[0], session_opts, providers=['CPUExecutionProvider']),
    ort.InferenceSession(MODELS[1], session_opts, providers=['CUDAExecutionProvider'], provider_options=providers),
    ort.InferenceSession(MODELS[2], session_opts, providers=['CPUExecutionProvider'])
]

def TTS(ref_audio, ref_text, gen_text, speed=1.0, output="generated.wav", verbose=False):
    """
    - ref_audio : ไฟล์เสียงอ้างอิง
    - ref_text : ข้อความอ้างอิงที่ตรงกับไฟล์เสียง
    - gen_text : ข้อความที่ต้องการสร้างเสียงพูด
    - speed : ความเร็วของเสียง (ช้า < 1.0 < เร็ว)
    - output : ไฟล์ปลายทางที่ต้องการบันทึกเสียง
    - verbose : True แสดงข้อมูล
    """
    speed = max(0.4, min(1.5, speed))
    audio = np.array(AudioSegment.from_file(ref_audio).set_channels(1).set_frame_rate(SAMPLE_RATE).get_array_of_samples(), dtype=np.float32)
    audio = normalize_to_int16(audio).reshape(1, 1, -1)

    start = time.time()
    ref_text_g2p = th2ipa(ref_text)
    batches = chunk_text(gen_text)
    if verbose:
        print("\n===== Generate TTS Info =====")
        print(f"Ref Text IPA : {ref_text_g2p}")
        print(f"Speed : {speed}")
        print(f"Chunks Text:")
        for i, gen_text in enumerate(batches):
            print(f"    Gen Text {i} :", gen_text)
        print("\n")

    generated_wavs = []

    for idx, t in enumerate(batches):
        gen_text = th2ipa(t)
        gen_text_len = (len(gen_text) - gen_text.count(" ")) * 5 + (gen_text.count(" ") * 2.5)
        max_duration = np.array([audio.shape[-1] // HOP_LENGTH + 1 + int(gen_text_len / speed)], dtype=np.int64)
        text_ids = list_str_to_idx([ref_text_g2p + ". " + gen_text], vocab_char_map)
        if verbose:
            print(f"Gen Text IPA {idx} : {gen_text}")

        outputs_A = sessions[0].run(None, {
            sessions[0].get_inputs()[0].name: audio,
            sessions[0].get_inputs()[1].name: text_ids,
            sessions[0].get_inputs()[2].name: max_duration
        })
        
        noise, time_step = outputs_A[0], np.array([0], dtype=np.int32)
        input_names_B = [inp.name for inp in sessions[1].get_inputs()]
        
        for i in trange(1, NFE_STEP + 1, desc=f"Step", disable=not verbose):
            noise, time_step = sessions[1].run(None, dict(zip(
                input_names_B,
                [noise] + list(outputs_A[1:7]) + [time_step]
            )))[:2]
        
        generated_signal = sessions[2].run(None, {
            sessions[2].get_inputs()[0].name: noise,
            sessions[2].get_inputs()[1].name: outputs_A[7]
        })[0]
        
        generated_wavs.append(generated_signal)

    sf.write(output, np.concatenate(generated_wavs, axis=-1).reshape(-1), SAMPLE_RATE)

    print(f"\nGeneration time: {time.time() - start:.3f}s")
