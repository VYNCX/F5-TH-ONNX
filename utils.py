import os
import re
from pythainlp.tokenize import word_tokenize

def load_model():
    model_dir = "./checkpoints"
    model_files = ["Preprocess.onnx", "Transformer.onnx", "Vocoder.onnx", "vocab.json"]
    model_paths = [os.path.join(model_dir, f) for f in model_files]
    if all(os.path.exists(f) for f in model_paths):
        return
    try:
        from huggingface_hub import snapshot_download
        repo_id = "VIZINTZOR/F5-TH-ONNX"
        model_path = snapshot_download(
            repo_id=repo_id,
            local_dir=model_dir,
            allow_patterns=model_files
        )
        print(f"Model downloaded to: {model_path}")
    except ImportError:
        raise RuntimeError("Please install huggingface-hub: pip install huggingface-hub")
    except Exception as e:
        print(f"An error occurred: {e}")

def prepare_text(text, max_chars=300):
    chunks = []
    current_chunk = ""
    text = text.replace(" ", "<unk>")
    segments = re.split(r"(<unk>|\s+)", text)
    for segment in segments:
        if not segment or segment in ("<unk>", " "):
            continue
        if len((current_chunk + segment).encode("utf-8")) <= max_chars:
            current_chunk += segment
            current_chunk += " "
        else:
            if current_chunk:
                chunks.append(current_chunk.strip())
            current_chunk = segment + " "
    if current_chunk:
        chunks.append(current_chunk.strip())
    chunks = [chunk.replace("<unk>", " ") for chunk in chunks]
    return chunks

def chunk_text(text, max_chars=300):
    words = " ".join(word_tokenize(text))
    sentence = prepare_text(words.replace("   ","|"),max_chars)
    sentences = []
    for i in sentence:
        text = i.replace(" ","").replace("|"," ")
        sentences.append(text) 
    return sentences
