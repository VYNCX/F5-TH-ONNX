# F5-TH-ONNX

Text-to-Speech (TTS) ภาษาไทย — เครื่องมือสร้างเสียงพูดจากข้อความ Zero Shot TTS ด้วยโมเดล F5-TTS ที่ปรับแต่งให้อยู่ในรูปแบบ ONNX ให้สามารถรันบน GPU ขนาดเล็กกว่าและใช้ทรัพยากรน้อยกว่า

- 🔥 สถาปัตยกรรม: [F5-TTS](https://arxiv.org/abs/2410.06885)  
- 🚀 Export ONNX: [F5-TTS-ONNX](https://github.com/DakeQQ/F5-TTS-ONNX)

### ติดตั้ง

```
pip install f5-th-onnx
```

 ### การใช้งาน

```
from f5_th_onnx import TTS

TTS(
    ref_audio="YOUR_AUDIO_PATH",
    ref_text="นี่คือเสียงพูดต้นฉบับ.", 
    gen_text="สวัสดีครับ นี่คือเสียงพูดภาษาไทย.", 
    speed=1.0,
    output="generated.wav"
)
```
