import whisperx
import gc
from threading import Thread

def process():
    device = "cuda"
    audio_file = "audio.mp3"
    batch_size = 16 # reduce if low on GPU mem
    compute_type = "float16" # change to "int8" if low on GPU mem (may reduce accuracy)

    # 1. Transcribe with original whisper (batched)
    model = whisperx.load_model("base.en", device, compute_type=compute_type)
    audio = whisperx.load_audio(audio_file)
    result = model.transcribe(audio, batch_size=batch_size)
    print(result["segments"])

# Create and start 6 threads
threads = []
for _ in range(6):
    t = Thread(target=process)
    threads.append(t)
    t.start()

# Wait for all threads to complete
for t in threads:
    t.join()