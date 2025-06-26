import onnxruntime as ort
import numpy as np
import open_clip
import time

sess = ort.InferenceSession("clip_text_int8.onnx", providers=["CPUExecutionProvider"])


tokzr = open_clip.get_tokenizer("ViT-H-14-quickgelu")

start_time = time.time()

ids = tokzr(["cat"]).cpu().numpy().astype(np.int32)
vec = sess.run(None, {"input_ids": ids})[0]
# vec /= np.linalg.norm(vec, axis=-1, keepdims=True)

end_time = time.time()

print(vec)

print(f"Time taken: {end_time - start_time} seconds")
