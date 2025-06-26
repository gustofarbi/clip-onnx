import onnxruntime as ort, numpy as np, open_clip

sess = ort.InferenceSession(
    "clip_text_int8.onnx", providers=["CPUExecutionProvider"]
)

tokzr = open_clip.get_tokenizer("ViT-H-14-quickgelu")
ids = tokzr(["cat"]).cpu().numpy().astype(np.int64)
vec = sess.run(None, {"input_ids": ids})[0]
# vec /= np.linalg.norm(vec, axis=-1, keepdims=True)

print(vec)
