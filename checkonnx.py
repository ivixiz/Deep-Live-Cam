# save as check_onnx.py and run in your virtualenv
import onnx, onnxruntime as ort, sys
p='models/inswapper_128.onnx'
print("File:", p)
try:
    m = onnx.load(p)
    onnx.checker.check_model(m)
    print("ONNX: model OK. opset:", m.opset_import[0].version)
except Exception as e:
    print("ONNX check failed:", e)

try:
    sess = ort.InferenceSession(p, providers=['CUDAExecutionProvider','CPUExecutionProvider'])
    print("Providers:", sess.get_providers())
    ins = sess.get_inputs()
    outs = sess.get_outputs()
    for i in ins:
        print("Input:", i.name, i.shape, i.type)
    for o in outs:
        print("Output:", o.name, o.shape, o.type)
except Exception as e:
    print("ORT session creation failed:", e)
