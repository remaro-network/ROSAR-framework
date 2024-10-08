import onnx
import onnxruntime
import torch

def check_model(model_path):
    model = onnx.load(model_path)
    try:
        onnx.checker.check_model(model)
    except onnx.checker.ValidationError as e:
        print(f"The model is invalid: {e}")
    else:
        print("The model is valid")

def get_model_interface(model_path):
    session = onnxruntime.InferenceSession(model_path)
    input = session.get_inputs()[0]
    output = session.get_outputs()[0]
    h = input.shape[2]
    w = input.shape[3]
    n_bboxes = output.shape[1]
    n_data = output.shape[2]
    return h, w, n_bboxes, n_data

def predict(im_scaled, model_path):
    session = onnxruntime.InferenceSession(model_path)
    inputs = session.get_inputs()
    outputs = session.get_outputs()
    #for i, info in enumerate(inputs):
    #    print(f"Input {i}: name = {info.name}, shape = {info.shape}")
    #for i, info in enumerate(outputs):
    #    print(f"Output {i}: name = {info.name}, shape = {info.shape}")
    input_name = inputs[0].name
    output_name = outputs[0].name
    # Perform prediction
    raw_pred = session.run([output_name], {input_name: im_scaled.numpy()})
    if isinstance(raw_pred, (list, tuple)):
        raw_pred = raw_pred[0]
    raw_pred = raw_pred[0]

    # print('Raw_pred:', raw_pred)
    # print('Raw_pred shape:', raw_pred.shape)
    return torch.from_numpy(raw_pred)
