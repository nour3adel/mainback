import numpy as np
from PIL import Image
import torch
from torchvision import transforms
from u2net import utils, model
import os
import onnxruntime as ort  # ONNX runtime for inference


onnx_model_path = './ckpt/u2net.quant.onnx'

# Global model variables
model_pred = None
onnx_session = None

# Load ONNX model
onnx_session = ort.InferenceSession(onnx_model_path, providers=['CPUExecutionProvider'])
use_onnx = True
print("Loaded ONNX model: u2net.quant.onnx")



def norm_pred(d):
    """Normalize the prediction map."""
    ma = np.max(d)
    mi = np.min(d)
    dn = (d - mi) / (ma - mi)
    return dn


def preprocess(image):
    """Preprocess the image for the model."""
    label_3 = np.zeros(image.shape)
    label = np.zeros(label_3.shape[0:2])

    if len(label_3.shape) == 3:
        label = label_3[:, :, 0]
    elif len(label_3.shape) == 2:
        label = label_3

    if len(image.shape) == 3 and len(label.shape) == 2:
        label = label[:, :, np.newaxis]
    elif len(image.shape) == 2 and len(label.shape) == 2:
        image = image[:, :, np.newaxis]
        label = label[:, :, np.newaxis]

    transform = transforms.Compose([utils.RescaleT(320), utils.ToTensorLab(flag=0)])
    sample = transform({"imidx": np.array([0]), "image": image, "label": label})

    return sample


def run_model(inputs_test):
    """Run the model inference for both PyTorch and ONNX."""
    if use_onnx:
        # Convert PyTorch tensor to NumPy array
        inputs_test_np = inputs_test.numpy()

        # Run ONNX inference
        onnx_inputs = {onnx_session.get_inputs()[0].name: inputs_test_np}
        onnx_output = onnx_session.run(None, onnx_inputs)

        # Extract the first output (assumed segmentation mask)
        pred = onnx_output[0][0, 0, :, :]
    else:
        with torch.no_grad():
            d1, _, _, _, _, _, _ = model_pred(inputs_test)
            pred = d1[:, 0, :, :].squeeze().cpu().detach().numpy()

    return norm_pred(pred)


def remove_bg(image):
    """Remove background using the model."""
    sample = preprocess(np.array(image))
    inputs_test = torch.FloatTensor(sample["image"].unsqueeze(0).float())

    # Get model prediction
    predict = run_model(inputs_test)

    # Convert prediction to image mask
    img_out = Image.fromarray(predict * 255).convert("RGB")
    img_out = img_out.resize(image.size, resample=Image.BILINEAR)
    
    # Apply the mask
    empty_img = Image.new("RGBA", image.size, 0)
    img_out = Image.composite(image, empty_img, img_out.convert("L"))

    return img_out


def remove_bg_mult(image):
    """Apply background removal multiple times for better accuracy."""
    img_out = image.copy()
    for _ in range(4):
        img_out = remove_bg(img_out)

    img_out = img_out.resize(image.size, resample=Image.BILINEAR)
    empty_img = Image.new("RGBA", image.size, 0)
    img_out = Image.composite(image, empty_img, img_out)
    return img_out


def change_background(image, background):
    """Change the background of the image."""
    background = background.resize(image.size, resample=Image.BILINEAR)
    img_out = Image.alpha_composite(background, image)
    return img_out
