from flask import Flask, request, jsonify
from PIL import Image
import io
import numpy as np
import torch
import torchvision.transforms as transforms
from archs_ucm_v2 import UCM_NetV2
from torchvision.transforms import InterpolationMode
from flask_cors import CORS



app=Flask(__name__)
CORS(app)  # <--- enable CORS for all routes

model=UCM_NetV2(num_classes=1)
model.load_state_dict(torch.load('fine_tuned_min_loss.pth', map_location='cpu'))
model.eval()




preprocess = transforms.Compose([
    transforms.Resize((256, 256), interpolation=InterpolationMode.BILINEAR),
    transforms.ToTensor(),
    # Add normalization here if your model needs it, e.g. transforms.Normalize(...)
])

@app.route('/predict',methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image in request'}), 400
    image_file = request.files['image']

    img_bytes = image_file.read()
    img = Image.open(io.BytesIO(img_bytes))
    input_tensor = preprocess(img).unsqueeze(0)

    with torch.no_grad():
        _,output = model(input_tensor)
        print(output.shape)
        print(output)
        # Example: get sigmoid and threshold 0.5 for binary prediction
        pred = torch.sigmoid(output)
        pred_mask = (pred > 0.5).cpu().numpy().astype(np.uint8)  # numpy array with 0/1

        # Convert numpy array to list (or you can encode it as base64 image)
        pred_list = pred_mask.tolist()
        mask_img = Image.fromarray(pred_mask.squeeze() * 255)  # multiply by 255 for visibility
        mask_img.save('predicted_mask.png')

        return jsonify({'prediction_mask': pred_list})


if __name__ == '__main__':
    app.run(debug=True)
