# Vehicle and Object Detection for Traffic Surveillance using CNN-Based Models

Team Tron | Author: Sachin Gupta | Guide: Dr. Rosepreet Kaur Bhogal

Live demo: https://tron-traffic-detection.streamlit.app

A traffic surveillance detector built around a CNN backbone designed and trained from scratch, with no pretrained weights. It finds cars, buses, trucks and people in road scenes and ships with a Streamlit app for images, video and camera input.

## Results

Evaluated on 1,000 validation images from BDD100K.

| Metric | Value |
| --- | --- |
| mAP50 | 0.443 |
| mAP50-95 | 0.256 |
| Inference speed | 22.5 ms per image (about 44 FPS) on a Tesla T4 |

Per-class mAP50-95:

| Class | mAP50-95 |
| --- | --- |
| Car | 0.376 |
| Bus | 0.258 |
| Truck | 0.256 |
| Person | 0.135 |

The class scores follow the amount of training data. The 8,000 training images contain 82,875 car boxes, 10,698 person boxes, 3,515 truck boxes and 1,299 bus boxes. Person is also the hardest class because pedestrians are small in the frame.

## Model

The backbone is a plain multi-layer CNN:

```
Input 640 x 360 x 3
Stage 1: Conv3x3(32)  + BN + ReLU, Conv3x3(32)  + BN + ReLU, MaxPool
Stage 2: Conv3x3(64)  + BN + ReLU, Conv3x3(64)  + BN + ReLU, MaxPool
Stage 3: Conv3x3(128) + BN + ReLU, Conv3x3(128) + BN + ReLU, MaxPool
Stage 4: Conv3x3(256) + BN + ReLU, Conv3x3(256) + BN + ReLU
Stage 5: Conv3x3(512) + BN + ReLU, Conv3x3(512) + BN + ReLU
Output: 512-channel feature map at stride 8
```

The feature map feeds a Faster R-CNN detection head (region proposal network, RoIAlign and a box classifier and regressor) from torchvision. The backbone has 4.71M parameters and the full detector has 33.88M.

## Training

| Setting | Value |
| --- | --- |
| Dataset | BDD100K in YOLO format, 8,000 training images |
| Classes | car, bus, truck, person |
| Epochs | 30 (131 minutes on a Kaggle T4) |
| Optimiser | AdamW, learning rate 1e-3, weight decay 1e-4 |
| Schedule | linear warmup, then cosine decay |
| Precision | mixed precision with gradient clipping |
| Augmentation | horizontal flip, brightness jitter |
| Initialisation | Kaiming normal, no pretrained weights |

Training ran in a Kaggle notebook on a T4 GPU, and only the resulting weights (`tron_cnn.pt`) were downloaded to this project. The notebook code is included as `tron_cnn_detector.py` for reference. It uses Kaggle paths, so to rerun it, add the BDD100K YOLO-format dataset as a Kaggle input, enable a GPU, set `SMOKE = False` and run it. With `SMOKE = True` it does a short test run to check the setup. It writes the weights along with the metrics, training curves and sample detections.

## Project structure

```
app.py                    Streamlit app
tron_cnn.pt               trained weights (kept out of this repo, hosted on Hugging Face)
requirements.txt          dependencies
tron_cnn_detector.py      training and evaluation code from the Kaggle notebook (reference)
samples/                  optional demo images shown in the app
README.md
```

## Run locally

Python 3.11 or 3.12 is recommended.

```
python -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

For an NVIDIA GPU on Windows, install PyTorch first:

```
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
```

Then install the rest and start the app:

```
pip install -r requirements.txt
streamlit run app.py
```

The app looks for `tron_cnn.pt` next to `app.py`. If the file is not there, it downloads the weights from the Hugging Face repository `sachingupta07/tron-cnn-weights` the first time it starts.

Without an NVIDIA GPU, skip the PyTorch line and the app runs on the CPU. To check which one is in use:

```
python -c "import torch; print(torch.cuda.is_available())"
```

## Using the app

| Tab | What it does |
| --- | --- |
| Image | Upload a road image or pick one from `samples/` |
| Video | Upload a short clip, choose how many frames to skip and a maximum frame count |
| Camera snapshot | Take a photo with your browser camera |
| Live webcam | Continuous detection from a camera on the machine running the app (local only) |
| About the model | Architecture, training setup, results and limitations |

The sidebar sets the confidence threshold, which classes to show, and whether labels are drawn. A threshold of 0.6 to 0.7 gives the cleanest boxes.

## Deployment

The app is deployed on Streamlit Community Cloud from this repository: https://tron-traffic-detection.streamlit.app

The weights file is too large for GitHub (over 100 MB), so it is hosted in a Hugging Face model repository, `sachingupta07/tron-cnn-weights`, and the app downloads it on first start. The cloud version runs on CPU, so each image takes a fraction of a second to a few seconds. The live webcam tab needs a camera on the server, so it only works when the app runs locally. The camera snapshot tab works online because it uses the browser's camera. Apps on the free tier go to sleep after a period without visitors, so the first load after a quiet period takes longer.

## Limitations

- The model detects four classes. Motorcycles, bicycles, riders and auto-rickshaws are not classes. The output layer keeps a motorcycle slot, but it was never trained, so the app does not show it.
- It was trained on 8,000 images from BDD100K, which is mostly road-level footage from US driving. Accuracy drops on dense, heavily occluded scenes and on road types that look different from that data.
- Person detection is the weakest result.
- Training from scratch on a small subset gives lower accuracy than fine-tuning a pretrained model would. More data and longer training are the most direct improvements.

## Acknowledgements

BDD100K dataset by Berkeley DeepDrive. The trained weights are shared under CC BY-NC 4.0 because the dataset terms cover educational, research and not-for-profit use. Built with PyTorch, torchvision, OpenCV and Streamlit. Training ran on Kaggle.