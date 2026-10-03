import os
import time
import tempfile
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import streamlit as st
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.ops import MultiScaleRoIAlign

IMG_W, IMG_H = 640, 360
NAMES = ['__background__', 'car', 'bus', 'truck', 'motorcycle', 'person']
NAME_TO_ID = {'car': 1, 'bus': 2, 'truck': 3, 'person': 5}
SHOWN = [1, 2, 3, 5]
COLORS = {1: (0, 220, 0), 2: (255, 165, 0), 3: (255, 60, 60), 4: (0, 255, 255), 5: (255, 235, 0)}

BASE = Path(__file__).parent
WEIGHTS = BASE / 'tron_cnn.pt'
SAMPLE_DIR = BASE / 'samples'


def conv_bn_relu(cin, cout):
    return nn.Sequential(nn.Conv2d(cin, cout, 3, padding=1, bias=False),
                         nn.BatchNorm2d(cout), nn.ReLU(inplace=True))


class TronBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        ch = [32, 64, 128, 256, 512]
        pool = [True, True, True, False, False]
        layers, cin = [], 3
        for c, p in zip(ch, pool):
            layers += [conv_bn_relu(cin, c), conv_bn_relu(c, c)]
            if p:
                layers.append(nn.MaxPool2d(2))
            cin = c
        self.body = nn.Sequential(*layers)
        self.out_channels = cin

    def forward(self, x):
        return self.body(x)


def build_model(num_classes=len(NAMES)):
    backbone = TronBackbone()
    anchors = AnchorGenerator(sizes=((16, 32, 64, 128, 256),), aspect_ratios=((0.5, 1.0, 2.0),))
    roi = MultiScaleRoIAlign(featmap_names=['0'], output_size=7, sampling_ratio=2)
    return FasterRCNN(backbone, num_classes=num_classes,
                      rpn_anchor_generator=anchors, box_roi_pool=roi,
                      min_size=IMG_H, max_size=IMG_W,
                      rpn_pre_nms_top_n_train=1000, rpn_post_nms_top_n_train=500,
                      rpn_pre_nms_top_n_test=500, rpn_post_nms_top_n_test=300,
                      box_batch_size_per_image=128, box_score_thresh=0.05,
                      box_detections_per_img=100)


@st.cache_resource(show_spinner='Loading model')
def load_model(path):
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    m = build_model()
    ckpt = torch.load(path, map_location=dev)
    m.load_state_dict(ckpt['model'])
    m.to(dev).eval()
    return m, dev


def detect(model, dev, rgb, thr):
    h, w = rgb.shape[:2]
    small = cv2.resize(rgb, (IMG_W, IMG_H))
    x = torch.from_numpy(small).permute(2, 0, 1).float().div(255).to(dev)
    with torch.no_grad():
        o = model([x])[0]
    keep = o['scores'] >= thr
    boxes = o['boxes'][keep].cpu().numpy()
    labels = o['labels'][keep].cpu().numpy()
    scores = o['scores'][keep].cpu().numpy()
    if len(boxes):
        boxes[:, [0, 2]] *= w / IMG_W
        boxes[:, [1, 3]] *= h / IMG_H
    return boxes, labels, scores


def select(boxes, labels, scores, active):
    if len(labels) == 0:
        return boxes, labels, scores
    m = np.isin(labels, active)
    return boxes[m], labels[m], scores[m]


def draw(rgb, boxes, labels, scores, show_labels=True):
    out = np.ascontiguousarray(rgb.copy())
    h, w = out.shape[:2]
    th = max(2, w // 500)
    fs = max(0.45, w / 1600)
    for b, l, s in zip(boxes, labels, scores):
        x1, y1, x2, y2 = [int(v) for v in b]
        c = COLORS[int(l)]
        cv2.rectangle(out, (x1, y1), (x2, y2), c, th)
        if show_labels:
            text = f'{NAMES[int(l)]} {s:.2f}'
            (tw, tht), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, fs, 1)
            ty = max(tht + 4, y1)
            cv2.rectangle(out, (x1, ty - tht - 4), (x1 + tw + 4, ty), c, -1)
            cv2.putText(out, text, (x1 + 2, ty - 3), cv2.FONT_HERSHEY_SIMPLEX,
                        fs, (0, 0, 0), 1, cv2.LINE_AA)
    return out


def cards(items):
    html = ''.join(f'<div class="card"><div class="v">{v}</div><div class="k">{k}</div></div>'
                   for k, v in items)
    st.markdown(f'<div class="cards">{html}</div>', unsafe_allow_html=True)


def summary_cards(labels, ms):
    items = [('Objects', len(labels)), ('Latency', f'{ms:.0f} ms')]
    items += [(NAMES[k].capitalize(), int((labels == k).sum())) for k in SHOWN]
    cards(items)


st.set_page_config(page_title='Tron Traffic Detection', page_icon='🚦', layout='wide')

st.markdown(
    '''
    <style>
    #MainMenu {visibility: hidden;}
    .stDeployButton {display: none;}
    footer {visibility: hidden;}
    .block-container {padding-top: 2rem; max-width: 1200px;}
    .hero {background: linear-gradient(135deg, #0f2027 0%, #203a43 50%, #2c5364 100%);
           color: #ffffff; padding: 28px 32px; border-radius: 16px; margin-bottom: 18px;}
    .hero h1 {margin: 0 0 6px 0; font-size: 2rem; line-height: 1.2; color: #ffffff;}
    .hero p {margin: 0; opacity: 0.85; font-size: 1rem;}
    .cards {display: flex; gap: 12px; flex-wrap: wrap; margin: 14px 0;}
    .card {flex: 1; min-width: 110px; background: rgba(128,128,128,0.12);
           border: 1px solid rgba(128,128,128,0.25); border-radius: 12px; padding: 14px 16px;}
    .card .v {font-size: 1.7rem; font-weight: 700; line-height: 1.2;}
    .card .k {font-size: 0.85rem; opacity: 0.7;}
    .chip {display: inline-block; width: 12px; height: 12px; border-radius: 3px; margin-right: 8px;}
    </style>
    <div class="hero">
      <h1>Vehicle and Object Detection for Traffic Surveillance</h1>
      <p>Team Tron | Custom CNN backbone with a Faster R-CNN detection head | Trained from scratch on BDD100K</p>
    </div>
    ''',
    unsafe_allow_html=True,
)

if not WEIGHTS.exists():
    st.error(f'Model file not found: {WEIGHTS.name}. Place it next to app.py.')
    st.stop()

model, dev = load_model(str(WEIGHTS))

with st.sidebar:
    st.header('Settings')
    thr = st.slider('Confidence threshold', 0.1, 0.95, 0.6, 0.05)
    chosen = st.multiselect('Classes to show', list(NAME_TO_ID), default=list(NAME_TO_ID))
    show_labels = st.toggle('Show labels', value=True)
    active = [NAME_TO_ID[c] for c in chosen]
    st.markdown('**Legend**')
    legend = ''.join(
        f'<div><span class="chip" style="background: rgb{COLORS[k]};"></span>{NAMES[k].capitalize()}</div>'
        for k in SHOWN)
    st.markdown(legend, unsafe_allow_html=True)
    st.caption(f'Running on {dev.type.upper()}')


def infer(rgb):
    t = time.time()
    boxes, labels, scores = detect(model, dev, rgb, thr)
    ms = (time.time() - t) * 1000
    boxes, labels, scores = select(boxes, labels, scores, active)
    return draw(rgb, boxes, labels, scores, show_labels), labels, ms


def show_result(rgb):
    out, labels, ms = infer(rgb)
    left, right = st.columns(2)
    left.image(rgb, caption='Input')
    right.image(out, caption='Detections')
    summary_cards(labels, ms)


tab_img, tab_vid, tab_cam, tab_live, tab_about = st.tabs(
    ['Image', 'Video', 'Camera snapshot', 'Live webcam', 'About the model'])

with tab_img:
    up = st.file_uploader('Upload a road scene', type=['jpg', 'jpeg', 'png'], key='img')
    samples = []
    if SAMPLE_DIR.exists():
        samples = sorted(list(SAMPLE_DIR.glob('*.jpg')) + list(SAMPLE_DIR.glob('*.png')))
    pick = 'None'
    if samples:
        pick = st.selectbox('Or choose a sample image', ['None'] + [p.name for p in samples])
    rgb = None
    if up is not None:
        arr = cv2.imdecode(np.frombuffer(up.read(), np.uint8), cv2.IMREAD_COLOR)
        rgb = cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)
    elif pick != 'None':
        arr = cv2.imread(str(SAMPLE_DIR / pick))
        rgb = cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)
    if rgb is not None:
        show_result(rgb)
    else:
        st.info('Upload an image or pick a sample to run detection.')

with tab_vid:
    vid = st.file_uploader('Upload a short traffic video', type=['mp4', 'avi', 'mov', 'mkv'], key='vid')
    c1, c2 = st.columns(2)
    stride = c1.slider('Process every Nth frame', 1, 10, 3)
    max_frames = c2.slider('Maximum frames to process', 30, 300, 90, 10)
    if vid is not None and st.button('Process video', type='primary'):
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
        tmp.write(vid.read())
        tmp.close()
        cap = cv2.VideoCapture(tmp.name)
        frame_box = st.empty()
        bar = st.progress(0.0)
        done, idx, counts = 0, 0, []
        started = time.time()
        try:
            while done < max_frames:
                ok, frame = cap.read()
                if not ok:
                    break
                idx += 1
                if idx % stride:
                    continue
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                out, labels, ms = infer(rgb)
                frame_box.image(out)
                counts.append(len(labels))
                done += 1
                bar.progress(min(done / max_frames, 1.0))
        finally:
            cap.release()
            os.unlink(tmp.name)
        bar.progress(1.0)
        if counts:
            elapsed = time.time() - started
            cards([('Frames processed', done),
                   ('Average objects per frame', f'{np.mean(counts):.1f}'),
                   ('Processing speed', f'{done / elapsed:.1f} frames/s')])
        else:
            st.warning('No frames could be read from this video.')

with tab_cam:
    shot = st.camera_input('Take a photo')
    if shot is not None:
        arr = cv2.imdecode(np.frombuffer(shot.getvalue(), np.uint8), cv2.IMREAD_COLOR)
        show_result(cv2.cvtColor(arr, cv2.COLOR_BGR2RGB))

with tab_live:
    st.caption('Reads a camera attached to the machine running the app, so it works when the app runs locally.')
    src = st.number_input('Camera index', min_value=0, max_value=5, value=0, step=1)
    run = st.checkbox('Start webcam')
    frame_box = st.empty()
    info_box = st.empty()
    if run:
        cap = cv2.VideoCapture(int(src), cv2.CAP_DSHOW)
        if not cap.isOpened():
            st.error('Could not open the camera. Try another camera index.')
        else:
            try:
                last = time.time()
                while True:
                    ok, frame = cap.read()
                    if not ok:
                        break
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    out, labels, ms = infer(rgb)
                    now = time.time()
                    fps = 1.0 / max(now - last, 1e-6)
                    last = now
                    cv2.putText(out, f'FPS {fps:.1f}', (10, 28), cv2.FONT_HERSHEY_SIMPLEX,
                                0.8, (255, 255, 255), 2, cv2.LINE_AA)
                    frame_box.image(out)
                    info_box.write(', '.join(f'{NAMES[k]}: {int((labels == k).sum())}' for k in SHOWN))
            finally:
                cap.release()

with tab_about:
    st.subheader('Architecture')
    st.write(
        'A custom CNN backbone built from scratch: 5 stages of 2 convolution layers each '
        '(10 in total), every layer followed by batch normalisation and ReLU, with 3 max-pooling '
        'steps and 512 output channels. The backbone feeds a Faster R-CNN head with a region '
        'proposal network and RoIAlign. No pretrained weights are used.')
    cards([('Backbone parameters', '4.71M'), ('Full detector', '33.88M'),
           ('Input size', '640 x 360'), ('Classes', '4')])
    st.subheader('Training')
    st.write(
        'BDD100K, 8,000 training images, 30 epochs, AdamW with warmup and cosine decay, '
        'mixed precision, horizontal flip and brightness augmentation. Evaluation uses 1,000 '
        'validation images.')
    st.subheader('Results')
    cards([('mAP50', '0.443'), ('mAP50-95', '0.256'), ('Speed on Tesla T4', '44 FPS')])
    st.table({'Class': ['Car', 'Bus', 'Truck', 'Person'],
              'mAP50-95': [0.376, 0.258, 0.256, 0.135]})
    st.subheader('Limitations')
    st.write(
        'The model detects four classes only: car, bus, truck and person. Motorcycles, bicycles, '
        'riders and auto-rickshaws are not classes. Accuracy drops in dense or heavily occluded '
        'scenes and on road types that differ from the training data, and person is the weakest '
        'class because pedestrians are small.')