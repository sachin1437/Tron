import os, json, time, math, random
from pathlib import Path
import numpy as np
import torch, torch.nn as nn
from PIL import Image, ImageDraw
from torch.utils.data import Dataset, DataLoader

SMOKE = False
N_TRAIN = 8000
N_VAL = 1000
EPOCHS = 30
BATCH = 8
IMG_W, IMG_H = 640, 360
LR = 1e-3
TIME_BUDGET_MIN = 180
EVAL_EVERY_SUBSET = 200
SEED = 42
OUT = Path('/kaggle/working')

if SMOKE:
    N_TRAIN, N_VAL, EPOCHS, EVAL_EVERY_SUBSET = 200, 50, 1, 50

random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('device:', device, '| SMOKE:', SMOKE)

try:
    from torchmetrics.detection import MeanAveragePrecision
except ImportError:
    os.system('pip install -q torchmetrics pycocotools')
    from torchmetrics.detection import MeanAveragePrecision

ROOT = '/kaggle/input'
yaml_path = None
cands = {}
SPLITS = ('train', 'val', 'valid', 'test')

for dp, dn, fn in os.walk(ROOT):
    if 'data.yaml' in fn and yaml_path is None:
        yaml_path = os.path.join(dp, 'data.yaml')
    base = os.path.basename(dp)
    parent = os.path.basename(os.path.dirname(dp))
    if base in ('images', 'labels'):
        if parent in SPLITS:
            cands[(base, parent)] = dp
        for s in dn:
            if s in SPLITS:
                cands[(base, s)] = os.path.join(dp, s)
        dn[:] = []

print('yaml:', yaml_path)
print('folders found:', sorted(cands.keys()))

TRAIN_SPLIT = 'train'
VAL_SPLIT = 'val' if ('images', 'val') in cands else 'valid'
img_dir = {'train': cands[('images', TRAIN_SPLIT)], 'val': cands[('images', VAL_SPLIT)]}
lab_dir = {'train': cands[('labels', TRAIN_SPLIT)], 'val': cands[('labels', VAL_SPLIT)]}
print('train images:', img_dir['train'])
print('train labels:', lab_dir['train'])

DEFAULT_BDD = ['pedestrian', 'rider', 'car', 'truck', 'bus', 'train',
               'motorcycle', 'bicycle', 'traffic light', 'traffic sign']
old_names = DEFAULT_BDD
if yaml_path:
    import yaml
    with open(yaml_path) as f:
        y = yaml.safe_load(f)
    n = y.get('names', DEFAULT_BDD)
    old_names = [n[k] for k in sorted(n)] if isinstance(n, dict) else list(n)
print('dataset classes:', old_names)

NEW_NAMES = ['__background__', 'car', 'bus', 'truck', 'motorcycle', 'person']
NAME_TO_NEW = {'car': 1, 'bus': 2, 'truck': 3, 'motorcycle': 4, 'motorbike': 4,
               'pedestrian': 5, 'person': 5}
OLD2NEW = {i: NAME_TO_NEW[str(nm).lower()] for i, nm in enumerate(old_names)
           if str(nm).lower() in NAME_TO_NEW}
print('mapping (dataset id -> our id):', OLD2NEW)
assert len(OLD2NEW) >= 4, 'Class mapping looks wrong. Check the printed class names above.'

def read_labels(txt_path):
    boxes, labels = [], []
    if not os.path.exists(txt_path):
        return boxes, labels
    with open(txt_path) as f:
        for line in f:
            p = line.split()
            if len(p) < 5:
                continue
            c = int(float(p[0]))
            if c not in OLD2NEW:
                continue
            cx, cy, w, h = [float(v) for v in p[1:5]]
            x1 = max(0.0, (cx - w / 2) * IMG_W); y1 = max(0.0, (cy - h / 2) * IMG_H)
            x2 = min(float(IMG_W), (cx + w / 2) * IMG_W); y2 = min(float(IMG_H), (cy + h / 2) * IMG_H)
            if x2 - x1 < 3 or y2 - y1 < 3:
                continue
            boxes.append([x1, y1, x2, y2]); labels.append(OLD2NEW[c])
    return boxes, labels

def build_items(split, n):
    files = sorted(f for f in os.listdir(img_dir[split]) if f.lower().endswith(('.jpg', '.jpeg', '.png')))
    random.Random(SEED).shuffle(files)
    items = []
    for fn in files:
        b, l = read_labels(os.path.join(lab_dir[split], os.path.splitext(fn)[0] + '.txt'))
        if len(b) == 0:
            continue
        items.append((os.path.join(img_dir[split], fn), b, l))
        if len(items) >= n:
            break
    return items

class BDDDet(Dataset):
    def __init__(self, items, train):
        self.items, self.train = items, train
    def __len__(self):
        return len(self.items)
    def __getitem__(self, i):
        path, b, l = self.items[i]
        img = Image.open(path).convert('RGB').resize((IMG_W, IMG_H))
        x = torch.from_numpy(np.asarray(img).copy()).permute(2, 0, 1).float() / 255.0
        boxes = torch.tensor(b, dtype=torch.float32)
        labels = torch.tensor(l, dtype=torch.int64)
        if self.train:
            if random.random() < 0.5:
                x = x.flip(-1)
                boxes = boxes[:, [2, 1, 0, 3]]
                boxes[:, 0] = IMG_W - boxes[:, 0]
                boxes[:, 2] = IMG_W - boxes[:, 2]
            x = (x * random.uniform(0.7, 1.3)).clamp(0, 1)
        return x, {'boxes': boxes, 'labels': labels}

def collate(batch):
    return tuple(zip(*batch))

t0 = time.time()
train_items = build_items('train', N_TRAIN)
val_items = build_items('val', N_VAL)
print(f'train {len(train_items)} | val {len(val_items)} | loaded labels in {time.time()-t0:.0f}s')

counts = {k: 0 for k in NEW_NAMES[1:]}
for _, _, l in train_items:
    for c in l:
        counts[NEW_NAMES[c]] += 1
print('train object counts:', counts)

train_loader = DataLoader(BDDDet(train_items, True), batch_size=BATCH, shuffle=True,
                          num_workers=2, collate_fn=collate, drop_last=True)
val_loader = DataLoader(BDDDet(val_items, False), batch_size=BATCH, shuffle=False,
                        num_workers=2, collate_fn=collate)

from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.ops import MultiScaleRoIAlign

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
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
    def forward(self, x):
        return self.body(x)

def build_model(num_classes=len(NEW_NAMES)):
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

model = build_model().to(device)
n_bb = sum(p.numel() for p in model.backbone.parameters())
n_all = sum(p.numel() for p in model.parameters())
print(f'backbone params: {n_bb/1e6:.2f}M | full detector params: {n_all/1e6:.2f}M')
print(model.backbone)

@torch.no_grad()
def evaluate(loader, max_images=None):
    model.eval()
    metric = MeanAveragePrecision(iou_type='bbox', class_metrics=True)
    seen = 0
    for imgs, tgts in loader:
        imgs = [im.to(device) for im in imgs]
        with torch.autocast(device_type='cuda', enabled=(device.type == 'cuda')):
            outs = model(imgs)
        preds = [{k: v.float().cpu() if k != 'labels' else v.cpu() for k, v in o.items()} for o in outs]
        metric.update(preds, [{k: v for k, v in t.items()} for t in tgts])
        seen += len(imgs)
        if max_images and seen >= max_images:
            break
    return metric.compute()

params = [p for p in model.parameters() if p.requires_grad]
opt = torch.optim.AdamW(params, lr=LR, weight_decay=1e-4)
scaler = torch.amp.GradScaler('cuda', enabled=(device.type == 'cuda'))

steps_per_epoch = len(train_loader)
total_steps = EPOCHS * steps_per_epoch
warmup = min(300, total_steps // 5)
budget_s = TIME_BUDGET_MIN * 60
history = {'epoch': [], 'loss': [], 'map50': [], 'map': []}
start, step, stop = time.time(), 0, False

for epoch in range(1, EPOCHS + 1):
    model.train()
    run, nb = 0.0, 0
    for imgs, tgts in train_loader:
        progress = max(step / total_steps, (time.time() - start) / budget_s)
        if progress >= 1.0:
            stop = True
            break
        lr = LR * 0.5 * (1 + math.cos(math.pi * progress))
        if step < warmup:
            lr *= (step + 1) / warmup
        for g in opt.param_groups:
            g['lr'] = lr

        imgs = [im.to(device) for im in imgs]
        tgts = [{k: v.to(device) for k, v in t.items()} for t in tgts]
        with torch.autocast(device_type='cuda', enabled=(device.type == 'cuda')):
            losses = model(imgs, tgts)
            loss = sum(losses.values())
        if not torch.isfinite(loss):
            opt.zero_grad(set_to_none=True)
            step += 1
            continue
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(params, 10.0)
        scaler.step(opt); scaler.update()

        run += loss.item(); nb += 1; step += 1
        if step % 50 == 0:
            el = (time.time() - start) / 60
            print(f'ep {epoch} step {step}/{total_steps} loss {run/nb:.3f} lr {lr:.2e} | {el:.1f} min')

    if nb == 0:
        break
    res = evaluate(val_loader, EVAL_EVERY_SUBSET)
    history['epoch'].append(epoch); history['loss'].append(run / nb)
    history['map50'].append(float(res['map_50'])); history['map'].append(float(res['map']))
    print(f'=== epoch {epoch} | train loss {run/nb:.3f} | val mAP50 {float(res["map_50"]):.3f} | mAP50-95 {float(res["map"]):.3f}')
    torch.save({'model': model.state_dict(), 'epoch': epoch, 'classes': NEW_NAMES,
                'img_size': (IMG_W, IMG_H)}, OUT / 'tron_cnn.pt')
    json.dump(history, open(OUT / 'history.json', 'w'))
    if stop:
        break
print(f'training finished in {(time.time()-start)/60:.1f} min')

res = evaluate(val_loader)
final = {'mAP50': float(res['map_50']), 'mAP50-95': float(res['map']),
         'per_class_mAP': {}}
cls_ids = res['classes'].tolist() if 'classes' in res else []
pcm = res['map_per_class'].tolist() if res['map_per_class'].ndim > 0 else []
for cid, v in zip(cls_ids, pcm):
    final['per_class_mAP'][NEW_NAMES[cid]] = round(v, 4)
print(json.dumps(final, indent=2))
json.dump(final, open(OUT / 'final_metrics.json', 'w'), indent=2)

model.eval()
x = [torch.rand(3, IMG_H, IMG_W, device=device)]
with torch.no_grad():
    for _ in range(5):
        model(x)
    if device.type == 'cuda':
        torch.cuda.synchronize()
    t = time.time()
    for _ in range(30):
        model(x)
    if device.type == 'cuda':
        torch.cuda.synchronize()
ms = (time.time() - t) / 30 * 1000
print(f'inference: {ms:.1f} ms/image  (~{1000/ms:.0f} FPS)')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

if history['epoch']:
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.5))
    ax[0].plot(history['epoch'], history['loss'], marker='o'); ax[0].set_title('Training loss'); ax[0].set_xlabel('epoch')
    ax[1].plot(history['epoch'], history['map50'], marker='o', label='mAP50')
    ax[1].plot(history['epoch'], history['map'], marker='o', label='mAP50-95')
    ax[1].set_title('Validation mAP'); ax[1].set_xlabel('epoch'); ax[1].legend()
    plt.tight_layout(); plt.savefig(OUT / 'training_curves.png', dpi=150); plt.close()

COLORS = {1: 'lime', 2: 'orange', 3: 'red', 4: 'cyan', 5: 'yellow'}

@torch.no_grad()
def predict(pil_img, thr=0.4):
    model.eval()
    img = pil_img.convert('RGB').resize((IMG_W, IMG_H))
    x = torch.from_numpy(np.asarray(img).copy()).permute(2, 0, 1).float().div(255).to(device)
    o = model([x])[0]
    keep = o['scores'] >= thr
    return img, o['boxes'][keep].cpu(), o['labels'][keep].cpu(), o['scores'][keep].cpu()

def draw(img, boxes, labels, scores):
    d = ImageDraw.Draw(img)
    for b, l, s in zip(boxes.tolist(), labels.tolist(), scores.tolist()):
        d.rectangle(b, outline=COLORS[l], width=2)
        d.text((b[0] + 2, max(0, b[1] - 11)), f'{NEW_NAMES[l]} {s:.2f}', fill=COLORS[l])
    return img

os.makedirs(OUT / 'samples', exist_ok=True)
for i, (path, _, _) in enumerate(val_items[:8]):
    img, b, l, s = predict(Image.open(path))
    draw(img, b, l, s).save(OUT / 'samples' / f'sample_{i}.jpg')
print('saved: tron_cnn.pt, history.json, final_metrics.json, training_curves.png, samples/')