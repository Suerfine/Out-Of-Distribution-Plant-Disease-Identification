import io
import sys
from pathlib import Path

# Ensure repo root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
from PIL import Image

from dpd.gradcam import colormap_jet, overlay_cam, find_target_layer, explain_cnn
from dpd.cnn import MultiHeadNet, HEAD_SIZES
from dpd.data import read_metadata, IMAGE_DIR
from dpd.inference import list_runs, load_run_model

print("=== 1. Testing Colormap & Overlay ===")
cam_dummy = np.linspace(0, 1, 100).reshape(10, 10)
jet_rgb = colormap_jet(cam_dummy)
assert jet_rgb.shape == (10, 10, 3)
assert jet_rgb.dtype == np.uint8
print("Colormap output shape:", jet_rgb.shape, "dtype:", jet_rgb.dtype)

orig = Image.new("RGB", (100, 100), color=(128, 128, 128))
blended = overlay_cam(orig, cam_dummy, alpha=0.5)
assert blended.size == (100, 100)
print("Blended overlay size:", blended.size)

print("=== 2. Testing Target Layer Detection on Architectures ===")
# ResNet-50
net_resnet = MultiHeadNet("resnet50", pretrained=False, heads={"crop": 55, "disease": 175}, dropout=0.1)
tl_resnet = find_target_layer(net_resnet.backbone)
print("ResNet-50 target layer:", type(tl_resnet).__name__)
assert tl_resnet is not None

# ConvNeXt Tiny
net_convnext = MultiHeadNet("convnext_tiny", pretrained=False, heads={"crop": 55, "disease": 175}, dropout=0.1)
tl_convnext = find_target_layer(net_convnext.backbone)
print("ConvNeXt target layer:", type(tl_convnext).__name__)
assert tl_convnext is not None

# Timm ResNetV2 (as used in Nat_multitask_resnetv2_50)
net_bit = MultiHeadNet("resnetv2_50x1_bit.goog_in21k_ft_in1k", pretrained=False, heads={"crop": 55, "disease": 175}, dropout=0.1)
tl_bit = find_target_layer(net_bit.backbone)
print("BiT ResNetV2 target layer:", type(tl_bit).__name__)
assert tl_bit is not None

print("=== 3. Testing with Real Run in runs/ ===")
df = read_metadata()
runs = list_runs()
print(f"Total runs found: {len(runs)}")

# Test each unique backbone family
test_candidates = ["convnext", "resnetv2_50", "efficientnetv2"]
sample_file = df.iloc[0]["file"]
sample_path = IMAGE_DIR / sample_file

for cand in test_candidates:
    matched = [r for r in runs if cand in r["version"].lower()]
    if matched:
        run = matched[0]
        print(f"\n--- Testing run: {run['version']} (method: {run['method']}) ---")
        model = load_run_model(run, df)
        print("model.supports_gradcam:", model.supports_gradcam)
        assert model.supports_gradcam is True
        
        pred_c = int(df.iloc[0]["crop_id"])
        pred_d = int(df.iloc[0]["disease_id"])
        
        expl = model.explain(sample_path, pred_c, pred_d)
        print("Explanation supported:", expl["supported"])
        assert expl["supported"] is True
        dis_cam = expl["cams"]["disease"]["cam"]
        crop_cam = expl["cams"]["crop"]["cam"]
        pair_cam = expl["cams"]["combined"]["cam"]
        print(f"  Disease CAM shape: {dis_cam.shape}, range: [{dis_cam.min():.2f}, {dis_cam.max():.2f}]")
        print(f"  Crop CAM shape:    {crop_cam.shape}, range: [{crop_cam.min():.2f}, {crop_cam.max():.2f}]")
        print(f"  Pair CAM shape:    {pair_cam.shape}, range: [{pair_cam.min():.2f}, {pair_cam.max():.2f}]")
        assert dis_cam.shape == (224, 224)
        assert 0.0 <= dis_cam.min() <= dis_cam.max() <= 1.0

print("=== 4. Testing VLM / Ensemble Graceful Disablement ===")
from dpd.inference import ClipRunModel, EnsembleRunModel
print("ClipRunModel.supports_gradcam:", ClipRunModel.supports_gradcam)
print("EnsembleRunModel.supports_gradcam:", EnsembleRunModel.supports_gradcam)
assert ClipRunModel.supports_gradcam is False
assert EnsembleRunModel.supports_gradcam is False

print("\nALL GRAD-CAM CHECKS PASSED SUCCESSFULLY!")
