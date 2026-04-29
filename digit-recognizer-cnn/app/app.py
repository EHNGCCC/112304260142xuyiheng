from __future__ import annotations

import base64
from io import BytesIO
import os
from pathlib import Path
import sys

from flask import Flask, jsonify, render_template_string, request
import numpy as np
from PIL import Image, ImageOps
import torch


def ensure_localhost_no_proxy() -> None:
    localhost_values = ["127.0.0.1", "localhost"]
    for key in ("NO_PROXY", "no_proxy"):
        current = os.environ.get(key, "")
        values = [value.strip() for value in current.split(",") if value.strip()]
        for value in localhost_values:
            if value not in values:
                values.append(value)
        os.environ[key] = ",".join(values)


ensure_localhost_no_proxy()

PROJECT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_DIR / "src"
MODEL_PATH = PROJECT_DIR / "models" / "final_cnn.pt"
sys.path.insert(0, str(SRC_DIR))

from model import FinalCNN, MNIST_MEAN, MNIST_STD  # noqa: E402


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
app = Flask(__name__)


def load_model() -> FinalCNN:
    model = FinalCNN().to(DEVICE)
    state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(state_dict)
    model.eval()
    return model


MODEL = load_model()


def preprocess_image(pil_image: Image.Image) -> torch.Tensor:
    if pil_image.mode == "RGBA":
        background = Image.new("RGBA", pil_image.size, (255, 255, 255, 255))
        pil_image = Image.alpha_composite(background, pil_image)

    gray = ImageOps.grayscale(pil_image)
    arr = np.asarray(gray, dtype=np.float32)

    if arr.mean() > 127:
        arr = 255 - arr

    mask = arr > 20
    if mask.any():
        ys, xs = np.where(mask)
        y1, y2 = ys.min(), ys.max() + 1
        x1, x2 = xs.min(), xs.max() + 1
        arr = arr[y1:y2, x1:x2]

    height, width = arr.shape
    size = max(height, width)
    square = np.zeros((size, size), dtype=np.float32)
    top = (size - height) // 2
    left = (size - width) // 2
    square[top : top + height, left : left + width] = arr

    resized = Image.fromarray(square.astype(np.uint8)).resize(
        (28, 28),
        Image.Resampling.LANCZOS,
    )
    tensor = torch.from_numpy(np.asarray(resized, dtype=np.float32) / 255.0)
    tensor = tensor.view(1, 1, 28, 28)
    tensor = (tensor - MNIST_MEAN) / MNIST_STD
    return tensor.to(DEVICE)


def predict_digit(pil_image: Image.Image):
    tensor = preprocess_image(pil_image)
    with torch.no_grad():
        logits = MODEL(tensor)
        probabilities = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()

    prediction = int(probabilities.argmax())
    top3 = sorted(
        [{"digit": str(i), "confidence": float(probabilities[i])} for i in range(10)],
        key=lambda item: item["confidence"],
        reverse=True,
    )[:3]
    return {
        "prediction": prediction,
        "confidence": float(probabilities[prediction]),
        "top3": top3,
    }


def image_from_request() -> Image.Image:
    if "image" in request.files and request.files["image"].filename:
        return Image.open(request.files["image"].stream).convert("RGBA")

    data_url = request.form.get("canvas", "")
    if "," not in data_url:
        raise ValueError("请先上传图片或在画板上写一个数字。")

    payload = data_url.split(",", 1)[1]
    image_bytes = base64.b64decode(payload)
    return Image.open(BytesIO(image_bytes)).convert("RGBA")


@app.get("/")
def index():
    return render_template_string(PAGE_HTML)


@app.post("/predict")
def predict():
    try:
        pil_image = image_from_request()
        return jsonify({"ok": True, **predict_digit(pil_image)})
    except Exception as exc:  # noqa: BLE001
        return jsonify({"ok": False, "error": str(exc)}), 400


PAGE_HTML = r"""
<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>CNN 手写数字识别</title>
  <style>
    :root {
      --bg: #040807;
      --panel: rgba(9, 22, 20, .78);
      --panel-2: rgba(13, 32, 29, .9);
      --line: rgba(112, 255, 223, .22);
      --text: #ecfff8;
      --muted: #8daaa0;
      --cyan: #46f4d9;
      --lime: #c8ff5c;
      --amber: #ffc857;
      --danger: #ff715f;
    }

    * { box-sizing: border-box; }

    body {
      margin: 0;
      min-height: 100vh;
      color: var(--text);
      font-family: "Microsoft YaHei", "Segoe UI", sans-serif;
      background:
        linear-gradient(rgba(70,244,217,.055) 1px, transparent 1px),
        linear-gradient(90deg, rgba(70,244,217,.055) 1px, transparent 1px),
        radial-gradient(circle at 17% 12%, rgba(70,244,217,.22), transparent 28%),
        radial-gradient(circle at 84% 18%, rgba(200,255,92,.14), transparent 26%),
        radial-gradient(circle at 52% 100%, rgba(255,200,87,.12), transparent 36%),
        var(--bg);
      background-size: 38px 38px, 38px 38px, auto, auto, auto, auto;
    }

    body::before {
      content: "";
      position: fixed;
      inset: 0;
      pointer-events: none;
      background: linear-gradient(180deg, transparent, rgba(70,244,217,.06), transparent);
      mix-blend-mode: screen;
      animation: scan 4.2s ease-in-out infinite;
    }

    @keyframes scan {
      0% { transform: translateY(-120%); opacity: 0; }
      25% { opacity: .8; }
      100% { transform: translateY(120%); opacity: 0; }
    }

    .shell {
      width: min(1180px, calc(100% - 32px));
      margin: 0 auto;
      padding: 28px 0 38px;
    }

    .hero {
      position: relative;
      overflow: hidden;
      min-height: 250px;
      border: 1px solid rgba(70,244,217,.26);
      border-radius: 26px;
      padding: 32px;
      background:
        linear-gradient(118deg, rgba(4,10,9,.97), rgba(9,32,28,.94) 58%, rgba(38,41,16,.86)),
        repeating-linear-gradient(90deg, rgba(255,255,255,.05) 0 1px, transparent 1px 22px);
      box-shadow: 0 30px 90px rgba(0,0,0,.48), inset 0 0 0 1px rgba(255,255,255,.05);
    }

    .hero::after {
      content: "";
      position: absolute;
      right: -10%;
      bottom: -34%;
      width: 72%;
      height: 230px;
      background: linear-gradient(90deg, transparent, rgba(70,244,217,.45), rgba(200,255,92,.28), transparent);
      filter: blur(34px);
      transform: rotate(-8deg);
    }

    .hero-grid {
      position: relative;
      z-index: 1;
      display: grid;
      grid-template-columns: minmax(0, 1.08fr) 410px;
      gap: 28px;
      align-items: center;
    }

    .eyebrow {
      color: var(--lime);
      font-size: 13px;
      font-weight: 800;
      letter-spacing: .22em;
      text-transform: uppercase;
      text-shadow: 0 0 18px rgba(200,255,92,.45);
    }

    h1 {
      margin: 12px 0 14px;
      font-size: clamp(42px, 6vw, 72px);
      line-height: 1.02;
      letter-spacing: 0;
      text-shadow: 0 0 34px rgba(70,244,217,.34);
    }

    .subtitle {
      max-width: 650px;
      margin: 0;
      color: rgba(236,255,248,.72);
      font-size: 16px;
      line-height: 1.7;
    }

    .signals {
      display: flex;
      flex-wrap: wrap;
      gap: 10px;
      margin-top: 22px;
    }

    .signal {
      border: 1px solid rgba(70,244,217,.24);
      border-radius: 999px;
      padding: 8px 12px;
      color: rgba(236,255,248,.84);
      background: rgba(70,244,217,.08);
      font-size: 12px;
    }

    .screen {
      border: 1px solid rgba(70,244,217,.28);
      border-radius: 22px;
      padding: 18px;
      background: linear-gradient(180deg, rgba(9,34,30,.9), rgba(3,12,11,.78));
      box-shadow: inset 0 0 40px rgba(70,244,217,.1), 0 0 40px rgba(70,244,217,.08);
    }

    .digits {
      display: grid;
      grid-template-columns: repeat(5, 1fr);
      gap: 10px;
    }

    .digit {
      display: grid;
      place-items: center;
      aspect-ratio: 1;
      border: 1px solid rgba(236,255,248,.12);
      border-radius: 14px;
      color: rgba(236,255,248,.74);
      background: rgba(255,255,255,.045);
      font-size: 24px;
      font-weight: 850;
    }

    .digit.hot {
      color: #04100d;
      background: linear-gradient(135deg, var(--cyan), var(--lime));
      box-shadow: 0 0 28px rgba(70,244,217,.36);
    }

    .metrics {
      display: grid;
      grid-template-columns: repeat(3, 1fr);
      gap: 12px;
      margin-top: 14px;
    }

    .metric {
      border: 1px solid rgba(70,244,217,.18);
      border-radius: 14px;
      padding: 14px 12px;
      background: rgba(0,0,0,.22);
    }

    .metric b {
      display: block;
      color: #fff;
      font-size: 22px;
      line-height: 1;
    }

    .metric span {
      display: block;
      margin-top: 7px;
      color: rgba(236,255,248,.62);
      font-size: 12px;
    }

    .workspace {
      display: grid;
      grid-template-columns: minmax(0, 1.22fr) minmax(330px, .78fr);
      gap: 20px;
      margin-top: 20px;
    }

    .panel {
      border: 1px solid var(--line);
      border-radius: 24px;
      padding: 20px;
      background: var(--panel);
      backdrop-filter: blur(18px);
      box-shadow: 0 24px 70px rgba(0,0,0,.35), inset 0 0 0 1px rgba(255,255,255,.04);
    }

    .panel h2 {
      margin: 0 0 14px;
      font-size: 18px;
      letter-spacing: 0;
    }

    .tabs {
      display: flex;
      gap: 10px;
      margin-bottom: 16px;
    }

    .tab {
      border: 1px solid rgba(70,244,217,.2);
      border-radius: 14px;
      padding: 10px 14px;
      color: rgba(236,255,248,.78);
      background: rgba(0,0,0,.18);
      cursor: pointer;
      font-weight: 700;
    }

    .tab.active {
      color: #04100d;
      background: linear-gradient(135deg, var(--cyan), var(--lime));
      box-shadow: 0 12px 26px rgba(70,244,217,.2);
    }

    .mode { display: none; }
    .mode.active { display: block; }

    .upload-zone {
      display: grid;
      place-items: center;
      min-height: 390px;
      border: 1px dashed rgba(70,244,217,.28);
      border-radius: 20px;
      background: rgba(0,0,0,.22);
      overflow: hidden;
    }

    .upload-zone img {
      max-width: 100%;
      max-height: 390px;
      object-fit: contain;
    }

    .upload-copy {
      color: var(--muted);
      text-align: center;
      line-height: 1.8;
    }

    .file-input {
      margin-top: 14px;
      color: rgba(236,255,248,.76);
    }

    canvas {
      display: block;
      width: min(100%, 420px);
      aspect-ratio: 1;
      margin: 0 auto;
      border: 1px solid rgba(70,244,217,.28);
      border-radius: 20px;
      background: #000;
      box-shadow: inset 0 0 32px rgba(70,244,217,.08);
      touch-action: none;
      cursor: crosshair;
    }

    .actions {
      display: flex;
      gap: 12px;
      margin-top: 16px;
      flex-wrap: wrap;
    }

    button {
      border: 0;
      border-radius: 15px;
      min-height: 46px;
      padding: 0 18px;
      font: inherit;
      font-weight: 800;
      cursor: pointer;
    }

    .primary {
      flex: 1;
      color: #04100d;
      background: linear-gradient(135deg, var(--cyan), var(--lime));
      box-shadow: 0 16px 34px rgba(70,244,217,.24);
    }

    .secondary {
      color: rgba(236,255,248,.84);
      background: rgba(255,255,255,.08);
      border: 1px solid rgba(236,255,248,.14);
    }

    .result-card {
      min-height: 280px;
      border: 1px solid rgba(200,255,92,.22);
      border-radius: 22px;
      padding: 22px;
      background: linear-gradient(180deg, rgba(11,28,24,.96), rgba(6,15,13,.9));
      box-shadow: inset 0 0 36px rgba(200,255,92,.07);
    }

    .kicker {
      color: var(--muted);
      font-size: 12px;
      letter-spacing: .18em;
      text-transform: uppercase;
    }

    .prediction {
      margin-top: 8px;
      color: var(--lime);
      font-size: 108px;
      font-weight: 900;
      line-height: .96;
      text-shadow: 0 0 30px rgba(200,255,92,.48);
    }

    .confidence {
      color: rgba(236,255,248,.74);
      font-size: 15px;
    }

    .bars {
      margin-top: 24px;
      display: grid;
      gap: 12px;
    }

    .bar-row {
      display: grid;
      grid-template-columns: 32px 1fr 62px;
      gap: 10px;
      align-items: center;
      color: rgba(236,255,248,.76);
      font-size: 13px;
    }

    .bar-track {
      height: 10px;
      border-radius: 999px;
      background: rgba(255,255,255,.08);
      overflow: hidden;
    }

    .bar-fill {
      height: 100%;
      width: 0%;
      border-radius: inherit;
      background: linear-gradient(90deg, var(--cyan), var(--lime));
      box-shadow: 0 0 18px rgba(70,244,217,.3);
      transition: width .45s ease;
    }

    .status {
      margin-top: 14px;
      color: var(--muted);
      min-height: 22px;
    }

    @media (max-width: 920px) {
      .hero-grid,
      .workspace {
        grid-template-columns: 1fr;
      }

      .metrics {
        grid-template-columns: 1fr;
      }
    }
  </style>
</head>
<body>
  <main class="shell">
    <section class="hero">
      <div class="hero-grid">
        <div>
          <div class="eyebrow">NEURAL VISION TERMINAL</div>
          <h1>手写数字识别系统</h1>
          <p class="subtitle">基于 PyTorch CNN 的手写数字识别展示页，支持图片上传与网页手写板输入，实时输出预测类别和 Top-3 置信度。</p>
          <div class="signals">
            <span class="signal">GPU READY</span>
            <span class="signal">CNN ONLINE</span>
            <span class="signal">MNIST MODE</span>
          </div>
        </div>
        <div class="screen">
          <div class="digits">
            <div class="digit">0</div><div class="digit">7</div><div class="digit hot">9</div><div class="digit">2</div><div class="digit">5</div>
            <div class="digit">4</div><div class="digit hot">1</div><div class="digit">8</div><div class="digit">3</div><div class="digit">6</div>
          </div>
          <div class="metrics">
            <div class="metric"><b>0.99378</b><span>Kaggle Score</span></div>
            <div class="metric"><b>10</b><span>Classes</span></div>
            <div class="metric"><b>Top-3</b><span>Confidence</span></div>
          </div>
        </div>
      </div>
    </section>

    <section class="workspace">
      <div class="panel">
        <div class="tabs">
          <button class="tab active" data-mode="upload">上传图片</button>
          <button class="tab" data-mode="draw">手写板</button>
        </div>

        <div id="upload-mode" class="mode active">
          <h2>上传手写数字图片</h2>
          <div class="upload-zone" id="upload-zone">
            <div class="upload-copy">将图片拖放到这里<br />或选择本地图片进行识别</div>
            <img id="preview" hidden alt="上传预览" />
          </div>
          <input class="file-input" id="image-file" type="file" accept="image/*" />
          <div class="actions">
            <button class="primary" id="predict-upload">开始识别</button>
          </div>
        </div>

        <div id="draw-mode" class="mode">
          <h2>在画板上写一个数字</h2>
          <canvas id="draw-canvas" width="420" height="420"></canvas>
          <div class="actions">
            <button class="primary" id="predict-draw">识别手写内容</button>
            <button class="secondary" id="clear-canvas">清空画板</button>
          </div>
        </div>
      </div>

      <aside class="panel result-card">
        <div class="kicker">Neural Prediction</div>
        <div class="prediction" id="prediction">-</div>
        <div class="confidence" id="confidence">等待输入</div>
        <div class="bars" id="bars">
          <div class="bar-row"><span>--</span><div class="bar-track"><div class="bar-fill"></div></div><span>0.00%</span></div>
          <div class="bar-row"><span>--</span><div class="bar-track"><div class="bar-fill"></div></div><span>0.00%</span></div>
          <div class="bar-row"><span>--</span><div class="bar-track"><div class="bar-fill"></div></div><span>0.00%</span></div>
        </div>
        <div class="status" id="status">模型已加载，等待图片或手写输入。</div>
      </aside>
    </section>
  </main>

  <script>
    const tabs = document.querySelectorAll(".tab");
    const uploadMode = document.querySelector("#upload-mode");
    const drawMode = document.querySelector("#draw-mode");
    const fileInput = document.querySelector("#image-file");
    const preview = document.querySelector("#preview");
    const uploadZone = document.querySelector("#upload-zone");
    const statusEl = document.querySelector("#status");
    const predictionEl = document.querySelector("#prediction");
    const confidenceEl = document.querySelector("#confidence");
    const barsEl = document.querySelector("#bars");
    const canvas = document.querySelector("#draw-canvas");
    const ctx = canvas.getContext("2d");

    function switchMode(mode) {
      tabs.forEach(tab => tab.classList.toggle("active", tab.dataset.mode === mode));
      uploadMode.classList.toggle("active", mode === "upload");
      drawMode.classList.toggle("active", mode === "draw");
    }

    tabs.forEach(tab => tab.addEventListener("click", () => switchMode(tab.dataset.mode)));

    function resetCanvas() {
      ctx.fillStyle = "#000";
      ctx.fillRect(0, 0, canvas.width, canvas.height);
      ctx.lineWidth = 24;
      ctx.lineCap = "round";
      ctx.lineJoin = "round";
      ctx.strokeStyle = "#fff";
    }

    resetCanvas();

    let drawing = false;
    function point(event) {
      const rect = canvas.getBoundingClientRect();
      const client = event.touches ? event.touches[0] : event;
      return {
        x: (client.clientX - rect.left) * canvas.width / rect.width,
        y: (client.clientY - rect.top) * canvas.height / rect.height,
      };
    }

    function startDraw(event) {
      drawing = true;
      const p = point(event);
      ctx.beginPath();
      ctx.moveTo(p.x, p.y);
      event.preventDefault();
    }

    function draw(event) {
      if (!drawing) return;
      const p = point(event);
      ctx.lineTo(p.x, p.y);
      ctx.stroke();
      event.preventDefault();
    }

    function stopDraw() {
      drawing = false;
    }

    canvas.addEventListener("mousedown", startDraw);
    canvas.addEventListener("mousemove", draw);
    window.addEventListener("mouseup", stopDraw);
    canvas.addEventListener("touchstart", startDraw, { passive: false });
    canvas.addEventListener("touchmove", draw, { passive: false });
    canvas.addEventListener("touchend", stopDraw);

    document.querySelector("#clear-canvas").addEventListener("click", resetCanvas);

    fileInput.addEventListener("change", () => {
      const file = fileInput.files[0];
      if (!file) return;
      preview.src = URL.createObjectURL(file);
      preview.hidden = false;
      uploadZone.querySelector(".upload-copy").hidden = true;
    });

    uploadZone.addEventListener("dragover", event => {
      event.preventDefault();
    });

    uploadZone.addEventListener("drop", event => {
      event.preventDefault();
      const file = event.dataTransfer.files[0];
      if (!file) return;
      const transfer = new DataTransfer();
      transfer.items.add(file);
      fileInput.files = transfer.files;
      preview.src = URL.createObjectURL(file);
      preview.hidden = false;
      uploadZone.querySelector(".upload-copy").hidden = true;
    });

    function updateResult(data) {
      predictionEl.textContent = data.prediction;
      confidenceEl.textContent = `置信度 ${(data.confidence * 100).toFixed(2)}%`;
      barsEl.innerHTML = "";
      data.top3.forEach(item => {
        const percent = (item.confidence * 100).toFixed(2);
        const row = document.createElement("div");
        row.className = "bar-row";
        row.innerHTML = `
          <span>${item.digit}</span>
          <div class="bar-track"><div class="bar-fill" style="width:${percent}%"></div></div>
          <span>${percent}%</span>
        `;
        barsEl.appendChild(row);
      });
      statusEl.textContent = "识别完成。";
    }

    async function sendForm(formData) {
      statusEl.textContent = "正在识别...";
      const response = await fetch("/predict", { method: "POST", body: formData });
      const data = await response.json();
      if (!data.ok) {
        statusEl.textContent = data.error || "识别失败。";
        return;
      }
      updateResult(data);
    }

    document.querySelector("#predict-upload").addEventListener("click", () => {
      const file = fileInput.files[0];
      if (!file) {
        statusEl.textContent = "请先选择一张图片。";
        return;
      }
      const formData = new FormData();
      formData.append("image", file);
      sendForm(formData);
    });

    document.querySelector("#predict-draw").addEventListener("click", () => {
      const formData = new FormData();
      formData.append("canvas", canvas.toDataURL("image/png"));
      sendForm(formData);
    });
  </script>
</body>
</html>
"""


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "7860"))
    app.run(host="0.0.0.0", port=port, debug=False)
