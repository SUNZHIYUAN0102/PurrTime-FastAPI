# main.py

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
import numpy as np
import joblib

# === 初始化 FastAPI 应用 ===
app = FastAPI(title="Cat Behavior Classifier")

# === 数据模型，用于接收输入 ===
class InputFeatures(BaseModel):
    features: list[float]  # e.g. 30维的加速度特征

# === 加载模型和预处理器（只加载一次） ===
try:
    model = torch.jit.load("model.pt")
    model.eval()

    scaler = joblib.load("scaler.pkl")
    label_encoder = joblib.load("label_encoder.pkl")

    print("✅ 模型与预处理器加载成功！")
except Exception as e:
    print("❌ 加载失败:", e)
    raise RuntimeError("初始化失败：" + str(e))

# === 预测接口 ===
@app.post("/predict")
async def predict(input: InputFeatures):
    try:
        # 转换输入为 numpy 并标准化
        input_np = np.array(input.features).reshape(1, -1)
        input_scaled = scaler.transform(input_np)
        input_tensor = torch.tensor(input_scaled, dtype=torch.float32)

        # 模型推理
        with torch.no_grad():
            output = model(input_tensor)
            predicted_class = torch.argmax(output, dim=1).item()
            predicted_label = label_encoder.inverse_transform([predicted_class])[0]

        return {
            "class_id": predicted_class,
            "class_name": predicted_label
        }

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"预测失败: {str(e)}")
