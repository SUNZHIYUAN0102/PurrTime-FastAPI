from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
import numpy as np
import joblib

# === 初始化 FastAPI 应用 ===
app = FastAPI(title="Cat Behavior Classifier")

# === 你模型的输入特征长度 ===
EXPECTED_FEATURE_DIM = 12

# === 输入格式定义 ===
class InputFeatures(BaseModel):
    features: list[float]

# === 加载模型与预处理器 ===
try:
    model = torch.jit.load("model.pt")
    model.eval()

    scaler = joblib.load("scaler.pkl")
    label_encoder = joblib.load("label_encoder.pkl")

    print("✅ 模型、scaler、encoder 加载成功！")

except Exception as e:
    print("❌ 加载失败:", e)
    raise RuntimeError("模型初始化失败：" + str(e))


# === 推理接口 ===
@app.post("/predict")
async def predict(input: InputFeatures):
    try:
        if len(input.features) != EXPECTED_FEATURE_DIM:
            raise HTTPException(status_code=400, detail=f"输入特征数量必须是 {EXPECTED_FEATURE_DIM}，实际为 {len(input.features)}")

        # 1. 标准化
        input_np = np.array(input.features).reshape(1, -1)
        input_scaled = scaler.transform(input_np)
        input_tensor = torch.tensor(input_scaled, dtype=torch.float32)

        # 2. 推理 + 概率
        with torch.no_grad():
            output = model(input_tensor)
            probabilities = torch.softmax(output, dim=1).squeeze().numpy()
            predicted_index = int(np.argmax(probabilities))
            predicted_label = label_encoder.inverse_transform([predicted_index])[0]

        # 3. 构建概率字典
        class_names = label_encoder.classes_
        prob_dict = {class_name: float(probabilities[i]) for i, class_name in enumerate(class_names)}

        return {
            "class_id": predicted_index,
            "class_name": predicted_label,
            "probabilities": prob_dict,
        }

    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"服务器错误: {str(e)}")

