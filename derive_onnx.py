from ultralytics import YOLO
import os

# 1. 加载模型
MODEL_PATH = r"C:\Users\ASUS\PycharmProjects\YOLOv12\runs\detect\train4\weights\best.pt"
model = YOLO(MODEL_PATH)

# 2. 导出为 ONNX 格式（img sz 需与训练时一致，这里设为640）
export_results = model.export(
    format="onnx",
    imgsz=640,
    batch=1,  # 部署时通常用批量1
    opset=12,  # ONNX算子集版本，兼容大多数框架（12是稳定版本）
    simplify=True  # 简化ONNX模型，减小体积并提升推理速度
)

print(f"✅ ONNX模型导出完成！保存路径：{export_results}")
# 导出的模型默认保存到：runs/detect/train4/weights/best.onnx