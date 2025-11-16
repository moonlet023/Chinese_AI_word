import os
import cv2
import json
import numpy as np
from pathlib import Path
from flask import Flask, render_template, request, jsonify, send_from_directory
from werkzeug.utils import secure_filename
import base64
from io import BytesIO
from PIL import Image

# TensorFlow 導入（延遲導入以處理錯誤）
tf = None
try:
    import tensorflow as tf
    print("✅ TensorFlow 成功載入")
except ImportError:
    print("❌ TensorFlow 未安裝")

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB 最大檔案大小

# 確保上傳目錄存在
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# 全域變數
model = None
class_names = []
IMG_SIZE = 64

def _infer_model_input_size_and_channels():
    """從已載入的模型推斷輸入大小和通道數，回傳 (width, height, channels)
    若無法推斷，回傳 (IMG_SIZE, IMG_SIZE, 1)
    """
    try:
        # 當模型尚未載入時，回傳預設值
        if model is None:
            return (IMG_SIZE, IMG_SIZE, 1)

        # 盡量從 model 找到 input shape
        shape = None
        # model.input_shape 常見為 (None, H, W, C) 或 (H, W, C)
        if hasattr(model, 'input_shape') and model.input_shape is not None:
            shape = model.input_shape
        elif hasattr(model, 'inputs') and len(model.inputs) > 0:
            try:
                # inputs[0].shape 可能是 TensorShape
                shape = tuple(model.inputs[0].shape.as_list())
            except Exception:
                shape = tuple(model.inputs[0].shape)

        if shape is None:
            return (IMG_SIZE, IMG_SIZE, 1)

        # Normalize shape to (batch?, H, W, C) or (H, W, C)
        s = list(shape)
        # remove None batch dims
        s = [x for x in s if x is not None]

        # 最常見是 [H, W, C]
        if len(s) == 3:
            h, w, c = s
        elif len(s) == 2:
            # 可能是 (H, W)
            h, w = s
            c = 1
        else:
            # fallback
            return (IMG_SIZE, IMG_SIZE, 1)

        # 轉為 int，並以預設值作為保險
        h = int(h) if h is not None else IMG_SIZE
        w = int(w) if w is not None else IMG_SIZE
        c = int(c) if c is not None else 1

        # 回傳順序為 (width, height, channels) 以方便 cv2.resize
        return (w, h, c)
    except Exception:
        return (IMG_SIZE, IMG_SIZE, 1)

def _find_latest_results_json():
    """尋找最近更新的結果 JSON（支援多種命名樣式）"""
    patterns = [
        'ultra_training_results_*.json',
        'high_accuracy_*_results_*.json',
        'practical_results_*.json',
        'quick_test_results_*.json',
        'model_results_*.json',
    ]
    candidates = []
    for pat in patterns:
        candidates.extend(list(Path('.').glob(pat)))
    if not candidates:
        return None
    return max(candidates, key=lambda x: x.stat().st_mtime)


def load_model_and_classes():
    """載入最佳模型和類別名稱"""
    global model, class_names
    
    if tf is None:
        print("❌ TensorFlow 未可用，無法載入模型")
        return False
    
    # 使用最新的高準確率模型（由更新器注入此檔名）
    model_file = "ultra_final_model.h5"
    
    if not os.path.exists(model_file):
        print(f"❌ 模型文件不存在: {model_file}")
        return False
    
    try:
        model = tf.keras.models.load_model(model_file)
        print(f"✅ 高準確率模型載入成功: {model_file}")

    # 嘗試從最近的結果 JSON 讀取資訊（accuracy、class_names）
        latest_json = _find_latest_results_json()
        results = None
        if latest_json is not None:
            try:
                with open(latest_json, 'r', encoding='utf-8') as f:
                    results = json.load(f)
            except Exception as e:
                print(f"⚠️ 無法讀取結果檔案 {latest_json}: {e}")

        # 顯示準確率（若結果檔提供）
        acc = None
        if isinstance(results, dict):
            for k in ['overall_accuracy', 'val_accuracy', 'accuracy', 'best_val_accuracy']:
                if k in results and isinstance(results[k], (int, float)):
                    acc = float(results[k])
                    break
            if acc is not None:
                print(f"   模型準確率: {acc:.1f}%")

        # 設置類別名稱：優先使用模型旁的類別清單檔案，其次使用結果檔 class_names，最後才從資料夾載入
        loaded_mapping = False

        # 1) 嘗試 ultra_classes.json 或 *classes*.json
        sidecar_jsons = list(Path('.').glob('ultra_classes.json')) + list(Path('.').glob('*classes*.json'))
        if sidecar_jsons:
            # 取最近的
            best_sidecar = max(sidecar_jsons, key=lambda x: x.stat().st_mtime)
            try:
                with open(best_sidecar, 'r', encoding='utf-8') as f:
                    sc = json.load(f)
                if isinstance(sc, dict) and isinstance(sc.get('class_names'), list) and sc['class_names']:
                    class_names = sc['class_names']
                    loaded_mapping = True
                    print(f"✅ 從 {best_sidecar.name} 載入 {len(class_names)} 個字符類別")
            except Exception as e:
                print('⚠️ 讀取類別清單檔案失敗:', e)

        # 2) 若 sidecar 無，使用結果檔提供的 class_names
        if not loaded_mapping and isinstance(results, dict) and isinstance(results.get('class_names'), list) and results['class_names']:
            class_names = results['class_names']
            loaded_mapping = True
            print(f"✅ 從結果檔載入 {len(class_names)} 個字符類別")

        # 3) 仍無 -> 從資料目錄載入（可能與實際模型不一致，僅為後備）
        if not loaded_mapping:
            load_class_names()
        
        return True
        
    except Exception as e:
        print(f"❌ 模型載入失敗: {e}")
        return False

def load_class_names():
    """載入字符類別名稱"""
    global class_names
    
    sample_dir = Path('./data/sample')
    if sample_dir.exists():
        class_names = sorted([d.name for d in sample_dir.iterdir() if d.is_dir()])
        print(f"✅ 載入 {len(class_names)} 個字符類別")
    else:
        print("❌ 未找到 sample 資料目錄")

def preprocess_image(image_path_or_array, target_size=None):
    """
    預處理圖片以符合模型輸入要求
    
    Args:
        image_path_or_array: 圖片路徑或 numpy array
        target_size: 目標大小
    
    Returns:
        處理後的圖片 array
    """
    try:
        # 讀取圖片（支援路徑、PIL Image、numpy array）
        if isinstance(image_path_or_array, (str, Path)):
            img = cv2.imread(str(image_path_or_array), cv2.IMREAD_UNCHANGED)
        elif 'PIL' in str(type(image_path_or_array)):
            # PIL Image
            pil_img = image_path_or_array
            img = np.array(pil_img)
        else:
            img = image_path_or_array

        if img is None:
            raise ValueError("無法讀取圖片")

        # 推斷模型期望的輸入尺寸與通道
        target_w, target_h, target_c = _infer_model_input_size_and_channels()

        # 外部 override
        if target_size is not None:
            target_w, target_h = target_size

        # 若圖片為 PIL 轉 numpy 後，形狀可能是 (H,W) or (H,W,3) or (H,W,4)
        if img.ndim == 3 and img.shape[2] == 4:
            # RGBA -> RGB
            img = img[:, :, :3]

        # 若為彩色 (3)，轉為灰階（大多數模型為灰階），但若模型期望多通道則保留
        if img.ndim == 3 and img.shape[2] == 3 and int(target_c) == 1:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        elif img.ndim == 2:
            gray = img
        elif img.ndim == 3 and int(target_c) != 1:
            # 模型需要多通道 (例如 3)，直接轉換到 BGR 並使用
            # OpenCV uses BGR ordering
            if img.shape[2] == 3:
                img_rgb = img
            else:
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            # resize 彩色
            img_resized_color = cv2.resize(img_rgb, (int(target_w), int(target_h)), interpolation=cv2.INTER_AREA)
            img_out = img_resized_color.astype(np.float32) / 255.0
            img_out = np.expand_dims(img_out, axis=0)
            return img_out
        else:
            # 兜底
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img

        # resize grayscale
        img_resized = cv2.resize(gray, (int(target_w), int(target_h)), interpolation=cv2.INTER_AREA)

        # normalize
        img_resized = img_resized.astype(np.float32) / 255.0

        # channel handling
        if int(target_c) == 1:
            img_out = np.expand_dims(img_resized, axis=-1)
        else:
            # replicate gray to channels
            img_out = np.stack([img_resized] * int(target_c), axis=-1)

        # add batch
        img_out = np.expand_dims(img_out, axis=0)

        return img_out

    except Exception as e:
        print(f"❌ 圖片預處理錯誤: {e}")
        return None

@app.route('/')
def index():
    """主頁面"""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """預測手寫字符"""
    if model is None:
        return jsonify({'error': '模型未載入'}), 500
    
    try:
        # 檢查是否有檔案上傳
        if 'file' not in request.files:
            return jsonify({'error': '未找到檔案'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': '未選擇檔案'}), 400
        
        # 保存上傳的檔案
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        
        # 預處理圖片（會自動依模型輸入大小調整）
        processed_img = preprocess_image(file_path, target_size=None)
        if processed_img is None:
            return jsonify({'error': '圖片處理失敗'}), 400
        
        # 進行預測
        predictions = model.predict(processed_img, verbose=0)
        
        # 獲取預測結果
        predicted_class_idx = np.argmax(predictions[0])
        confidence = float(predictions[0][predicted_class_idx])
        
        # 獲取前 5 個預測
        top_5_idx = np.argsort(predictions[0])[-5:][::-1]
        top_5_predictions = [
            {
                'character': class_names[idx] if idx < len(class_names) else f'Class_{idx}',
                'confidence': float(predictions[0][idx])
            }
            for idx in top_5_idx
        ]
        
        # 清理暫存檔案
        try:
            os.remove(file_path)
        except:
            pass
        
        return jsonify({
            'predicted_character': class_names[predicted_class_idx] if predicted_class_idx < len(class_names) else f'Class_{predicted_class_idx}',
            'confidence': confidence,
            'top_5': top_5_predictions
        })
        
    except Exception as e:
        print(f"❌ 預測錯誤: {e}")
        return jsonify({'error': f'預測失敗: {str(e)}'}), 500

@app.route('/canvas_predict', methods=['POST'])
def canvas_predict():
    """從畫布預測手寫字符"""
    if model is None:
        return jsonify({'error': '模型未載入'}), 500
    
    try:
        # 獲取 base64 圖片資料
        data = request.get_json()
        image_data = data['image']
        
        # 解碼 base64 圖片
        image_data = image_data.split(',')[1]  # 移除 "data:image/png;base64," 前綴
        image_binary = base64.b64decode(image_data)
        
        # 轉換為 PIL Image
        pil_image = Image.open(BytesIO(image_binary))
        
        # 轉換為 numpy array
        img_array = np.array(pil_image)
        
        # 如果是 RGBA，轉換為 RGB
        if img_array.shape[-1] == 4:
            img_array = img_array[:, :, :3]
        
        # 預處理圖片（會自動依模型輸入大小調整）
        processed_img = preprocess_image(img_array, target_size=None)
        if processed_img is None:
            return jsonify({'error': '圖片處理失敗'}), 400
        
        # 進行預測
        predictions = model.predict(processed_img, verbose=0)
        
        # 獲取預測結果
        predicted_class_idx = np.argmax(predictions[0])
        confidence = float(predictions[0][predicted_class_idx])
        
        # 獲取前 5 個預測
        top_5_idx = np.argsort(predictions[0])[-5:][::-1]
        top_5_predictions = [
            {
                'character': class_names[idx] if idx < len(class_names) else f'Class_{idx}',
                'confidence': float(predictions[0][idx])
            }
            for idx in top_5_idx
        ]
        
        return jsonify({
            'predicted_character': class_names[predicted_class_idx] if predicted_class_idx < len(class_names) else f'Class_{predicted_class_idx}',
            'confidence': confidence,
            'top_5': top_5_predictions
        })
        
    except Exception as e:
        print(f"❌ 畫布預測錯誤: {e}")
        return jsonify({'error': f'預測失敗: {str(e)}'}), 500

@app.route('/model_info')
def model_info():
    """獲取模型資訊"""
    if model is None:
        return jsonify({'error': '模型未載入'}), 500
    
    try:
        latest_result = _find_latest_results_json()
        if latest_result:
            try:
                with open(latest_result, 'r', encoding='utf-8') as f:
                    results = json.load(f)
            except Exception as e:
                results = { 'error': f'讀取結果檔案失敗: {e}' }

            return jsonify({
                'total_classes': len(class_names),
                'model_results': results,
                'class_names': class_names[:20]  # 只顯示前 20 個類別
            })
        else:
            return jsonify({
                'total_classes': len(class_names),
                'class_names': class_names[:20]
            })
    except Exception as e:
        return jsonify({'error': f'獲取模型資訊失敗: {str(e)}'}), 500

@app.route('/supported_classes')
def supported_classes():
    """回傳完整可識別的字符清單（可能很多，請注意大小）"""
    if model is None:
        return jsonify({'error': '模型未載入'}), 500
    return jsonify({
        'total': len(class_names),
        'class_names': class_names
    })

if __name__ == '__main__':
    print("🚀 繁體中文手寫字符識別 Web 應用程序")
    print("=" * 50)
    
    # 載入模型
    if load_model_and_classes():
        print("🌐 啟動 Web 服務器...")
        app.run(debug=True, host='0.0.0.0', port=5000)
    else:
        print("❌ 無法載入模型，請先訓練模型")
        print("   執行: python simple_ai_training.py --epochs 10")