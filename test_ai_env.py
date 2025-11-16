#!/usr/bin/env python3
"""
快速測試 AI 模型訓練程式
"""

import sys
import os
from pathlib import Path

def test_imports():
    """測試必要套件是否已安裝"""
    print("🔍 測試套件導入...")
    
    try:
        import tensorflow as tf
        try:
            version = tf.__version__
        except:
            version = "已安裝"
        print(f"✅ TensorFlow {version}")
    except ImportError as e:
        print(f"❌ TensorFlow 導入失敗: {e}")
        return False
    
    try:
        import numpy as np
        print(f"✅ NumPy {np.__version__}")
    except ImportError as e:
        print(f"❌ NumPy 導入失敗: {e}")
        return False
    
    try:
        import cv2
        print(f"✅ OpenCV {cv2.__version__}")
    except ImportError as e:
        print(f"❌ OpenCV 導入失敗: {e}")
        return False
    
    try:
        import sklearn
        print(f"✅ Scikit-learn {sklearn.__version__}")
    except ImportError as e:
        print(f"❌ Scikit-learn 導入失敗: {e}")
        return False
    
    try:
        import matplotlib
        print(f"✅ Matplotlib {matplotlib.__version__}")
    except ImportError as e:
        print(f"⚠️ Matplotlib 可能未安裝: {e}")
    
    return True

def test_data_paths():
    """測試資料路徑是否存在"""
    print("\n📂 檢查資料路徑...")
    
    sample_dir = Path("./data/sample")
    test_dir = Path("./data/test")
    
    if sample_dir.exists():
        sample_chars = len([d for d in sample_dir.iterdir() if d.is_dir()])
        print(f"✅ Sample 目錄存在，包含 {sample_chars} 個字符")
    else:
        print(f"❌ Sample 目錄不存在: {sample_dir}")
        return False
    
    if test_dir.exists():
        test_chars = len([d for d in test_dir.iterdir() if d.is_dir()])
        print(f"✅ Test 目錄存在，包含 {test_chars} 個字符")
    else:
        print(f"❌ Test 目錄不存在: {test_dir}")
        return False
    
    return True

def create_simple_test():
    """創建簡單的模型測試"""
    print("\n🧪 創建簡單模型測試...")
    
    try:
        import tensorflow as tf
        try:
            from tensorflow import keras
            from tensorflow.keras import layers
        except ImportError:
            import keras
            from keras import layers
        import numpy as np
        
        # 創建虛擬資料
        X = np.random.random((100, 64, 64, 1))
        y = np.random.randint(0, 10, (100,))
        y_cat = keras.utils.to_categorical(y, 10)
        
        # 創建簡單模型
        model = keras.Sequential([
            layers.Conv2D(16, (3, 3), activation='relu', input_shape=(64, 64, 1)),
            layers.MaxPooling2D((2, 2)),
            layers.Flatten(),
            layers.Dense(32, activation='relu'),
            layers.Dense(10, activation='softmax')
        ])
        
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        
        # 測試訓練
        print("🚀 測試模型訓練...")
        model.fit(X, y_cat, epochs=1, batch_size=16, verbose=0)
        
        print("✅ 模型測試成功！")
        return True
        
    except Exception as e:
        print(f"❌ 模型測試失敗: {e}")
        return False

def main():
    """主測試函數"""
    print("🧪 AI 模型環境測試")
    print("=" * 40)
    
    # 測試套件導入
    if not test_imports():
        print("\n❌ 套件測試失敗，請確認所有依賴套件已正確安裝")
        return
    
    # 測試資料路徑
    data_ready = test_data_paths()
    
    # 測試簡單模型
    if not create_simple_test():
        print("\n❌ 模型測試失敗")
        return
    
    print("\n🎉 環境測試完成！")
    
    if data_ready:
        print("\n💡 建議執行步驟:")
        print("1. 首先運行 main.py 來處理資料:")
        print("   python main.py")
        print("2. 然後運行 AI 訓練:")
        print("   python AItraining.py")
    else:
        print("\n⚠️ 資料準備:")
        print("1. 請先運行 main.py 來處理和分離資料:")
        print("   python main.py")
        print("2. 確認產生了 data/sample 和 data/test 目錄")
        print("3. 然後再運行 AI 訓練:")
        print("   python AItraining.py")

if __name__ == "__main__":
    main()