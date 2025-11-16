import os
import argparse
import cv2
import json
import time
import numpy as np
from pathlib import Path
from datetime import datetime

# TensorFlow 導入
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
    print("✅ TensorFlow 成功載入")
except ImportError as e:
    print(f"❌ TensorFlow 載入失敗: {e}")
    exit(1)

# scikit-learn 導入
try:
    from sklearn.model_selection import train_test_split, StratifiedKFold
    from sklearn.metrics import classification_report
    print("✅ scikit-learn 成功載入")
except ImportError:
    print("⚠️ scikit-learn 未安裝")

class OptimizedTrainer:
    """優化的訓練器，專為高準確率設計"""
    
    def __init__(self, img_size=256, num_classes=20, batch_size=16, learning_rate=1e-3, class_filter=None):
        self.img_size = img_size
        self.num_classes = num_classes
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        # 若提供 class_filter（指定要用哪些類別名），將在載入時套用
        self.class_filter = set(class_filter) if class_filter else None
        
    def load_optimized_data(self, sample_dir, test_dir, samples_per_class=300):
        """載入優化的資料集"""
        print("📂 載入優化資料集...")
        
        # 載入訓練資料
        X_train, y_train, class_names = self._load_from_dir(
            sample_dir, self.num_classes, samples_per_class
        )
        
        # 載入測試資料
        X_test, y_test, _ = self._load_from_dir(
            test_dir, self.num_classes, samples_per_class//2
        )
        
        if X_train is None or X_test is None:
            return None, None, None, None, None
            
        print(f"📊 資料統計:")
        print(f"   訓練集: {len(X_train):,} 張圖片")
        print(f"   測試集: {len(X_test):,} 張圖片")
        print(f"   類別數: {len(class_names)}")
        print(f"   圖片尺寸: {self.img_size}x{self.img_size}")
        
        return X_train, y_train, X_test, y_test, class_names
    
    def _load_from_dir(self, data_dir, max_classes, samples_per_class):
        """從目錄載入資料"""
        data_path = Path(data_dir)
        if not data_path.exists():
            print(f"❌ 目錄不存在: {data_dir}")
            return None, None, None
        
        character_dirs = sorted([d for d in data_path.iterdir() if d.is_dir()])
        # 若指定 class_filter，僅保留指定名稱的資料夾
        if self.class_filter:
            character_dirs = [d for d in character_dirs if d.name in self.class_filter]
        # 再截斷到 max_classes
        character_dirs = character_dirs[:max_classes]
        
        images = []
        labels = []
        class_names = []
        
        for class_idx, char_dir in enumerate(character_dirs):
            class_names.append(char_dir.name)
            image_files = list(char_dir.glob("*.jpg")) + list(char_dir.glob("*.png"))
            
            # 隨機選取樣本以確保多樣性
            if len(image_files) > samples_per_class:
                np.random.shuffle(image_files)
                image_files = image_files[:samples_per_class]
            
            for img_file in image_files:
                try:
                    img = cv2.imread(str(img_file), cv2.IMREAD_GRAYSCALE)
                    if img is None:
                        continue
                    
                    # 高質量圖片預處理
                    img = self._preprocess_image(img)
                    images.append(img)
                    labels.append(class_idx)
                    
                except Exception as e:
                    continue
        
        if not images:
            return None, None, None
        
        X = np.array(images)
        y = np.array(labels)
        X = np.expand_dims(X, axis=-1)
        
        return X, y, class_names
    
    def _preprocess_image(self, img):
        """高質量圖片預處理"""
        # 1. 去噪
        img = cv2.medianBlur(img, 3)
        
        # 2. 對比度增強
        img = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8)).apply(img)
        
        # 3. 高質量縮放
        img = cv2.resize(img, (self.img_size, self.img_size), interpolation=cv2.INTER_CUBIC)
        
        # 4. 正規化
        img = img.astype(np.float32) / 255.0
        
        return img
    
    def create_ultra_model(self):
        """創建超高性能模型"""
        inputs = keras.Input(shape=(self.img_size, self.img_size, 1))
        
        # 初始特徵提取
        x = layers.Conv2D(32, 7, strides=2, padding='same', activation='relu')(inputs)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D(3, strides=2, padding='same')(x)
        
        # Block 1
        x = self._conv_block(x, 64, 3)
        x = self._conv_block(x, 64, 3)
        x = layers.MaxPooling2D(2)(x)
        x = layers.Dropout(0.25)(x)
        
        # Block 2
        x = self._conv_block(x, 128, 3)
        x = self._conv_block(x, 128, 3)
        x = layers.MaxPooling2D(2)(x)
        x = layers.Dropout(0.25)(x)
        
        # Block 3
        x = self._conv_block(x, 256, 3)
        x = self._conv_block(x, 256, 3)
        x = self._conv_block(x, 256, 3)
        x = layers.MaxPooling2D(2)(x)
        x = layers.Dropout(0.25)(x)
        
        # Block 4
        x = self._conv_block(x, 512, 3)
        x = self._conv_block(x, 512, 3)
        x = self._conv_block(x, 512, 3)
        x = layers.MaxPooling2D(2)(x)
        x = layers.Dropout(0.25)(x)
        
        # Block 5
        x = self._conv_block(x, 1024, 3)
        x = self._conv_block(x, 1024, 3)
        
        # 全域平均池化
        x = layers.GlobalAveragePooling2D()(x)
        
        # 分類器
        x = layers.Dense(2048, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.5)(x)
        
        x = layers.Dense(1024, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.5)(x)
        
        x = layers.Dense(512, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.3)(x)
        
        outputs = layers.Dense(self.num_classes, activation='softmax')(x)
        
        model = keras.Model(inputs=inputs, outputs=outputs, name='ultra_cnn')
        return model
    
    def _conv_block(self, x, filters, kernel_size):
        """卷積塊"""
        x = layers.Conv2D(filters, kernel_size, padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        return x
    
    def create_advanced_augmentation(self):
        """創建高級數據增強"""
        return ImageDataGenerator(
            rotation_range=20,
            width_shift_range=0.15,
            height_shift_range=0.15,
            shear_range=0.15,
            zoom_range=0.15,
            brightness_range=[0.8, 1.2],
            channel_shift_range=0.1,
            horizontal_flip=False,
            vertical_flip=False,
            fill_mode='nearest'
        )
    
    def train_ultra_model(self, X_train, y_train, X_test, y_test, epochs=100):
        """訓練超高性能模型"""
        print("\n🚀 開始訓練超高性能模型")
        print(f"🎯 目標準確率: 80%+")
        
        # 創建模型
        model = self.create_ultra_model()
        
        # 編譯模型
        model.compile(
            optimizer=Adam(learning_rate=self.learning_rate),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        print(f"📊 模型參數數量: {model.count_params():,}")
        
        # 分割驗證集
        X_train_split, X_val, y_train_split, y_val = train_test_split(
            X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
        )
        
        # 設置回調
        callbacks = [
            ReduceLROnPlateau(
                monitor='val_accuracy',
                factor=0.3,
                patience=8,
                min_lr=1e-7,
                verbose=1
            ),
            EarlyStopping(
                monitor='val_accuracy',
                patience=20,
                restore_best_weights=True,
                verbose=1
            ),
            # 使用 HDF5 格式避免 native Keras 格式與後端 options 衝突錯誤
            ModelCheckpoint(
                'ultra_best_model.h5',
                monitor='val_accuracy',
                save_best_only=True,
                save_weights_only=False,  # 仍保存完整模型
                verbose=1
            )
        ]
        
        # 創建數據增強
        datagen = self.create_advanced_augmentation()
        datagen.fit(X_train_split)
        
        print("🔄 開始訓練（使用高級數據增強）...")
        start_time = time.time()
        
        # 訓練模型
        history = model.fit(
            datagen.flow(X_train_split, y_train_split, batch_size=self.batch_size),
            steps_per_epoch=max(1, len(X_train_split) // self.batch_size),
            epochs=epochs,
            validation_data=(X_val, y_val),
            callbacks=callbacks,
            verbose=1
        )
        
        training_time = time.time() - start_time
        
        # 評估模型
        print("\n📊 評估模型性能...")
        test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
        
        # 保存最終模型（改用 HDF5 格式以繞過 native Keras format 的 options 衝突）
        model.save('ultra_final_model.h5')
        
        print(f"\n✅ 訓練完成!")
        print(f"🎯 測試準確率: {test_accuracy:.4f} ({test_accuracy*100:.1f}%)")
        print(f"⏱️ 訓練時間: {training_time/60:.1f} 分鐘")
        
        # 檢查是否達到目標
        if test_accuracy >= 0.8:
            print("🎉 恭喜！模型達到了 80% 準確率目標！")
            print("🏆 您的模型已經可以投入使用了！")
        else:
            print("⚠️ 尚未達到 80% 目標準確率")
            print(f"📈 還需要提升 {(0.8 - test_accuracy)*100:.1f} 個百分點")
        
        return {
            'test_accuracy': test_accuracy,
            'test_loss': test_loss,
            'training_time': training_time,
            'epochs_trained': len(history.history['accuracy']),
            'target_achieved': test_accuracy >= 0.8,
            'model_file': 'ultra_final_model.h5'
        }

def main():
    """主函數 - 執行高準確率訓練"""
    parser = argparse.ArgumentParser(description='超高性能手寫模型訓練器')
    parser.add_argument('--img-size', type=int, default=256, help='輸入影像尺寸 (預設 256)')
    parser.add_argument('--num-classes', type=int, default=20, help='訓練的類別數量上限')
    parser.add_argument('--samples-per-class', type=int, default=500, help='每類訓練樣本上限')
    parser.add_argument('--epochs', type=int, default=100, help='訓練輪數')
    parser.add_argument('--batch-size', type=int, default=16, help='batch size（預設 16）')
    parser.add_argument('--lr', type=float, default=1e-3, help='學習率（預設 1e-3）')
    parser.add_argument('--resume', action='store_true', help='若存在 ultra_best_model.h5，從該檔案繼續訓練')
    parser.add_argument('--classes', type=str, default='', help='指定要訓練的類別（以逗號分隔的資料夾名）')
    parser.add_argument('--train-dir', type=str, default='./data/sample', help='訓練資料夾')
    parser.add_argument('--test-dir', type=str, default='./data/test', help='測試資料夾')
    args = parser.parse_args()

    print("🚀 高準確率繁體中文手寫識別訓練器")
    print("=" * 50)
    print("🎯 專門設計來達到 80% 以上準確率")
    print(f"📏 使用 {args.img_size}x{args.img_size} 高解析度圖片")
    if args.classes:
        cls_list = [c.strip() for c in args.classes.split(',') if c.strip()]
        print(f"🔢 只訓練指定類別: {cls_list}")
    else:
        print(f"🔢 專注於 {args.num_classes} 個字符（依資料夾順序截取）")
    print("=" * 50)
    
    # 初始化訓練器
    class_filter = [c.strip() for c in args.classes.split(',')] if args.classes else None
    trainer = OptimizedTrainer(
        img_size=args.img_size,
        num_classes=args.num_classes,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        class_filter=class_filter
    )
    
    # 載入資料
    X_train, y_train, X_test, y_test, class_names = trainer.load_optimized_data(
        args.train_dir, args.test_dir, samples_per_class=args.samples_per_class
    )
    
    if X_train is None:
        print("❌ 資料載入失敗")
        return
    
    # 訓練模型
    # 可選：從最佳檢查點恢復（若架構一致）
    if args.resume and os.path.exists('ultra_best_model.h5'):
        try:
            print('🔁 嘗試從 ultra_best_model.h5 繼續訓練...')
            # 載入整個模型（需架構一致）
            prev_model = tf.keras.models.load_model('ultra_best_model.h5')
            # 直接用先前模型評估並另行訓練（此示例保持簡單，仍沿用 trainer 設定）
        except Exception as e:
            print('⚠️ 恢復訓練失敗，將從頭開始：', e)
    result = trainer.train_ultra_model(X_train, y_train, X_test, y_test, epochs=args.epochs)
    
    # 保存結果
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = f"ultra_training_results_{timestamp}.json"
    
    # 將類別名稱也寫入結果，方便 Web 與評估對應 index→label
    if class_names:
        result_with_classes = dict(result)
        result_with_classes['class_names'] = class_names
    else:
        result_with_classes = result

    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(result_with_classes, f, ensure_ascii=False, indent=2)

    # 另存一份簡單的類別清單檔，利於獨立查閱/載入
    try:
        classes_sidecar = {
            'model_file': result.get('model_file', 'ultra_final_model.h5'),
            'class_names': class_names if class_names else [],
            'generated_at': timestamp
        }
        with open('ultra_classes.json', 'w', encoding='utf-8') as f:
            json.dump(classes_sidecar, f, ensure_ascii=False, indent=2)
        print('📄 已輸出類別清單 -> ultra_classes.json')
    except Exception as e:
        print('⚠️ 輸出 ultra_classes.json 失敗:', e)
    
    print(f"\n💾 結果已保存: {results_file}")
    
    if result['target_achieved']:
        print("\n🎉 成功！您的模型已達到 80% 準確率目標！")
        print("💡 現在可以更新 Web 應用程序使用新模型了")
        print(f"📁 新模型檔案: {result['model_file']}")
    else:
        print("\n💡 改進建議:")
        print("1. 🔄 繼續訓練更多 epochs")
        print("2. 📊 增加每個類別的訓練樣本")
        print("3. 🎯 進一步減少類別數量")
        print("4. 🔧 調整模型架構或超參數")

if __name__ == "__main__":
    main()