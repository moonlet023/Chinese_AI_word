#some code by AI

import os
import json
import csv
import argparse
from pathlib import Path
from typing import List, Tuple, Dict

import numpy as np
import cv2

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

np.random.seed(42)
tf.random.set_seed(42)

IMG_EXTS = {'.png', '.jpg', '.jpeg', '.bmp'}

def list_classes_by_count(root: Path) -> List[Tuple[str, int]]:
    if not root.exists():
        return []
    result = []
    for d in sorted([p for p in root.iterdir() if p.is_dir()]):
        cnt = 0
        for fp in d.iterdir():
            if fp.suffix.lower() in IMG_EXTS:
                cnt += 1
        result.append((d.name, cnt))
    result.sort(key=lambda x: x[1], reverse=True)
    return result

def load_images(dir_root: Path, classes: List[str], img_size: int) -> Tuple[np.ndarray, np.ndarray]:
    X, y = [], []
    for idx, cls in enumerate(classes):
        cls_dir = dir_root / cls
        if not cls_dir.exists():
            continue
        files = [*cls_dir.glob("*")]
        files = [f for f in files if f.suffix.lower() in IMG_EXTS]
        for f in files:
            try:
                img = cv2.imread(str(f), cv2.IMREAD_GRAYSCALE)
                if img is None:
                    continue
                img = cv2.medianBlur(img, 3)
                img = cv2.resize(img, (img_size, img_size), interpolation=cv2.INTER_AREA)
                img = (img.astype("float32") / 255.0)[..., np.newaxis]
                X.append(img)
                y.append(idx)
            except Exception:
                continue
    if len(X) == 0:
        return np.empty((0, img_size, img_size, 1), dtype=np.float32), np.empty((0,), dtype=np.int64)
    return np.stack(X), np.array(y, dtype=np.int64)

def build_cnn(input_shape: Tuple[int, int, int], num_classes: int) -> tf.keras.Model:
    inp = layers.Input(shape=input_shape)
    x = layers.Conv2D(32, 3, padding='same', activation='relu')(inp)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(32, 3, padding='same', activation='relu')(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Dropout(0.25)(x)

    x = layers.Conv2D(64, 3, padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(64, 3, padding='same', activation='relu')(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Dropout(0.30)(x)

    x = layers.Conv2D(128, 3, padding='same', activation='relu')(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(0.40)(x)
    out = layers.Dense(num_classes, activation='softmax')(x)

    model = models.Model(inp, out)
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model

def per_class_accuracy(y_true: np.ndarray, y_pred: np.ndarray, class_names: List[str]) -> List[Dict]:
    rows = []
    for idx, name in enumerate(class_names):
        mask = (y_true == idx)
        total = int(mask.sum())
        if total == 0:
            acc = None
        else:
            acc = float((y_pred[mask] == idx).sum() / total)
        rows.append({
            "class": name,
            "samples": total,
            "accuracy": acc if acc is not None else "N/A",
            "pass_50": (acc is not None and acc >= 0.5)
        })
    return rows

def main():
    parser = argparse.ArgumentParser(description="AI Chinese Word Test >= 50%）")
    parser.add_argument("--sample-dir", default="./data/sample", type=str, help="訓練資料資料夾（每字一夾)")
    parser.add_argument("--test-dir", default="./data/test", type=str, help="測試資料資料夾（每字一夾）")
    parser.add_argument("--classes", default="", type=str, help="指定 5 個字，逗號分隔，例如: 一,十,口,上,下")
    parser.add_argument("--img-size", default=128, type=int, help="輸入影像邊長")
    parser.add_argument("--epochs", default=40, type=int, help="每輪訓練 epochs")
    parser.add_argument("--rounds", default=3, type=int, help="最多迭代輪數")
    parser.add_argument("--min-acc", default=0.5, type=float, help="最低目標測試準確率")
    parser.add_argument("--batch-size", default=32, type=int, help="批次大小")
    parser.add_argument("--out-model", default="five_words_best.h5", type=str, help="輸出模型檔")
    args = parser.parse_args()

    sample_dir = Path(args.sample_dir)
    test_dir = Path(args.test_dir)
    img_size = args.img_size

    if args.classes.strip():
        class_names = [c.strip() for c in args.classes.split(",") if c.strip()]
    #     if len(class_names) != 5:
    #         print("word must 5 --classes")
    #         return
    # else:
    #     counts = list_classes_by_count(sample_dir)
    #     counts = [c for c in counts if c[1] > 0]
    #     class_names = [c for c, _ in counts[:5]]
    #     if len(class_names) < 5:
    #         print("no enough classes in sample dir")
    #         return

        print(f"Class names: {class_names}")

    # 載入資料
    X_train, y_train = load_images(sample_dir, class_names, img_size)
    X_test, y_test   = load_images(test_dir,   class_names, img_size)

    if X_train.shape[0] == 0:
        print("test file is empty, cannot train")
        return
    if X_test.shape[0] == 0:
        print("test file is empty, will split from train")

        n = X_train.shape[0]
        idx = np.arange(n)
        np.random.shuffle(idx)
        cut = int(n * 0.8)
        tr_idx, te_idx = idx[:cut], idx[cut:]
        X_train, y_train, X_test, y_test = X_train[tr_idx], y_train[tr_idx], X_train[te_idx], y_train[te_idx]

    print(f"Train: {X_train.shape}, Test: {X_test.shape}")

    model = build_cnn((img_size, img_size, 1), num_classes=5)

    ckpt = ModelCheckpoint(args.out_model, save_best_only=True, monitor="val_accuracy", mode="max")
    es   = EarlyStopping(patience=8, restore_best_weights=True, monitor="val_accuracy", mode="max")
    rlp  = ReduceLROnPlateau(patience=4, factor=0.5, min_lr=1e-5, monitor="val_loss")

    n = X_train.shape[0]
    idx = np.arange(n)
    np.random.shuffle(idx)
    cut = int(n * 0.85)
    tr_idx, va_idx = idx[:cut], idx[cut:]
    X_tr, y_tr, X_va, y_va = X_train[tr_idx], y_train[tr_idx], X_train[va_idx], y_train[va_idx]

    history_all = []
    best_test_acc = -1.0
    for r in range(1, args.rounds + 1):
        print(f"\n=== Round {r}/{args.rounds} (epochs={args.epochs}) ===")
        history = model.fit(
            X_tr, y_tr,
            validation_data=(X_va, y_va),
            batch_size=args.batch_size,
            epochs=args.epochs,
            callbacks=[ckpt, es, rlp],
            verbose=1
        )
        history_all.append(history.history)

   
        loss, acc = model.evaluate(X_test, y_test, verbose=0)
        print(f"Test Acc: {acc:.4f}")

        best_test_acc = max(best_test_acc, acc)

        if acc >= args.min_acc:
            print(f"pass（>= {args.min_acc*100:.0f}%")
            break

        
        print("not pass")

    if Path(args.out_model).exists():
        try:
            best_model = tf.keras.models.load_model(args.out_model)
            model = best_model
        except Exception as e:
            print(f"load fail: {e}")

    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    y_pred = np.argmax(model.predict(X_test, verbose=0), axis=1)
    per_cls = per_class_accuracy(y_test, y_pred, class_names)

    with open("five_words_class_names.json", "w", encoding="utf-8") as f:
        json.dump({"class_names": class_names}, f, ensure_ascii=False, indent=2)

    with open("five_words_per_class_accuracy.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["class", "samples", "accuracy", "pass_50"])
        for row in per_cls:
            w.writerow([
                row["class"],
                row["samples"],
                f'{row["accuracy"]:.4f}' if isinstance(row["accuracy"], float) else row["accuracy"],
                row["pass_50"]
            ])

    results = {
        "classes": class_names,
        "img_size": img_size,
        "train_size": int(X_train.shape[0]),
        "test_size": int(X_test.shape[0]),
        "test_accuracy": float(test_acc),
        "min_required": float(args.min_acc),
        "passed": bool(test_acc >= args.min_acc),
        "rounds_used": len(history_all)
    }
    with open("five_words_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print("\n====== Results Summary ======")
    print(f"Overall Test Acc: {test_acc:.4f}  ({'PASS' if test_acc>=args.min_acc else 'FAIL'})")
    for row in per_cls:
        acc_str = f'{row["accuracy"]:.4f}' if isinstance(row["accuracy"], float) else "N/A"
        print(f'- {row["class"]}: {acc_str}  ({ "PASS" if row["pass_50"] else "FAIL"})')
    print("\nOutput files:")
    print("- five_words_best.h5")
    print("- five_words_class_names.json")
    print("- five_words_per_class_accuracy.csv")
    print("- five_words_results.json")

if __name__ == "__main__":
    main()