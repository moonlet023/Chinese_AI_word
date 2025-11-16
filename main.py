#!/usr/bin/env python3
"""
繁體中文手寫資料集處理器
將每個字符的前40張圖片移到 sample 目錄，剩餘圖片轉換後移到 test 目錄
"""

import cv2
import numpy as np
import os
import sys
import argparse
from pathlib import Path
from typing import List, Tuple

def setup_encoding():
    """設定輸入編碼"""
    try:
        sys.stdin.reconfigure(encoding='utf-8')
    except AttributeError:
        pass

def get_image_files(folder_path: Path) -> List[str]:
    """獲取資料夾中的所有圖片檔案"""
    valid_extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff'}
    try:
        files = [f for f in os.listdir(folder_path) 
                if Path(f).suffix.lower() in valid_extensions]
        return sorted(files)  # 按檔名排序
    except Exception as e:
        print(f"⚠️  警告：無法讀取資料夾 {folder_path} - {e}")
        return []

def create_transformations(img: np.ndarray) -> List[Tuple[str, np.ndarray, str]]:
    """創建圖片的各種變換版本"""
    transformations = [
        ("original", img.copy(), "0"),
        ("rot90", cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE), "1"),
        ("rot180", cv2.rotate(img, cv2.ROTATE_180), "2"),
        ("rot270", cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE), "3"),
    ]
    
    # 平移變換
    rows, cols = img.shape[:2]
    M = np.float32([[1, 0, 10], [0, 1, 10]])
    translated_img = cv2.warpAffine(img, M, (cols, rows))
    transformations.append(("translated", translated_img, "4"))
    
    return transformations

def process_character_dataset(char_name: str, char_folder: Path, 
                            sample_dir: Path, test_dir: Path, 
                            sample_limit: int = 40, verbose: bool = False):
    """處理單個字符的資料集"""
    
    if verbose:
        print(f"🔤 處理字符: {char_name}")
    
    # 創建字符專屬的輸出目錄
    char_sample_dir = sample_dir / char_name
    char_test_dir = test_dir / char_name
    
    char_sample_dir.mkdir(parents=True, exist_ok=True)
    char_test_dir.mkdir(parents=True, exist_ok=True)
    
    # 獲取所有圖片檔案
    image_files = get_image_files(char_folder)
    
    if not image_files:
        print(f"⚠️  警告：{char_name} 資料夾中沒有圖片檔案")
        return 0, 0
    
    sample_count = 0
    test_count = 0
    
    # 處理前 sample_limit 張圖片作為 sample
    for i, img_file in enumerate(image_files[:sample_limit]):
        img_path = char_folder / img_file
        img = cv2.imread(str(img_path))
        
        if img is None:
            continue
            
        # 直接複製到 sample 目錄
        output_filename = f"{char_name}_sample_{i:03d}.png"
        output_path = char_sample_dir / output_filename
        
        if not output_path.exists():
            success = cv2.imwrite(str(output_path), img)
            if success:
                sample_count += 1
                if verbose:
                    print(f"  📋 Sample: {output_filename}")
    
    # 處理剩餘圖片作為 test (應用各種變換)
    remaining_images = image_files[sample_limit:]
    
    for i, img_file in enumerate(remaining_images):
        img_path = char_folder / img_file
        img = cv2.imread(str(img_path))
        
        if img is None:
            continue
        
        # 創建各種變換版本
        transformations = create_transformations(img)
        
        for trans_name, transformed_img, suffix in transformations:
            output_filename = f"{char_name}_test_{i:03d}_{trans_name}_{suffix}.png"
            output_path = char_test_dir / output_filename
            
            if not output_path.exists():
                success = cv2.imwrite(str(output_path), transformed_img)
                if success:
                    test_count += 1
                    if verbose:
                        print(f"  🧪 Test: {output_filename}")
    
    return sample_count, test_count

def main():
    """主函數"""
    parser = argparse.ArgumentParser(
        description='繁體中文手寫資料集處理器 - 分離 sample 和 test 資料',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用範例:
  python main.py                                    # 使用預設設定
  python main.py -i ./data/Traditional_Chinese_Data # 指定輸入路徑
  python main.py -s 50 -v                          # 設定 sample 數量為50並顯示詳細資訊
  python main.py --sample-dir ./my_samples          # 自訂 sample 輸出目錄
        """
    )
    
    parser.add_argument('-i', '--input', type=str, 
                       default='./data/Traditional_Chinese_Data',
                       help='輸入資料夾路徑 (預設: ./data/Traditional_Chinese_Data)')
    
    parser.add_argument('--sample-dir', type=str,
                       default='./data/sample',
                       help='Sample 輸出資料夾路徑 (預設: ./data/sample)')
    
    parser.add_argument('--test-dir', type=str,
                       default='./data/test',
                       help='Test 輸出資料夾路徑 (預設: ./data/test)')
    
    parser.add_argument('-s', '--sample-limit', type=int, default=40,
                       help='每個字符的 sample 圖片數量 (預設: 40)')
    
    parser.add_argument('-v', '--verbose', action='store_true',
                       help='顯示詳細處理資訊')
    
    args = parser.parse_args()
    
    # 設定編碼
    setup_encoding()
    
    # 轉換為 Path 物件
    input_path = Path(args.input).resolve()
    sample_path = Path(args.sample_dir).resolve()
    test_path = Path(args.test_dir).resolve()
    
    print(f"🚀 繁體中文手寫資料集處理器")
    print(f"📂 輸入路徑: {input_path}")
    print(f"📋 Sample 輸出: {sample_path}")
    print(f"🧪 Test 輸出: {test_path}")
    print(f"📊 Sample 限制: {args.sample_limit} 張/字符")
    print("-" * 60)
    
    # 檢查輸入路徑
    if not input_path.exists():
        print(f"❌ 錯誤：輸入路徑不存在 - {input_path}")
        sys.exit(1)
    
    # 創建輸出目錄
    sample_path.mkdir(parents=True, exist_ok=True)
    test_path.mkdir(parents=True, exist_ok=True)
    
    # 獲取所有字符目錄
    char_folders = [f for f in os.listdir(input_path) 
                   if (input_path / f).is_dir() and not f.startswith('.')]
    char_folders.sort()
    
    print(f"🔍 找到 {len(char_folders)} 個字符目錄")
    print("-" * 60)
    
    total_sample_count = 0
    total_test_count = 0
    processed_chars = 0
    
    # 處理每個字符
    for char_name in char_folders:
        char_folder = input_path / char_name
        
        sample_count, test_count = process_character_dataset(
            char_name, char_folder, sample_path, test_path,
            args.sample_limit, args.verbose
        )
        
        if sample_count > 0 or test_count > 0:
            processed_chars += 1
            total_sample_count += sample_count
            total_test_count += test_count
            
            if not args.verbose:
                print(f"✅ {char_name}: {sample_count} samples, {test_count} test images")
    
    # 顯示最終統計
    print("-" * 60)
    print(f"🎉 處理完成！")
    print(f"📊 統計結果:")
    print(f"   處理字符數: {processed_chars}")
    print(f"   總 Sample 圖片: {total_sample_count}")
    print(f"   總 Test 圖片: {total_test_count}")
    print(f"   總圖片數: {total_sample_count + total_test_count}")

if __name__ == "__main__":
    main()