import cv2
import numpy as np
import os
import sys
import argparse
from pathlib import Path
from typing import List, Tuple

def setup_encoding():
    try:
        sys.stdin.reconfigure(encoding='utf-8')
    except AttributeError:
        pass

def get_image_files(folder_path: Path) -> List[str]:
    valid_extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff'}
    try:
        files = [f for f in os.listdir(folder_path) 
                if Path(f).suffix.lower() in valid_extensions]
        return sorted(files)
    except Exception as e:
        print(f"cannot read {folder_path} - {e}")
        return []

def create_transformations(img: np.ndarray) -> List[Tuple[str, np.ndarray, str]]:
    transformations = [
        ("original", img.copy(), "0"),
        ("rot90", cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE), "1"),
        ("rot180", cv2.rotate(img, cv2.ROTATE_180), "2"),
        ("rot270", cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE), "3"),
    ]
    
    rows, cols = img.shape[:2]
    M = np.float32([[1, 0, 10], [0, 1, 10]])
    translated_img = cv2.warpAffine(img, M, (cols, rows))
    transformations.append(("translated", translated_img, "4"))
    
    return transformations

def process_character_dataset(char_name: str, char_folder: Path, 
                            sample_dir: Path, test_dir: Path, 
                            sample_limit: int = 40, verbose: bool = False):
    
    if verbose:
        print(f"Processing character: {char_name}")
    
    char_sample_dir = sample_dir / char_name
    char_test_dir = test_dir / char_name
    
    char_sample_dir.mkdir(parents=True, exist_ok=True)
    char_test_dir.mkdir(parents=True, exist_ok=True)

    image_files = get_image_files(char_folder)
    
    if not image_files:
        print(f"No image files found")
        return 0, 0
    
    sample_count = 0
    test_count = 0
    
    for i, img_file in enumerate(image_files[:sample_limit]):
        img_path = char_folder / img_file
        img = cv2.imread(str(img_path))
        
        if img is None:
            continue
            
        output_filename = f"{char_name}_sample_{i:03d}.png"
        output_path = char_sample_dir / output_filename
        
        if not output_path.exists():
            success = cv2.imwrite(str(output_path), img)
            if success:
                sample_count += 1
                if verbose:
                    print(f"Sample: {output_filename}")

    remaining_images = image_files[sample_limit:]
    
    for i, img_file in enumerate(remaining_images):
        img_path = char_folder / img_file
        img = cv2.imread(str(img_path))
        
        if img is None:
            continue
        

        transformations = create_transformations(img)
        
        for trans_name, transformed_img, suffix in transformations:
            output_filename = f"{char_name}_test_{i:03d}_{trans_name}_{suffix}.png"
            output_path = char_test_dir / output_filename
            
            if not output_path.exists():
                success = cv2.imwrite(str(output_path), transformed_img)
                if success:
                    test_count += 1
                    if verbose:
                        print(f"Test: {output_filename}")
    
    return sample_count, test_count

def main():
    parser = argparse.ArgumentParser(
               formatter_class=argparse.RawDescriptionHelpFormatter,
               description="Traditional Chinese Handwritten Dataset Processor",
    )
    parser.add_argument('--input', type=str, required=True,
                        help='Path to the input dataset folder')
    parser.add_argument('--sample_dir', type=str, required=True,
                        help='Path to the output sample images folder')
    parser.add_argument('--test_dir', type=str, required=True,
                        help='Path to the output test images folder')
    parser.add_argument('--sample_limit', type=int, default=40,
                        help='Number of sample images per character (default: 40)')
    
    args = parser.parse_args()
    
    # 設定編碼
    setup_encoding()
    
    # 轉換為 Path 物件
    input_path = Path(args.input).resolve()
    sample_path = Path(args.sample_dir).resolve()
    test_path = Path(args.test_dir).resolve()
    
    print(f"Traditional Chinese Handwritten Dataset Processor")
    print(f"Input Path: {input_path}")
    print(f"Sample Output: {sample_path}")
    print(f"Test Output: {test_path}")
    print(f"Sample Limit: {args.sample_limit} images/character")
    print("-" * 60)
    
    if not input_path.exists():
        print(f"path not found - {input_path}")
        sys.exit(1)
    
    sample_path.mkdir(parents=True, exist_ok=True)
    test_path.mkdir(parents=True, exist_ok=True)
    
    char_folders = [f for f in os.listdir(input_path) 
                   if (input_path / f).is_dir() and not f.startswith('.')]
    char_folders.sort()
    
    print(f"Found {len(char_folders)} character")
    print("-" * 60)
    
    total_sample_count = 0
    total_test_count = 0
    processed_chars = 0
    
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
    
    print("-" * 60)
    print(f"Processing Complete!")
    print(f"Statistics:")
    print(f"Characters Processed: {processed_chars}")
    print(f"Total Sample Images: {total_sample_count}")
    print(f"Total Test Images: {total_test_count}")
    print(f"Total Images: {total_sample_count + total_test_count}")

if __name__ == "__main__":
    main()
    
# usage example:
# python main.py --input ./dataset --sample_dir ./output/samples --test_dir ./output/tests --sample_limit 40