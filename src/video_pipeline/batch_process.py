"""Batch process all YUV files in the source directory.

This script processes all YUV files in the source/ directory through the EDSR pipeline,
calculates PSNR for each, and stores the results in a CSV file.

Usage:
  python batch_process.py --weights PATH_TO_WEIGHTS --source-dir source/ --output-csv results/results.csv
"""

import os
import sys
import csv
import argparse
import re
from datapipeline import (
    extract_all_frames,
    downscale_frames,
    run_inference_batch,
    convert_to_yuv
)
from src.edsr import build_model
from tensorflow import keras


def parse_yuv_filename(filename):
    """Parse YUV filename to extract width, height, and fps.
    
    Expected format: NAME_WIDTHxHEIGHT_FPS_420.yuv
    Example: BasketballPass_416x240_50_420.yuv
    
    Returns:
        tuple: (width, height, fps) or None if parsing fails
    """
    # Pattern: anything_WIDTHxHEIGHT_FPS_420.yuv
    pattern = r'.*?_(\d+)x(\d+)_(\d+)_420\.yuv$'
    match = re.match(pattern, filename)
    
    if match:
        width = int(match.group(1))
        height = int(match.group(2))
        fps = int(match.group(3))
        return width, height, fps
    
    return None


def process_yuv_file(yuv_path, model, scale, output_base_dir, compute_psnr=True):
    """Process a single YUV file through the EDSR pipeline.
    
    Args:
        yuv_path: Path to input YUV file
        model: EDSR model
        scale: Upscaling factor
        output_base_dir: Base directory for outputs
        compute_psnr: If True, downscale frames then upscale to compute PSNR
    
    Returns:
        dict: Results containing filename, PSNR, and other metadata
    """
    filename = os.path.basename(yuv_path)
    name_without_ext = os.path.splitext(filename)[0]
    
    # Parse filename to get dimensions
    parsed = parse_yuv_filename(filename)
    if parsed is None:
        print(f'⚠ Warning: Could not parse dimensions from {filename}')
        print(f'  Please use format: NAME_WIDTHxHEIGHT_FPS_420.yuv')
        return None
    
    width, height, fps = parsed
    
    print(f'\n{"="*80}')
    print(f'Processing: {filename}')
    print(f'Dimensions: {width}x{height} @ {fps} fps')
    print(f'{"="*80}')
    
    # Create work directory for this video
    work_dir = os.path.join(output_base_dir, name_without_ext, 'work')
    extracted_dir = os.path.join(work_dir, 'extracted_frames')
    downscaled_dir = os.path.join(work_dir, 'downscaled_frames')
    upsampled_dir = os.path.join(work_dir, 'upsampled_frames')
    
    # Output YUV path
    output_yuv = os.path.join(output_base_dir, name_without_ext, f'{name_without_ext}_upscaled.yuv')
    os.makedirs(os.path.dirname(output_yuv), exist_ok=True)
    
    try:
        # STEP 1: Extract frames
        frame_count = extract_all_frames(yuv_path, width, height, extracted_dir)
        
        # STEP 1.5: Downscale frames if PSNR computation is enabled
        if compute_psnr:
            downscale_frames(extracted_dir, downscaled_dir, scale)
            inference_input_dir = downscaled_dir
            original_frames_dir = extracted_dir  # Original frames for PSNR comparison
        else:
            inference_input_dir = extracted_dir
            original_frames_dir = None
        
        # STEP 2: Run inference and calculate PSNR
        mean_psnr = run_inference_batch(model, inference_input_dir, upsampled_dir, original_frames_dir, scale)
        
        # STEP 3: Convert back to YUV
        output_width = width * scale
        output_height = height * scale
        convert_to_yuv(upsampled_dir, output_width, output_height, output_yuv)
        
        # Calculate output file size
        output_size_mb = os.path.getsize(output_yuv) / (1024 * 1024)
        input_size_mb = os.path.getsize(yuv_path) / (1024 * 1024)
        
        result = {
            'filename': filename,
            'input_width': width,
            'input_height': height,
            'output_width': output_width,
            'output_height': output_height,
            'fps': fps,
            'frame_count': frame_count,
            'scale_factor': scale,
            'mean_psnr_db': mean_psnr if mean_psnr is not None else 'N/A',
            'input_size_mb': f'{input_size_mb:.2f}',
            'output_size_mb': f'{output_size_mb:.2f}',
            'output_path': output_yuv,
            'status': 'Success'
        }
        
        print(f'\n✓ Successfully processed {filename}')
        if mean_psnr is not None:
            print(f'  Mean PSNR: {mean_psnr:.2f} dB')
        print(f'  Output: {output_yuv}')
        
        return result
        
    except Exception as e:
        print(f'\n✗ Error processing {filename}: {str(e)}')
        return {
            'filename': filename,
            'input_width': width,
            'input_height': height,
            'output_width': width * scale,
            'output_height': height * scale,
            'fps': fps,
            'frame_count': 'N/A',
            'scale_factor': scale,
            'mean_psnr_db': 'N/A',
            'input_size_mb': 'N/A',
            'output_size_mb': 'N/A',
            'output_path': 'N/A',
            'status': f'Failed: {str(e)}'
        }


def main():
    parser = argparse.ArgumentParser(description='Batch process YUV files with EDSR')
    parser.add_argument('--weights', required=True, help='Path to EDSR model weights (.h5)')
    parser.add_argument('--source-dir', default='source/', help='Directory containing YUV files')
    parser.add_argument('--output-dir', default='output/', help='Base directory for outputs')
    parser.add_argument('--output-csv', default='results.csv', help='Output CSV file for results')
    parser.add_argument('--scale', type=int, default=4, help='Upscaling factor')
    parser.add_argument('--compute-psnr', action='store_true', default=True, 
                        help='Compute PSNR by downscaling then upscaling (default: True)')
    parser.add_argument('--no-psnr', dest='compute_psnr', action='store_false',
                        help='Disable PSNR computation')
    
    args = parser.parse_args()
    
    # Check if source directory exists
    if not os.path.exists(args.source_dir):
        print(f'Error: Source directory not found: {args.source_dir}')
        sys.exit(1)
    
    # Find all YUV files
    yuv_files = [f for f in os.listdir(args.source_dir) 
                 if f.lower().endswith('.yuv') and '_420.yuv' in f.lower()]
    
    if not yuv_files:
        print(f'No YUV files found in {args.source_dir}')
        sys.exit(1)
    
    print(f'Found {len(yuv_files)} YUV files to process')
    print(f'Files: {", ".join(yuv_files)}')
    
    # Build and load model
    print('\n=== Loading EDSR Model ===')
    model = build_model(scale=args.scale, num_res_blocks=16, num_filters=64, loss="mae", metric="both")
    
    try:
        model.load_weights(args.weights)
        print(f'✓ Loaded weights from {args.weights}')
    except Exception as e:
        print(f'Failed to load weights: {e}')
        print('Trying to load as full model...')
        try:
            model = keras.models.load_model(args.weights)
            print('✓ Loaded full model')
        except Exception as e2:
            print(f'Failed to load model: {e2}')
            sys.exit(1)
    
    # Process each YUV file
    results = []
    for i, yuv_file in enumerate(yuv_files, 1):
        print(f'\n\n{"#"*80}')
        print(f'# Processing file {i}/{len(yuv_files)}: {yuv_file}')
        print(f'{"#"*80}')
        
        yuv_path = os.path.join(args.source_dir, yuv_file)
        result = process_yuv_file(yuv_path, model, args.scale, args.output_dir, args.compute_psnr)
        
        if result is not None:
            results.append(result)
    
    # Write results to CSV
    if results:
        print(f'\n{"="*80}')
        print('Writing results to CSV...')
        
        with open(args.output_csv, 'w', newline='') as csvfile:
            fieldnames = [
                'filename', 'input_width', 'input_height', 'output_width', 'output_height',
                'fps', 'frame_count', 'scale_factor', 'mean_psnr_db',
                'input_size_mb', 'output_size_mb', 'output_path', 'status'
            ]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            
            writer.writeheader()
            for result in results:
                writer.writerow(result)
        
        print(f'✓ Results saved to {args.output_csv}')
        
        # Print summary
        print(f'\n{"="*80}')
        print('BATCH PROCESSING COMPLETE!')
        print(f'{"="*80}')
        print(f'Total files processed: {len(results)}')
        
        successful = sum(1 for r in results if r['status'] == 'Success')
        failed = len(results) - successful
        
        print(f'Successful: {successful}')
        print(f'Failed: {failed}')
        
        # Calculate average PSNR for successful runs
        psnr_values = [float(r['mean_psnr_db']) for r in results 
                      if r['mean_psnr_db'] != 'N/A' and r['status'] == 'Success']
        
        if psnr_values:
            avg_psnr = sum(psnr_values) / len(psnr_values)
            print(f'\nAverage PSNR: {avg_psnr:.2f} dB')
            print(f'Min PSNR: {min(psnr_values):.2f} dB')
            print(f'Max PSNR: {max(psnr_values):.2f} dB')
        
        print(f'\nResults saved to: {args.output_csv}')
        print(f'Output files in: {args.output_dir}')
        print(f'{"="*80}')
    else:
        print('\nNo files were successfully processed.')


if __name__ == '__main__':
    main()
