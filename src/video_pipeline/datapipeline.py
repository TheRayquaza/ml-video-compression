"""Improved YUV inference pipeline that processes YUV files through EDSR model.

Usage:
  python yuv_pipeline.py --weights PATH_TO_WEIGHTS --yuv input.yuv --width 1920 --height 1080

This pipeline:
1. Extracts all frames from YUV file to PNG format
2. Runs EDSR inference on all extracted frames
3. Converts upsampled PNG frames back to YUV420 format
"""

import os
import argparse
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


def PSNR_non_training(super_resolution, high_resolution):
    """Compute the peak signal-to-noise ratio, measures quality of image."""
    # Max value of pixel is 255
    psnr_value = tf.image.psnr(high_resolution, super_resolution, max_val=255)
    return tf.get_static_value(psnr_value)


def yuv420_to_png(yuv_path, width, height, frame_index, out_path):
    """Extract one frame (YUV420 planar) and save as PNG (RGB) upsampled by NN."""
    frame_size = width * height * 3 // 2
    with open(yuv_path, 'rb') as f:
        f.seek(frame_index * frame_size)
        data = f.read(frame_size)
    if len(data) < frame_size:
        raise ValueError('frame_index out of range')

    y = np.frombuffer(data[0:width*height], dtype=np.uint8).reshape((height, width))
    u = np.frombuffer(data[width*height:width*height + (width//2)*(height//2)], dtype=np.uint8).reshape((height//2, width//2))
    v = np.frombuffer(data[width*height + (width//2)*(height//2):], dtype=np.uint8).reshape((height//2, width//2))

    u_up = u.repeat(2, axis=0).repeat(2, axis=1)
    v_up = v.repeat(2, axis=0).repeat(2, axis=1)

    y = y.astype(np.float32)
    u_up = u_up.astype(np.float32) - 128.0
    v_up = v_up.astype(np.float32) - 128.0

    r = y + 1.402 * v_up
    g = y - 0.344136 * u_up - 0.714136 * v_up
    b = y + 1.772 * u_up

    rgb = np.stack([r, g, b], axis=-1)
    rgb = np.clip(rgb, 0, 255).astype(np.uint8)

    img = Image.fromarray(rgb, mode='RGB')
    img.save(out_path)


def png_to_yuv420(png_dir, width, height, out_yuv_path):
    """Convert a directory of PNG RGB frames to YUV420 file."""
    pngs = sorted([p for p in os.listdir(png_dir) if p.lower().endswith('.png')])
    if not pngs:
        raise ValueError('no PNG files found in dir')

    with open(out_yuv_path, 'wb') as out_f:
        for name in pngs:
            path = os.path.join(png_dir, name)
            img = Image.open(path).convert('RGB').resize((width, height))
            arr = np.array(img).astype(np.float32)
            r = arr[..., 0]
            g = arr[..., 1]
            b = arr[..., 2]
            y = 0.299 * r + 0.587 * g + 0.114 * b
            u = -0.168736 * r - 0.331264 * g + 0.5 * b + 128
            v = 0.5 * r - 0.418688 * g - 0.081312 * b + 128

            y = np.clip(y, 0, 255).astype(np.uint8)
            u = np.clip(u, 0, 255).astype(np.uint8)
            v = np.clip(v, 0, 255).astype(np.uint8)

            u_ds = u.reshape((height//2, 2, width//2, 2)).mean(axis=(1,3)).astype(np.uint8)
            v_ds = v.reshape((height//2, 2, width//2, 2)).mean(axis=(1,3)).astype(np.uint8)

            out_f.write(y.tobytes())
            out_f.write(u_ds.tobytes())
            out_f.write(v_ds.tobytes())


def edsr_res_block(x_in, num_filters):
    """Residual block matching the notebook architecture."""
    x = layers.Conv2D(num_filters, 3, padding='same', activation='relu')(x_in)
    x = layers.Conv2D(num_filters, 3, padding='same')(x)
    x = layers.Add()([x_in, x])
    return x


def upsample_block(x, scale=2, num_filters=64):
    """Upsampling block supporting power-of-2 scales (2, 4, 8, 16).
    
    Args:
        x: Input tensor
        scale: Upscaling factor (must be 2, 4, 8, or 16)
        num_filters: Number of filters (default 64)
    
    Returns:
        Upsampled tensor
    """
    # Check if scale is a power of 2
    if scale and (scale & (scale - 1)) == 0 and scale > 1:  # Power of 2 check
        steps = int(tf.math.log(tf.cast(scale, tf.float32)) / tf.math.log(2.0))
        for _ in range(steps):
            x = layers.Conv2D(num_filters * (2 ** 2), 3, padding='same')(x)
            x = layers.Lambda(lambda t: tf.nn.depth_to_space(t, block_size=2))(x)
    else:
        raise ValueError(f"Scale {scale} is invalid. Must be 2, 4, 8, or 16.")
    
    return x


def build_edsr(scale=4, num_res_blocks=16, num_filters=64, input_shape=(None, None, 3)):
    """Build EDSR model with configurable scale factor.
    
    Args:
        scale: Upscaling factor (2, 4, 8, or 16)
        num_res_blocks: Number of residual blocks (typically 16)
        num_filters: Number of filters in Conv2D layers (typically 64)
        input_shape: Input shape (None, None, 3) for flexible input size
    
    Returns:
        EDSR Keras Model
    """
    inp = layers.Input(shape=input_shape)
    
    # Rescaling input from [0-255] to [0-1]
    x = layers.Rescaling(scale=1.0 / 255)(inp)
    
    # Initial convolution
    x = x_new = layers.Conv2D(num_filters, 3, padding='same')(x)
    
    # Residual blocks
    for _ in range(num_res_blocks):
        x_new = edsr_res_block(x_new, num_filters)
    
    x_new = layers.Conv2D(num_filters, 3, padding='same')(x_new)
    x = layers.Add()([x, x_new])
    
    # Upsampling with configurable scale
    x = upsample_block(x, scale=scale, num_filters=num_filters)
    
    # Final convolution
    x = layers.Conv2D(3, 3, padding='same')(x)
    
    # Rescaling output from [0-1] to [0-255]
    out = layers.Rescaling(scale=255)(x)
    
    model = keras.Model(inputs=inp, outputs=out, name='edsr')
    return model


def get_frame_count(yuv_path, width, height):
    """Calculate the number of frames in a YUV420 file."""
    frame_size = width * height * 3 // 2
    file_size = os.path.getsize(yuv_path)
    return file_size // frame_size


def extract_all_frames(yuv_path, width, height, output_dir):
    """Step 1: Extract all frames from YUV file to PNG format."""
    os.makedirs(output_dir, exist_ok=True)
    
    frame_count = get_frame_count(yuv_path, width, height)
    print(f'\n=== STEP 1: Extracting {frame_count} frames from YUV ===')
    
    for frame_idx in range(frame_count):
        out_path = os.path.join(output_dir, f'frame_{frame_idx:04d}.png')
        print(f'Extracting frame {frame_idx}/{frame_count-1} -> {out_path}')
        yuv420_to_png(yuv_path, width, height, frame_idx, out_path)
    
    print(f'✓ Extracted {frame_count} frames to {output_dir}')
    return frame_count


def downscale_frames(input_dir, output_dir, scale=4):
    """Step 1.5: Downscale frames by a given factor to simulate low-resolution input."""
    os.makedirs(output_dir, exist_ok=True)
    
    png_files = sorted([f for f in os.listdir(input_dir) if f.lower().endswith('.png')])
    print(f'\n=== STEP 1.5: Downscaling {len(png_files)} frames by {scale}x ===')
    
    for idx, png_file in enumerate(png_files):
        input_path = os.path.join(input_dir, png_file)
        output_path = os.path.join(output_dir, png_file)
        
        print(f'Downscaling {idx+1}/{len(png_files)}: {png_file}')
        
        # Load and downscale image
        img = Image.open(input_path).convert('RGB')
        original_width, original_height = img.size
        downscaled_width = original_width // scale
        downscaled_height = original_height // scale
        
        # Use BICUBIC for downscaling (good quality)
        downscaled_img = img.resize((downscaled_width, downscaled_height), Image.BICUBIC)
        downscaled_img.save(output_path)
        
        print(f'  {original_width}x{original_height} -> {downscaled_width}x{downscaled_height}')
    
    print(f'✓ Downscaled {len(png_files)} frames to {output_dir}')


def run_inference_batch(model, input_dir, output_dir, original_dir=None, scale=4):
    """Step 2: Run EDSR inference on all PNG frames and calculate PSNR if original frames provided."""
    os.makedirs(output_dir, exist_ok=True)
    
    png_files = sorted([f for f in os.listdir(input_dir) if f.lower().endswith('.png')])
    print(f'\n=== STEP 2: Running inference on {len(png_files)} frames ===')
    
    psnr_values = []
    
    for idx, png_file in enumerate(png_files):
        input_path = os.path.join(input_dir, png_file)
        output_path = os.path.join(output_dir, png_file)
        
        print(f'Processing {idx+1}/{len(png_files)}: {png_file}')
        
        # Load image (keep in [0-255] range as model expects this)
        img = Image.open(input_path).convert('RGB')
        arr = np.array(img).astype(np.float32)
        inp = np.expand_dims(arr, 0)
        
        # Run inference (model outputs [0-255] range)
        sr = model.predict(inp, verbose=0)
        sr = np.clip(sr[0], 0, 255).astype(np.uint8)
        
        # Calculate PSNR if original high-res frames are provided
        if original_dir is not None:
            original_path = os.path.join(original_dir, png_file)
            if os.path.exists(original_path):
                # Load original image (should already be same size as SR output after downscale->upscale)
                original_img = Image.open(original_path).convert('RGB')
                original_arr = np.array(original_img).astype(np.float32)
                
                # Verify dimensions match
                if sr.shape == original_arr.shape:
                    # Convert SR to float32 for consistent comparison
                    sr_float = sr.astype(np.float32)
                    
                    # Calculate PSNR (both images now in float32, range [0-255])
                    psnr = PSNR_non_training(sr_float, original_arr)
                    if psnr is not None:
                        psnr_values.append(float(psnr))
                        print(f'  PSNR: {psnr:.2f} dB')
                else:
                    print(f'  Warning: Shape mismatch - SR: {sr.shape}, Original: {original_arr.shape}')
        
        # Save result
        out_img = Image.fromarray(sr)
        out_img.save(output_path)
    
    print(f'✓ Processed {len(png_files)} frames to {output_dir}')
    
    # Calculate and display mean PSNR
    if psnr_values:
        mean_psnr = np.mean(psnr_values)
        print(f'\n=== PSNR Statistics ===')
        print(f'Mean PSNR: {mean_psnr:.2f} dB')
        print(f'Min PSNR: {np.min(psnr_values):.2f} dB')
        print(f'Max PSNR: {np.max(psnr_values):.2f} dB')
        return mean_psnr
    
    return None


def convert_to_yuv(png_dir, width, height, output_yuv):
    """Step 3: Convert upsampled PNG frames back to YUV420."""
    print(f'\n=== STEP 3: Converting PNG frames to YUV420 ===')
    print(f'Output resolution: {width}x{height}')
    
    png_to_yuv420(png_dir, width, height, output_yuv)
    
    file_size_mb = os.path.getsize(output_yuv) / (1024 * 1024)
    print(f'✓ Created YUV file: {output_yuv} ({file_size_mb:.2f} MB)')


def main():
    parser = argparse.ArgumentParser(description='YUV Inference Pipeline with EDSR')
    parser.add_argument('--weights', required=True, help='Path to EDSR model weights (.h5)')
    parser.add_argument('--yuv', required=True, help='Input YUV420 file')
    parser.add_argument('--width', type=int, default=1920, help='Input width')
    parser.add_argument('--height', type=int, default=1080, help='Input height')
    parser.add_argument('--scale', type=int, default=4, help='Upscaling factor')
    parser.add_argument('--work-dir', default='pipeline_work', help='Working directory for intermediate files')
    parser.add_argument('--output-yuv', default='output_upsampled.yuv', help='Output YUV file name')
    parser.add_argument('--compute-psnr', action='store_true', help='Compute PSNR by downscaling then upscaling')
    
    args = parser.parse_args()
    
    # Create working directories
    extracted_dir = os.path.join(args.work_dir, 'extracted_frames')
    downscaled_dir = os.path.join(args.work_dir, 'downscaled_frames')
    upsampled_dir = os.path.join(args.work_dir, 'upsampled_frames')
    
    print('='*60)
    print('YUV SUPER-RESOLUTION PIPELINE')
    print('='*60)
    print(f'Input YUV: {args.yuv}')
    print(f'Input resolution: {args.width}x{args.height}')
    print(f'Scale factor: {args.scale}x')
    print(f'Output resolution: {args.width*args.scale}x{args.height*args.scale}')
    print(f'Working directory: {args.work_dir}')
    if args.compute_psnr:
        print(f'PSNR Computation: Enabled (downscale -> upscale -> compare)')
    print('='*60)
    
    # Build and load model
    print('\n=== Loading EDSR Model ===')
    model = build_edsr(scale=args.scale, num_res_blocks=16, num_filters=64, input_shape=(None, None, 3))
    
    try:
        model.load_weights(args.weights)
        print(f'✓ Loaded weights from {args.weights}')
    except Exception as e:
        print(f'Failed to load weights: {e}')
        print('Trying to load as full model...')
        model = keras.models.load_model(args.weights)
        print('✓ Loaded full model')
    
    # STEP 1: Extract frames from YUV to PNG
    frame_count = extract_all_frames(args.yuv, args.width, args.height, extracted_dir)
    
    # STEP 1.5: Downscale frames if PSNR computation is enabled
    if args.compute_psnr:
        downscale_frames(extracted_dir, downscaled_dir, args.scale)
        inference_input_dir = downscaled_dir
        original_frames_dir = extracted_dir  # Original frames for PSNR comparison
    else:
        inference_input_dir = extracted_dir
        original_frames_dir = None
    
    # STEP 2: Run inference on frames
    mean_psnr = run_inference_batch(model, inference_input_dir, upsampled_dir, original_frames_dir, args.scale)
    
    # STEP 3: Convert upsampled PNGs back to YUV
    # If PSNR mode, output should match original resolution (upscaled)
    # Otherwise, output is upscaled version
    output_width = args.width
    output_height = args.height
    convert_to_yuv(upsampled_dir, output_width, output_height, args.output_yuv)
    
    print('\n' + '='*60)
    print('PIPELINE COMPLETE!')
    print('='*60)
    print(f'✓ Processed {frame_count} frames')
    print(f'✓ Output YUV: {args.output_yuv}')
    print(f'✓ Resolution: {output_width}x{output_height}')
    if mean_psnr is not None:
        print(f'✓ Mean PSNR: {mean_psnr:.2f} dB')
    print(f'\nYou can now view the upsampled YUV file with YUView:')
    print(f'  YUView {args.output_yuv}')
    print('='*60)
    
    return mean_psnr


if __name__ == '__main__':
    main()