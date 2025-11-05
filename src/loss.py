import tensorflow as tf

def PSNR(y_true, y_pred):
    # Ensure inputs are float32 and in correct range
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    
    # Calculate PSNR for each image in batch and return mean
    psnr_values = tf.image.psnr(y_true, y_pred, max_val=255.0)
    return tf.reduce_mean(psnr_values)

def SSIM(y_true, y_pred):
    # Ensure inputs are float32 and in correct range
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    
    # Calculate SSIM for each image in batch and return mean
    ssim_values = tf.image.ssim(y_true, y_pred, max_val=255.0)
    return tf.reduce_mean(ssim_values)