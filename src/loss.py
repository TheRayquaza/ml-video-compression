import tensorflow as tf

def PSNR(super_resolution, high_resolution):
    """Compute the peak signal-to-noise ratio, measures quality of image."""
    psnr_value = tf.image.psnr(high_resolution, super_resolution, max_val=255)[0]
    return psnr_value

def PSNR_non_training(super_resolution, high_resolution):
    """Compute the peak signal-to-noise ratio, measures quality of image."""
    psnr_value = tf.image.psnr(high_resolution, super_resolution, max_val=255)
    return tf.get_static_value(psnr_value)
