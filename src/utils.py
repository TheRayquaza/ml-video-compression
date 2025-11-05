import matplotlib.pyplot as plt

def plot_results(lowres, preds):
    """
    Displays low resolution image and super resolution image
    """
    plt.figure(figsize=(10, 10))
    plt.subplot(132), plt.imshow(lowres), plt.title("Low resolution")
    plt.subplot(133), plt.imshow(preds), plt.title("Prediction")
    plt.show()

def plot_history(history):
    plt.plot(history.history['psnr'])
    plt.plot(history.history['val_psnr'])
    plt.title('model psnr')
    plt.ylabel('psnr')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.show()

def plot_training_history(history):
    """
    Plot PSNR and SSIM metrics for training and validation.
    
    Args:
        history: Keras History object from model.fit()
    """
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    if 'psnr' in history.history:
        axes[0].plot(history.history['psnr'], label='Train PSNR', linewidth=2)
        axes[0].plot(history.history['val_psnr'], label='Val PSNR', linewidth=2)
        axes[0].set_xlabel('Epoch', fontsize=12)
        axes[0].set_ylabel('PSNR (dB)', fontsize=12)
        axes[0].set_title('PSNR over Epochs', fontsize=14, fontweight='bold')
        axes[0].legend(fontsize=11)
        axes[0].grid(True, alpha=0.3)
    
    if 'ssim' in history.history:
        axes[1].plot(history.history['ssim'], label='Train SSIM', linewidth=2)
        axes[1].plot(history.history['val_ssim'], label='Val SSIM', linewidth=2)
        axes[1].set_xlabel('Epoch', fontsize=12)
        axes[1].set_ylabel('SSIM', fontsize=12)
        axes[1].set_title('SSIM over Epochs', fontsize=14, fontweight='bold')
        axes[1].legend(fontsize=11)
        axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('training_history.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("\n" + "="*50)
    print("Final Training Metrics:")
    print("="*50)
    if 'psnr' in history.history:
        print(f"Train PSNR: {history.history['psnr'][-1]:.4f} dB")
        print(f"Val PSNR:   {history.history['val_psnr'][-1]:.4f} dB")
    if 'ssim' in history.history:
        print(f"Train SSIM: {history.history['ssim'][-1]:.4f}")
        print(f"Val SSIM:   {history.history['val_ssim'][-1]:.4f}")
    print("="*50)
