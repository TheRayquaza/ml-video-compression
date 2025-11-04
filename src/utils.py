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

