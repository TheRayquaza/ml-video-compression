import tensorflow_datasets as tfds

def load_data():
    # Train data : load (or Download if first usage) DIV2K from TF Datasets
    train = tfds.load("div2k/bicubic_x4", split="train", as_supervised=True)
    train_cache = train.cache()

    # Validation data : load (or Download if first usage) DIV2K from TF Datasets
    val = tfds.load("div2k/bicubic_x4", split="validation", as_supervised=True)
    val_cache = val.cache()
    
    return train, val, train_cache, val_cache
