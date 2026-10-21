"""
M.Tech Dissertation Architecture Implementation (TensorFlow / Keras 2.x)
Automated Pneumonia Detection from Chest Radiographs Using Deep
Convolutional Neural Networks with Batch-Normalized Spatial Regularization

Section 8 Layer Hierarchy Specification:
- InputLayer: (None, 150, 150, 1)
- Conv Block 1: Conv2D(32, (3,3)) + BatchNormalization + MaxPool(2,2) + Dropout(0.20) -> (None, 75, 75, 32)
- Conv Block 2: Conv2D(64, (3,3)) + BatchNormalization + MaxPool(2,2) + Dropout(0.20) -> (None, 37, 37, 64)
- Conv Block 3: Conv2D(128, (3,3)) + BatchNormalization + MaxPool(2,2) + Dropout(0.25) -> (None, 18, 18, 128)
- Conv Block 4: Conv2D(256, (3,3)) + BatchNormalization + MaxPool(2,2) + Dropout(0.30) -> (None, 9, 9, 256)
- Flatten: 1D Vector Reshaping -> (None, 20736)
- Dense Hidden: Dense(128) + BatchNormalization + Dropout(0.40) -> (None, 128)
- Output: Dense(1, activation='sigmoid') -> (None, 1)
- Optimizer: RMSprop(learning_rate=1e-3, rho=0.9, epsilon=1e-7)
- Loss: BinaryCrossentropy()
"""

def build_keras_pneumonia_cnn(input_shape=(150, 150, 1)):
    """
    Constructs the exact Keras Sequential/Functional model from Section 8 of the proposal.
    Requires: tensorflow >= 2.0
    """
    try:
        import tensorflow as tf
        from tensorflow.keras import layers, models, optimizers
    except ImportError:
        raise ImportError("TensorFlow is not installed in the current environment. Use model_torch.py for native PyTorch execution.")

    model = models.Sequential(name="Pneumonia_Detection_CNN_Regularized")
    
    # Input Layer
    model.add(layers.Input(shape=input_shape, name="input_grayscale"))
    
    # Conv Block 1
    model.add(layers.Conv2D(32, (3, 3), padding="same", activation="relu", name="conv1"))
    model.add(layers.BatchNormalization(name="bn1"))
    model.add(layers.MaxPooling2D(pool_size=(2, 2), name="pool1"))
    model.add(layers.Dropout(0.20, name="dropout1"))
    
    # Conv Block 2
    model.add(layers.Conv2D(64, (3, 3), padding="same", activation="relu", name="conv2"))
    model.add(layers.BatchNormalization(name="bn2"))
    model.add(layers.MaxPooling2D(pool_size=(2, 2), name="pool2"))
    model.add(layers.Dropout(0.20, name="dropout2"))
    
    # Conv Block 3
    model.add(layers.Conv2D(128, (3, 3), padding="same", activation="relu", name="conv3"))
    model.add(layers.BatchNormalization(name="bn3"))
    model.add(layers.MaxPooling2D(pool_size=(2, 2), name="pool3"))
    model.add(layers.Dropout(0.25, name="dropout3"))
    
    # Conv Block 4
    model.add(layers.Conv2D(256, (3, 3), padding="same", activation="relu", name="conv4"))
    model.add(layers.BatchNormalization(name="bn4"))
    model.add(layers.MaxPooling2D(pool_size=(2, 2), name="pool4"))
    model.add(layers.Dropout(0.30, name="dropout4"))
    
    # Flatten Layer: 9 x 9 x 256 = 20,736
    model.add(layers.Flatten(name="flatten_20736"))
    
    # Dense Hidden
    model.add(layers.Dense(128, activation="relu", name="dense_128"))
    model.add(layers.BatchNormalization(name="bn_dense"))
    model.add(layers.Dropout(0.40, name="dropout_dense"))
    
    # Output Layer
    model.add(layers.Dense(1, activation="sigmoid", name="output_sigmoid"))
    
    # Compile with Section 9 Hyperparameters
    optimizer = optimizers.RMSprop(
        learning_rate=1e-3,
        rho=0.9,
        epsilon=1e-7
    )
    
    model.compile(
        optimizer=optimizer,
        loss="binary_crossentropy",
        metrics=[
            "accuracy",
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
            tf.keras.metrics.AUC(name="auc")
        ]
    )
    
    return model
