"""
SE-ResNet18 Sequence Model Architecture and Singleton Model Loader.
Preserves exact model graph structure and weights compatibility with training notebook.
Loads model into memory ONCE at application startup.
"""

import os
import threading
import logging
from pathlib import Path
from typing import Tuple, Optional
import numpy as np

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from config import CFG

logger = logging.getLogger(__name__)

# Configure GPU Memory Growth if available
for gpu in tf.config.list_physical_devices("GPU"):
    try:
        tf.config.experimental.set_memory_growth(gpu, True)
        logger.info(f"Set memory growth for GPU: {gpu}")
    except Exception as exc:
        logger.warning(f"Memory growth could not be set for {gpu}: {exc}")


def se_block(x: tf.Tensor, reduction: int = 16, name: str = "se") -> tf.Tensor:
    """Squeeze-and-Excitation block for channel attention."""
    channels = int(x.shape[-1])
    squeeze = layers.GlobalAveragePooling2D(name=f"{name}_gap")(x)
    excitation = layers.Dense(
        max(channels // reduction, 1), activation="relu", name=f"{name}_fc1"
    )(squeeze)
    excitation = layers.Dense(channels, activation="sigmoid", name=f"{name}_fc2")(
        excitation
    )
    excitation = layers.Reshape((1, 1, channels), name=f"{name}_reshape")(
        excitation
    )
    return layers.Multiply(name=f"{name}_scale")([x, excitation])


def residual_se_block(
    x: tf.Tensor,
    filters: int,
    stride: int = 1,
    reduction: int = 16,
    name: str = "res_se",
) -> tf.Tensor:
    """Residual block with integrated Squeeze-and-Excitation module."""
    shortcut = x
    y = layers.Conv2D(
        filters, 3, strides=stride, padding="same", use_bias=False, name=f"{name}_conv1"
    )(x)
    y = layers.BatchNormalization(name=f"{name}_bn1")(y)
    y = layers.ReLU(name=f"{name}_relu1")(y)
    y = layers.Conv2D(
        filters, 3, strides=1, padding="same", use_bias=False, name=f"{name}_conv2"
    )(y)
    y = layers.BatchNormalization(name=f"{name}_bn2")(y)
    y = se_block(y, reduction=reduction, name=f"{name}_se")
    
    if stride != 1 or int(shortcut.shape[-1]) != filters:
        shortcut = layers.Conv2D(
            filters, 1, strides=stride, padding="same", use_bias=False, name=f"{name}_proj_conv"
        )(shortcut)
        shortcut = layers.BatchNormalization(name=f"{name}_proj_bn")(shortcut)
    
    y = layers.Add(name=f"{name}_add")([shortcut, y])
    return layers.ReLU(name=f"{name}_out")(y)


def scaled_filters(filters: int) -> int:
    """Scale standard ResNet width by width_multiplier factor (0.5 by default)."""
    return max(16, int(round(filters * CFG.width_multiplier)))


def build_se_resnet18_backbone(input_shape: Tuple[int, int, int]) -> keras.Model:
    """Construct 2D SE-ResNet18 frame feature extractor backbone."""
    inputs = keras.Input(shape=input_shape, name="frame")
    stem_filters = scaled_filters(64)
    x = layers.Conv2D(
        stem_filters, 3, strides=1, padding="same", use_bias=False, name="stem_conv"
    )(inputs)
    x = layers.BatchNormalization(name="stem_bn")(x)
    x = layers.ReLU(name="stem_relu")(x)
    
    block_id = 0
    block_configs = [
        (64, 1), (64, 1),
        (128, 2), (128, 1),
        (256, 2), (256, 1),
        (512, 2), (512, 1),
    ]
    for filters, stride in block_configs:
        x = residual_se_block(
            x, filters=scaled_filters(filters), stride=stride, name=f"block{block_id}"
        )
        block_id += 1
        
    x = layers.GlobalAveragePooling2D(name="frame_gap")(x)
    return keras.Model(inputs, x, name="SE_ResNet18_Frame_Backbone")


def build_sequence_model() -> keras.Model:
    """
    Construct end-to-end Sequence Classifier:
    TimeDistributed(SE-ResNet18) -> BiGRU -> Temporal Pooling (Avg+Max) -> Classifier Dense.
    """
    sequence_inputs = keras.Input(
        shape=(CFG.sequence_length, CFG.image_size, CFG.image_size, CFG.channels),
        name="sequence",
    )
    backbone = build_se_resnet18_backbone(
        (CFG.image_size, CFG.image_size, CFG.channels)
    )
    x = layers.TimeDistributed(backbone, name="frame_encoder")(sequence_inputs)
    x = layers.Bidirectional(
        layers.GRU(CFG.temporal_units, return_sequences=True), name="temporal_gru"
    )(x)
    x_avg = layers.GlobalAveragePooling1D(name="temporal_avg_pool")(x)
    x_max = layers.GlobalMaxPooling1D(name="temporal_max_pool")(x)
    x = layers.Concatenate(name="temporal_concat")([x_avg, x_max])
    x = layers.Dropout(0.35, name="classifier_dropout")(x)
    x = layers.Dense(
        max(64, CFG.temporal_units * 2),
        activation="relu",
        kernel_regularizer=keras.regularizers.l2(CFG.weight_decay),
        name="classifier_fc",
    )(x)
    x = layers.Dropout(0.25, name="classifier_dropout_2")(x)
    outputs = layers.Dense(
        CFG.num_classes, activation="softmax", dtype="float32", name="class_probabilities"
    )(x)
    return keras.Model(sequence_inputs, outputs, name="SE_ResNet18_Federated_VAD")


class ModelLoader:
    """
    Thread-safe Singleton Model Loader.
    Ensures model architecture is constructed and weights are loaded ONLY ONCE.
    """
    _instance: Optional["ModelLoader"] = None
    _lock: threading.Lock = threading.Lock()

    def __init__(self) -> None:
        if ModelLoader._instance is not None:
            raise RuntimeError("ModelLoader is a singleton. Use ModelLoader.get_instance().")
        
        self.model_path: Path = CFG.model_weights_path
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Trained weights file not found at '{self.model_path}'. "
                "Ensure best_model.weights.h5 is placed in models/ directory."
            )
        
        logger.info("Initializing SE-ResNet18 Federated VAD Model...")
        self.model: keras.Model = build_sequence_model()
        logger.info(f"Loading weights from {self.model_path}...")
        self.model.load_weights(str(self.model_path))
        logger.info("Model weights loaded successfully.")

        # Warmup dummy prediction to compile graph trace
        dummy_input = np.zeros(
            (1, CFG.sequence_length, CFG.image_size, CFG.image_size, CFG.channels),
            dtype=np.float32,
        )
        _ = self.model.predict(dummy_input, verbose=0)
        logger.info("Model warmup completed. Ready for inference.")

    @classmethod
    def get_instance(cls) -> "ModelLoader":
        """Get or initialize singleton instance thread-safely."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    @tf.function(reduce_retracing=True)
    def predict_batch_compiled(self, sequences: tf.Tensor) -> tf.Tensor:
        """
        Compiled prediction step using @tf.function for fast batch inference.
        """
        return self.model(sequences, training=False)

    def predict(self, sequences: np.ndarray, batch_size: int = CFG.batch_size) -> np.ndarray:
        """
        Run inference on sequence array using loaded singleton model.
        """
        return self.model.predict(sequences, batch_size=batch_size, verbose=0)
