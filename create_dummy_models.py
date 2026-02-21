import tensorflow as tf
from pathlib import Path

MODELS = ["VGG16", "ResNet50", "MobileNetV3Small"]

for name in MODELS:
    path = Path("/models") / name
    path.mkdir(parents=True, exist_ok=True)

    model = tf.keras.Sequential([
        tf.keras.layers.Input((224, 224, 3)),
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(43, activation="softmax")
    ])

    tf.saved_model.save(model, path)

print("Dummy models created.")