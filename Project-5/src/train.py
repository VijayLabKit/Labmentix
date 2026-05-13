import tensorflow as tf
from preprocess import get_datasets, build_custom_model, build_transfer_model
import os

if not os.path.exists('models'):
    os.makedirs('models')

def train_models():
    train_ds, val_ds, test_ds = get_datasets()

    early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3)

    print("\nTraining Custom CNN...")
    custom_model = build_custom_model()
    custom_model.fit(train_ds, validation_data=val_ds, epochs=15, 
                     callbacks=[early_stop, reduce_lr, 
                                tf.keras.callbacks.ModelCheckpoint('models/custom_bird_drone.keras', save_best_only=True)])

    print("\nTraining Transfer Learning Model...")
    transfer_model = build_transfer_model()
    transfer_model.fit(train_ds, validation_data=val_ds, epochs=15, 
                       callbacks=[early_stop, reduce_lr, 
                                  tf.keras.callbacks.ModelCheckpoint('models/transfer_bird_drone.keras', save_best_only=True)])

if __name__ == "__main__":
    train_models()