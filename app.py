{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 3,
   "id": "831738f7-d99c-4308-aa9b-008817fd93b1",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Import Libraries\n",
    "import numpy as np\n",
    "import tensorflow as tf\n",
    "from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array\n",
    "from tensorflow.keras.applications import MobileNet\n",
    "from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout\n",
    "from tensorflow.keras.models import Sequential\n",
    "from tensorflow.keras.optimizers import Adam\n",
    "import matplotlib.pyplot as plt"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "id": "2a42cb9a-512e-457b-acb1-cffd28cf3628",
   "metadata": {},
   "outputs": [],
   "source": [
    "\n",
    "\n",
    "# Data Augmentation and Preprocessing\n",
    "train_datagen = ImageDataGenerator(\n",
    "    rescale=1./255,  # Normalize pixel values\n",
    "    rotation_range=40,\n",
    "    width_shift_range=0.2,\n",
    "    height_shift_range=0.2,\n",
    "    shear_range=0.2,\n",
    "    zoom_range=0.2,\n",
    "    horizontal_flip=True,\n",
    "    fill_mode='nearest'\n",
    ")\n",
    "\n",
    "test_datagen = ImageDataGenerator(rescale=1./255)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "id": "97e2b326-1cfd-43b7-93ef-9fb5592ff59d",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Found 11170 images belonging to 2 classes.\n"
     ]
    }
   ],
   "source": [
    "train_set = train_datagen.flow_from_directory(\n",
    "    \"C:/Users/HP/Downloads/archive/dataset/train\",  # Replace with your training data path\n",
    "    target_size=(224, 224),  # Resize images to match MobileNet input\n",
    "    batch_size=32,\n",
    "    class_mode='binary'  # Binary classification\n",
    ")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "id": "395763a8-1473-47c8-b12c-a94a6469d175",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Found 2792 images belonging to 2 classes.\n"
     ]
    }
   ],
   "source": [
    "test_set = test_datagen.flow_from_directory(\n",
    "    \"C:/Users/HP/Downloads/archive/dataset/test\",  # Replace with your test data path\n",
    "    target_size=(224, 224),\n",
    "    batch_size=32,\n",
    "    class_mode='binary'\n",
    ")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 11,
   "id": "eacad707-f7b1-4128-b879-c9f7ecf8b021",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Class Indices: {'millitary': 0, 'other': 1}\n"
     ]
    }
   ],
   "source": [
    "# Display Class Indices\n",
    "print(\"Class Indices:\", train_set.class_indices)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 13,
   "id": "fc5539ed-582a-40c6-be1b-4911f7761862",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Downloading data from https://storage.googleapis.com/tensorflow/keras-applications/mobilenet/mobilenet_1_0_224_tf_no_top.h5\n",
      "\u001b[1m17225924/17225924\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m11s\u001b[0m 1us/step\n"
     ]
    }
   ],
   "source": [
    "base_model = MobileNet(weights='imagenet', include_top=False, input_shape=(224, 224, 3))\n",
    "base_model.trainable = False  # Freeze the base model"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 15,
   "id": "1579f9fe-cace-428a-b8d8-cd933a6d9d38",
   "metadata": {},
   "outputs": [],
   "source": [
    "model = Sequential([\n",
    "    base_model,\n",
    "    GlobalAveragePooling2D(),  # Add a global average pooling layer\n",
    "    Dense(512, activation='relu'),  # Fully connected layer\n",
    "    Dropout(0.5),  # Dropout to prevent overfitting\n",
    "    Dense(1, activation='sigmoid')  # Output layer for binary classification\n",
    "])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 17,
   "id": "09738a39-ff52-4951-b90d-cfaaf6084636",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Compile the Model\n",
    "model.compile(optimizer=Adam(learning_rate=0.0001), loss='binary_crossentropy', metrics=['accuracy'])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 19,
   "id": "a914f6b8-5691-4106-a29a-36bbb543a732",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<pre style=\"white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace\"><span style=\"font-weight: bold\">Model: \"sequential\"</span>\n",
       "</pre>\n"
      ],
      "text/plain": [
       "\u001b[1mModel: \"sequential\"\u001b[0m\n"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "text/html": [
       "<pre style=\"white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace\">┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━┓\n",
       "┃<span style=\"font-weight: bold\"> Layer (type)                         </span>┃<span style=\"font-weight: bold\"> Output Shape                </span>┃<span style=\"font-weight: bold\">         Param # </span>┃\n",
       "┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━┩\n",
       "│ mobilenet_1.00_224 (<span style=\"color: #0087ff; text-decoration-color: #0087ff\">Functional</span>)      │ (<span style=\"color: #00d7ff; text-decoration-color: #00d7ff\">None</span>, <span style=\"color: #00af00; text-decoration-color: #00af00\">7</span>, <span style=\"color: #00af00; text-decoration-color: #00af00\">7</span>, <span style=\"color: #00af00; text-decoration-color: #00af00\">1024</span>)          │       <span style=\"color: #00af00; text-decoration-color: #00af00\">3,228,864</span> │\n",
       "├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤\n",
       "│ global_average_pooling2d             │ (<span style=\"color: #00d7ff; text-decoration-color: #00d7ff\">None</span>, <span style=\"color: #00af00; text-decoration-color: #00af00\">1024</span>)                │               <span style=\"color: #00af00; text-decoration-color: #00af00\">0</span> │\n",
       "│ (<span style=\"color: #0087ff; text-decoration-color: #0087ff\">GlobalAveragePooling2D</span>)             │                             │                 │\n",
       "├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤\n",
       "│ dense (<span style=\"color: #0087ff; text-decoration-color: #0087ff\">Dense</span>)                        │ (<span style=\"color: #00d7ff; text-decoration-color: #00d7ff\">None</span>, <span style=\"color: #00af00; text-decoration-color: #00af00\">512</span>)                 │         <span style=\"color: #00af00; text-decoration-color: #00af00\">524,800</span> │\n",
       "├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤\n",
       "│ dropout (<span style=\"color: #0087ff; text-decoration-color: #0087ff\">Dropout</span>)                    │ (<span style=\"color: #00d7ff; text-decoration-color: #00d7ff\">None</span>, <span style=\"color: #00af00; text-decoration-color: #00af00\">512</span>)                 │               <span style=\"color: #00af00; text-decoration-color: #00af00\">0</span> │\n",
       "├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤\n",
       "│ dense_1 (<span style=\"color: #0087ff; text-decoration-color: #0087ff\">Dense</span>)                      │ (<span style=\"color: #00d7ff; text-decoration-color: #00d7ff\">None</span>, <span style=\"color: #00af00; text-decoration-color: #00af00\">1</span>)                   │             <span style=\"color: #00af00; text-decoration-color: #00af00\">513</span> │\n",
       "└──────────────────────────────────────┴─────────────────────────────┴─────────────────┘\n",
       "</pre>\n"
      ],
      "text/plain": [
       "┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━┓\n",
       "┃\u001b[1m \u001b[0m\u001b[1mLayer (type)                        \u001b[0m\u001b[1m \u001b[0m┃\u001b[1m \u001b[0m\u001b[1mOutput Shape               \u001b[0m\u001b[1m \u001b[0m┃\u001b[1m \u001b[0m\u001b[1m        Param #\u001b[0m\u001b[1m \u001b[0m┃\n",
       "┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━┩\n",
       "│ mobilenet_1.00_224 (\u001b[38;5;33mFunctional\u001b[0m)      │ (\u001b[38;5;45mNone\u001b[0m, \u001b[38;5;34m7\u001b[0m, \u001b[38;5;34m7\u001b[0m, \u001b[38;5;34m1024\u001b[0m)          │       \u001b[38;5;34m3,228,864\u001b[0m │\n",
       "├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤\n",
       "│ global_average_pooling2d             │ (\u001b[38;5;45mNone\u001b[0m, \u001b[38;5;34m1024\u001b[0m)                │               \u001b[38;5;34m0\u001b[0m │\n",
       "│ (\u001b[38;5;33mGlobalAveragePooling2D\u001b[0m)             │                             │                 │\n",
       "├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤\n",
       "│ dense (\u001b[38;5;33mDense\u001b[0m)                        │ (\u001b[38;5;45mNone\u001b[0m, \u001b[38;5;34m512\u001b[0m)                 │         \u001b[38;5;34m524,800\u001b[0m │\n",
       "├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤\n",
       "│ dropout (\u001b[38;5;33mDropout\u001b[0m)                    │ (\u001b[38;5;45mNone\u001b[0m, \u001b[38;5;34m512\u001b[0m)                 │               \u001b[38;5;34m0\u001b[0m │\n",
       "├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤\n",
       "│ dense_1 (\u001b[38;5;33mDense\u001b[0m)                      │ (\u001b[38;5;45mNone\u001b[0m, \u001b[38;5;34m1\u001b[0m)                   │             \u001b[38;5;34m513\u001b[0m │\n",
       "└──────────────────────────────────────┴─────────────────────────────┴─────────────────┘\n"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "text/html": [
       "<pre style=\"white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace\"><span style=\"font-weight: bold\"> Total params: </span><span style=\"color: #00af00; text-decoration-color: #00af00\">3,754,177</span> (14.32 MB)\n",
       "</pre>\n"
      ],
      "text/plain": [
       "\u001b[1m Total params: \u001b[0m\u001b[38;5;34m3,754,177\u001b[0m (14.32 MB)\n"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "text/html": [
       "<pre style=\"white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace\"><span style=\"font-weight: bold\"> Trainable params: </span><span style=\"color: #00af00; text-decoration-color: #00af00\">525,313</span> (2.00 MB)\n",
       "</pre>\n"
      ],
      "text/plain": [
       "\u001b[1m Trainable params: \u001b[0m\u001b[38;5;34m525,313\u001b[0m (2.00 MB)\n"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "text/html": [
       "<pre style=\"white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace\"><span style=\"font-weight: bold\"> Non-trainable params: </span><span style=\"color: #00af00; text-decoration-color: #00af00\">3,228,864</span> (12.32 MB)\n",
       "</pre>\n"
      ],
      "text/plain": [
       "\u001b[1m Non-trainable params: \u001b[0m\u001b[38;5;34m3,228,864\u001b[0m (12.32 MB)\n"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "model.summary()\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 21,
   "id": "24009b53-6e6c-4928-bb5d-e8263c8621b2",
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "C:\\Users\\HP\\anaconda3\\Lib\\site-packages\\keras\\src\\trainers\\data_adapters\\py_dataset_adapter.py:121: UserWarning: Your `PyDataset` class should call `super().__init__(**kwargs)` in its constructor. `**kwargs` can include `workers`, `use_multiprocessing`, `max_queue_size`. Do not pass these arguments to `fit()`, as they will be ignored.\n",
      "  self._warn_if_super_not_called()\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch 1/10\n",
      "\u001b[1m349/349\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m469s\u001b[0m 1s/step - accuracy: 0.7527 - loss: 0.4892 - val_accuracy: 0.9253 - val_loss: 0.1944\n",
      "Epoch 2/10\n",
      "\u001b[1m  1/349\u001b[0m \u001b[37m━━━━━━━━━━━━━━━━━━━━\u001b[0m \u001b[1m2:51\u001b[0m 493ms/step - accuracy: 0.9375 - loss: 0.1871"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "C:\\Users\\HP\\anaconda3\\Lib\\site-packages\\keras\\src\\trainers\\epoch_iterator.py:107: UserWarning: Your input ran out of data; interrupting training. Make sure that your dataset or generator can generate at least `steps_per_epoch * epochs` batches. You may need to use the `.repeat()` function when building your dataset.\n",
      "  self._interrupted_warning()\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\u001b[1m349/349\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m43s\u001b[0m 122ms/step - accuracy: 0.9375 - loss: 0.1871 - val_accuracy: 0.9256 - val_loss: 0.1939\n",
      "Epoch 3/10\n",
      "\u001b[1m349/349\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m376s\u001b[0m 1s/step - accuracy: 0.9010 - loss: 0.2366 - val_accuracy: 0.9303 - val_loss: 0.1715\n",
      "Epoch 4/10\n",
      "\u001b[1m349/349\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m44s\u001b[0m 126ms/step - accuracy: 0.9062 - loss: 0.1853 - val_accuracy: 0.9300 - val_loss: 0.1719\n",
      "Epoch 5/10\n",
      "\u001b[1m349/349\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m1019s\u001b[0m 3s/step - accuracy: 0.9146 - loss: 0.2059 - val_accuracy: 0.9407 - val_loss: 0.1533\n",
      "Epoch 6/10\n",
      "\u001b[1m349/349\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m45s\u001b[0m 128ms/step - accuracy: 0.9375 - loss: 0.1610 - val_accuracy: 0.9400 - val_loss: 0.1526\n",
      "Epoch 7/10\n",
      "\u001b[1m349/349\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m371s\u001b[0m 1s/step - accuracy: 0.9234 - loss: 0.1936 - val_accuracy: 0.9382 - val_loss: 0.1510\n",
      "Epoch 8/10\n",
      "\u001b[1m349/349\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m45s\u001b[0m 128ms/step - accuracy: 0.8750 - loss: 0.2256 - val_accuracy: 0.9389 - val_loss: 0.1508\n",
      "Epoch 9/10\n",
      "\u001b[1m349/349\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m371s\u001b[0m 1s/step - accuracy: 0.9265 - loss: 0.1859 - val_accuracy: 0.9450 - val_loss: 0.1408\n",
      "Epoch 10/10\n",
      "\u001b[1m349/349\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m45s\u001b[0m 127ms/step - accuracy: 0.9688 - loss: 0.1906 - val_accuracy: 0.9436 - val_loss: 0.1429\n"
     ]
    }
   ],
   "source": [
    "history = model.fit(\n",
    "    train_set,\n",
    "    validation_data=test_set,\n",
    "    epochs=10,  # Train for 10 epochs (adjust as needed)\n",
    "    steps_per_epoch=train_set.samples // train_set.batch_size,\n",
    "    validation_steps=test_set.samples // test_set.batch_size\n",
    ")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 27,
   "id": "01a711a8-ab65-444e-8c90-46e52f47bf61",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\u001b[1m88/88\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m44s\u001b[0m 503ms/step - accuracy: 0.9405 - loss: 0.1441\n"
     ]
    },
    {
     "data": {
      "text/plain": [
       "[0.14270152151584625, 0.9437679052352905]"
      ]
     },
     "execution_count": 27,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "model.evaluate(test_set)\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 31,
   "id": "87f55506-c71e-48b5-ad7c-e915990a23bb",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\u001b[1m88/88\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m45s\u001b[0m 506ms/step - accuracy: 0.9410 - loss: 0.1525\n",
      "Test Accuracy: 94.38%\n"
     ]
    }
   ],
   "source": [
    "# Evaluate the Model\n",
    "loss, accuracy = model.evaluate(test_set)\n",
    "print(f\"Test Accuracy: {accuracy * 100:.2f}%\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 32,
   "id": "23c4cb81-6eb3-4495-ac84-fa1c478c6f18",
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "WARNING:absl:You are saving your model as an HDF5 file via `model.save()` or `keras.saving.save_model(model)`. This file format is considered legacy. We recommend using instead the native Keras format, e.g. `model.save('my_model.keras')` or `keras.saving.save_model(model, 'my_model.keras')`. \n"
     ]
    }
   ],
   "source": [
    "\n",
    "model.save(\"military_vs_transport_model.h5\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 35,
   "id": "27577a1e-26fb-4a05-b26d-cd10648e8f54",
   "metadata": {},
   "outputs": [],
   "source": [
    "def predict_image(image_path, model):\n",
    "    # Load and preprocess the image\n",
    "    test_image = load_img(image_path, target_size=(224, 224))  # Resize to match model input\n",
    "    test_image = img_to_array(test_image) / 255.0  # Normalize pixel values\n",
    "    test_image = np.expand_dims(test_image, axis=0)  # Add batch dimension\n",
    "\n",
    "    # Make prediction\n",
    "    result = model.predict(test_image)\n",
    "    confidence = result[0][0]\n",
    "    if confidence < 0.5:\n",
    "        prediction = \"military\"\n",
    "    else:\n",
    "        prediction = \"other transport\"\n",
    "    return prediction, confidence"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 45,
   "id": "751b2072-de6d-4394-ac8a-b13d4f51dd76",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\u001b[1m1/1\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 116ms/step\n",
      "Prediction: military, Confidence: 0.00\n"
     ]
    }
   ],
   "source": [
    "image_path = \"C:/Users/HP/Downloads/military.webp\"  # Replace with your image path\n",
    "prediction, confidence = predict_image(image_path, model)\n",
    "print(f\"Prediction: {prediction}, Confidence: {confidence:.2f}\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 47,
   "id": "3d31a3f0-5eb3-4e5f-9f20-e3fe86acc637",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\u001b[1m1/1\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 103ms/step\n",
      "Prediction: other transport, Confidence: 1.00\n"
     ]
    }
   ],
   "source": [
    "image_path = \"C:/Users/HP/Downloads/v.webp\"  # Replace with your image path\n",
    "prediction, confidence = predict_image(image_path, model)\n",
    "print(f\"Prediction: {prediction}, Confidence: {confidence:.2f}\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 49,
   "id": "5b2c8c53-5755-4993-a7ac-c5955613f550",
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "WARNING:absl:You are saving your model as an HDF5 file via `model.save()` or `keras.saving.save_model(model)`. This file format is considered legacy. We recommend using instead the native Keras format, e.g. `model.save('my_model.keras')` or `keras.saving.save_model(model, 'my_model.keras')`. \n"
     ]
    }
   ],
   "source": [
    "model.save(\"military_vs_transport_model.h5\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 51,
   "id": "f0a4996b-d12d-4ec6-aaa1-4004cd32a7c3",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Requirement already satisfied: streamlit in c:\\users\\hp\\anaconda3\\lib\\site-packages (1.37.1)\n",
      "Requirement already satisfied: altair<6,>=4.0 in c:\\users\\hp\\anaconda3\\lib\\site-packages (from streamlit) (5.0.1)\n",
      "Requirement already satisfied: blinker<2,>=1.0.0 in c:\\users\\hp\\anaconda3\\lib\\site-packages (from streamlit) (1.6.2)\n",
      "Requirement already satisfied: cachetools<6,>=4.0 in c:\\users\\hp\\anaconda3\\lib\\site-packages (from streamlit) (5.3.3)\n",
      "Requirement already satisfied: click<9,>=7.0 in c:\\users\\hp\\anaconda3\\lib\\site-packages (from streamlit) (8.1.7)\n",
      "Requirement already satisfied: numpy<3,>=1.20 in c:\\users\\hp\\anaconda3\\lib\\site-packages (from streamlit) (1.26.4)\n",
      "Requirement already satisfied: packaging<25,>=20 in c:\\users\\hp\\anaconda3\\lib\\site-packages (from streamlit) (24.1)\n",
      "Requirement already satisfied: pandas<3,>=1.3.0 in c:\\users\\hp\\anaconda3\\lib\\site-packages (from streamlit) (2.2.2)\n",
      "Requirement already satisfied: pillow<11,>=7.1.0 in c:\\users\\hp\\anaconda3\\lib\\site-packages (from streamlit) (10.4.0)\n",
      "Requirement already satisfied: protobuf<6,>=3.20 in c:\\users\\hp\\anaconda3\\lib\\site-packages (from streamlit) (4.25.3)\n",
      "Requirement already satisfied: pyarrow>=7.0 in c:\\users\\hp\\anaconda3\\lib\\site-packages (from streamlit) (16.1.0)\n",
      "Requirement already satisfied: requests<3,>=2.27 in c:\\users\\hp\\anaconda3\\lib\\site-packages (from streamlit) (2.32.3)\n",
      "Requirement already satisfied: rich<14,>=10.14.0 in c:\\users\\hp\\anaconda3\\lib\\site-packages (from streamlit) (13.7.1)\n",
      "Requirement already satisfied: tenacity<9,>=8.1.0 in c:\\users\\hp\\anaconda3\\lib\\site-packages (from streamlit) (8.2.3)\n",
      "Requirement already satisfied: toml<2,>=0.10.1 in c:\\users\\hp\\anaconda3\\lib\\site-packages (from streamlit) (0.10.2)\n",
      "Requirement already satisfied: typing-extensions<5,>=4.3.0 in c:\\users\\hp\\anaconda3\\lib\\site-packages (from streamlit) (4.11.0)\n",
      "Requirement already satisfied: gitpython!=3.1.19,<4,>=3.0.7 in c:\\users\\hp\\anaconda3\\lib\\site-packages (from streamlit) (3.1.43)\n",
      "Requirement already satisfied: pydeck<1,>=0.8.0b4 in c:\\users\\hp\\anaconda3\\lib\\site-packages (from streamlit) (0.8.0)\n",
      "Requirement already satisfied: tornado<7,>=6.0.3 in c:\\users\\hp\\anaconda3\\lib\\site-packages (from streamlit) (6.4.1)\n",
      "Requirement already satisfied: watchdog<5,>=2.1.5 in c:\\users\\hp\\anaconda3\\lib\\site-packages (from streamlit) (4.0.1)\n",
      "Requirement already satisfied: jinja2 in c:\\users\\hp\\anaconda3\\lib\\site-packages (from altair<6,>=4.0->streamlit) (3.1.4)\n",
      "Requirement already satisfied: jsonschema>=3.0 in c:\\users\\hp\\anaconda3\\lib\\site-packages (from altair<6,>=4.0->streamlit) (4.23.0)\n",
      "Requirement already satisfied: toolz in c:\\users\\hp\\anaconda3\\lib\\site-packages (from altair<6,>=4.0->streamlit) (0.12.0)\n",
      "Requirement already satisfied: colorama in c:\\users\\hp\\anaconda3\\lib\\site-packages (from click<9,>=7.0->streamlit) (0.4.6)\n",
      "Requirement already satisfied: gitdb<5,>=4.0.1 in c:\\users\\hp\\anaconda3\\lib\\site-packages (from gitpython!=3.1.19,<4,>=3.0.7->streamlit) (4.0.7)\n",
      "Requirement already satisfied: python-dateutil>=2.8.2 in c:\\users\\hp\\anaconda3\\lib\\site-packages (from pandas<3,>=1.3.0->streamlit) (2.9.0.post0)\n",
      "Requirement already satisfied: pytz>=2020.1 in c:\\users\\hp\\anaconda3\\lib\\site-packages (from pandas<3,>=1.3.0->streamlit) (2024.1)\n",
      "Requirement already satisfied: tzdata>=2022.7 in c:\\users\\hp\\anaconda3\\lib\\site-packages (from pandas<3,>=1.3.0->streamlit) (2023.3)\n",
      "Requirement already satisfied: charset-normalizer<4,>=2 in c:\\users\\hp\\anaconda3\\lib\\site-packages (from requests<3,>=2.27->streamlit) (3.3.2)\n",
      "Requirement already satisfied: idna<4,>=2.5 in c:\\users\\hp\\anaconda3\\lib\\site-packages (from requests<3,>=2.27->streamlit) (3.7)\n",
      "Requirement already satisfied: urllib3<3,>=1.21.1 in c:\\users\\hp\\anaconda3\\lib\\site-packages (from requests<3,>=2.27->streamlit) (2.2.3)\n",
      "Requirement already satisfied: certifi>=2017.4.17 in c:\\users\\hp\\anaconda3\\lib\\site-packages (from requests<3,>=2.27->streamlit) (2025.1.31)\n",
      "Requirement already satisfied: markdown-it-py>=2.2.0 in c:\\users\\hp\\anaconda3\\lib\\site-packages (from rich<14,>=10.14.0->streamlit) (2.2.0)\n",
      "Requirement already satisfied: pygments<3.0.0,>=2.13.0 in c:\\users\\hp\\anaconda3\\lib\\site-packages (from rich<14,>=10.14.0->streamlit) (2.15.1)\n",
      "Requirement already satisfied: smmap<5,>=3.0.1 in c:\\users\\hp\\anaconda3\\lib\\site-packages (from gitdb<5,>=4.0.1->gitpython!=3.1.19,<4,>=3.0.7->streamlit) (4.0.0)\n",
      "Requirement already satisfied: MarkupSafe>=2.0 in c:\\users\\hp\\anaconda3\\lib\\site-packages (from jinja2->altair<6,>=4.0->streamlit) (2.1.3)\n",
      "Requirement already satisfied: attrs>=22.2.0 in c:\\users\\hp\\anaconda3\\lib\\site-packages (from jsonschema>=3.0->altair<6,>=4.0->streamlit) (23.1.0)\n",
      "Requirement already satisfied: jsonschema-specifications>=2023.03.6 in c:\\users\\hp\\anaconda3\\lib\\site-packages (from jsonschema>=3.0->altair<6,>=4.0->streamlit) (2023.7.1)\n",
      "Requirement already satisfied: referencing>=0.28.4 in c:\\users\\hp\\anaconda3\\lib\\site-packages (from jsonschema>=3.0->altair<6,>=4.0->streamlit) (0.30.2)\n",
      "Requirement already satisfied: rpds-py>=0.7.1 in c:\\users\\hp\\anaconda3\\lib\\site-packages (from jsonschema>=3.0->altair<6,>=4.0->streamlit) (0.10.6)\n",
      "Requirement already satisfied: mdurl~=0.1 in c:\\users\\hp\\anaconda3\\lib\\site-packages (from markdown-it-py>=2.2.0->rich<14,>=10.14.0->streamlit) (0.1.0)\n",
      "Requirement already satisfied: six>=1.5 in c:\\users\\hp\\anaconda3\\lib\\site-packages (from python-dateutil>=2.8.2->pandas<3,>=1.3.0->streamlit) (1.16.0)\n"
     ]
    }
   ],
   "source": [
    "!pip install streamlit"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 53,
   "id": "10e4cf99-c1f9-4154-9e88-7502159c711f",
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "WARNING:absl:Compiled the loaded model, but the compiled metrics have yet to be built. `model.compile_metrics` will be empty until you train or evaluate the model.\n",
      "2025-02-09 11:08:31.102 \n",
      "  \u001b[33m\u001b[1mWarning:\u001b[0m to view this Streamlit app on a browser, run it with the following\n",
      "  command:\n",
      "\n",
      "    streamlit run C:\\Users\\HP\\anaconda3\\Lib\\site-packages\\ipykernel_launcher.py [ARGUMENTS]\n"
     ]
    }
   ],
   "source": [
    "# app.py\n",
    "import streamlit as st\n",
    "from tensorflow.keras.models import load_model\n",
    "from tensorflow.keras.preprocessing.image import load_img, img_to_array\n",
    "import numpy as np\n",
    "from PIL import Image\n",
    "\n",
    "# Load the trained model\n",
    "model = load_model(\"military_vs_transport_model.h5\")\n",
    "\n",
    "# Set the title of the app\n",
    "st.title(\"Military vs. Transport Vehicle Classifier\")\n",
    "\n",
    "# Add a file uploader for image input\n",
    "uploaded_file = st.file_uploader(\"Upload an image of a vehicle\", type=[\"jpg\", \"jpeg\", \"png\"])\n",
    "\n",
    "# Function to preprocess the image and make predictions\n",
    "def predict_image(image):\n",
    "    # Resize the image to match the model's input shape\n",
    "    image = image.resize((224, 224))\n",
    "    # Convert the image to a numpy array\n",
    "    image = img_to_array(image) / 255.0  # Normalize pixel values\n",
    "    image = np.expand_dims(image, axis=0)  # Add batch dimension\n",
    "\n",
    "    # Make prediction\n",
    "    result = model.predict(image)\n",
    "    confidence = result[0][0]\n",
    "    if confidence < 0.5:\n",
    "        prediction = \"military\"\n",
    "    else:\n",
    "        prediction = \"other transport\"\n",
    "    return prediction, confidence\n",
    "\n",
    "# Display the uploaded image and make predictions\n",
    "if uploaded_file is not None:\n",
    "    # Load the image\n",
    "    image = Image.open(uploaded_file)\n",
    "    st.image(image, caption=\"Uploaded Image\", use_column_width=True)\n",
    "\n",
    "    # Make prediction\n",
    "    prediction, confidence = predict_image(image)\n",
    "    st.write(f\"**Prediction:** {prediction}\")\n",
    "    st.write(f\"**Confidence:** {confidence:.2f}\")"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python [conda env:base] *",
   "language": "python",
   "name": "conda-base-py"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.12.7"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
