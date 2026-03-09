# =====================================================================
# 1. Imports
# =====================================================================
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.utils import to_categorical

# =====================================================================
# 2. Carregar banco CIFAR-10
# =====================================================================
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

#----------------------------------------------------------------------
# Normalização para [0,1]
#---------------------------------------------------------------------
x_train = x_train.astype("float32") / 255.0
x_test  = x_test.astype("float32") / 255.0

# One-hot encoding dos rótulos
y_train_oh = to_categorical(y_train, 10)
y_test_oh  = to_categorical(y_test, 10)


# =====================================================================
# 3. Mostrar algumas imagens de treinamento
# =====================================================================
def mostrar_amostras():
    fig, axes = plt.subplots(2, 5, figsize=(10,4))
    idx = np.random.randint(0, x_train.shape[0], 10)

    for i, ax in enumerate(axes.flat):
        ax.imshow(x_train[idx[i]])
        ax.set_title(f"Label: {y_train[idx[i]][0]}")
        ax.axis("off")

    plt.show()

mostrar_amostras()


# =====================================================================
# 4. Definição da CNN conforme discutido
#    Camada 1: 32 filtros, cada filtro tem 3 kernels
#    Camada 2: 32 filtros, cada filtro tem 3 kernels (RGB)
#    Camada 3: 64 filtros, cada filtro tem 32 kernels da camada anterior
#    Camada 4: 64 filtros, cada filtro tem 32 kernels da camada anterior
# =====================================================================
model = Sequential([
    Conv2D(filters=32, kernel_size=(3,3), activation='relu', padding='same',
           input_shape=(32,32,3)),

    Conv2D(filters=32, kernel_size=(3,3), activation='relu', padding='same',
           input_shape=(32,32,3)),

    MaxPooling2D(pool_size=(2,2)),

    Dropout(0.25),

    Conv2D(filters=64, kernel_size=(3,3), activation='relu', padding='same'),

    Conv2D(filters=64, kernel_size=(3,3), activation='relu', padding='same'),

    MaxPooling2D(pool_size=(2,2)),

    Dropout(0.25),

    Flatten(),

    Dense(512, activation='relu'),

    Dropout(0.25),

    Dense(10, activation='softmax')
])

model.summary()


# =====================================================================
# 5. Compilação e Treinamento
# =====================================================================
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

history = model.fit(x_train, y_train_oh,
                    epochs=10,
                    batch_size=64,
                    validation_split=0.1)


# =====================================================================
# 6. Gráficos de evolução do erro e acurácia
# =====================================================================
def plot_metric(history, metric):
    plt.figure(figsize=(7,4))
    plt.plot(history.history[metric], label=f"Treino {metric}")
    plt.plot(history.history[f"val_{metric}"], label=f"Val {metric}")
    plt.xlabel("Época")
    plt.ylabel(metric.capitalize())
    plt.legend()
    plt.grid()
    plt.show()

plot_metric(history, "loss")
plot_metric(history, "accuracy")


# =====================================================================
# 7. Visualizar predições do modelo
# =====================================================================
def mostrar_predicoes(num=10):
    idx = np.random.randint(0, x_test.shape[0], num)
    imagens = x_test[idx]
    rotulos = y_test[idx]

    preds = model.predict(imagens)
    preds_cls = np.argmax(preds, axis=1)

    fig, axes = plt.subplots(2, num//2, figsize=(15,6))

    for i, ax in enumerate(axes.flat):
        ax.imshow(imagens[i])
        ax.set_title(f"Verdade: {rotulos[i][0]} | Pred: {preds_cls[i]}")
        ax.axis("off")

    plt.show()

mostrar_predicoes(10)
