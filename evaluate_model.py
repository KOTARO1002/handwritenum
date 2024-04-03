# 必要なライブラリのインポート
from tensorflow.keras.models import load_model
import numpy as np
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# MNISTデータセットの読み込み (この部分は既にトレーニングデータとテストデータがある場合は不要です)
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
test_images = test_images.reshape((10000, 28, 28, 1)).astype('float32') / 255
test_labels = to_categorical(test_labels)

# モデルのロード
model = load_model('mnist_model.h5')

# モデルの評価
test_loss, test_acc = model.evaluate(test_images, test_labels)
print(f'Test accuracy: {test_acc}')

# 推論
predictions = model.predict(test_images[:5])
predicted_classes = np.argmax(predictions, axis=1)
true_classes = np.argmax(test_labels[:5], axis=1)
print(f'Predicted classes: {predicted_classes}')
print(f'True classes: {true_classes}')

# 混同行列を生成
predictions_full = model.predict(test_images)
predicted_classes_full = np.argmax(predictions_full, axis=1)
true_classes_full = np.argmax(test_labels, axis=1)
cm = confusion_matrix(true_classes_full, predicted_classes_full)

# 混同行列をプロット
plt.figure(figsize=(10,8))
sns.heatmap(cm, annot=True, fmt="d", cmap='Blues')
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.show()

# 分類レポートを表示
print(classification_report(true_classes_full, predicted_classes_full))
