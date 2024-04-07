from flask import Flask, render_template, request, redirect, url_for
from werkzeug.utils import secure_filename
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import os

app = Flask(__name__)

# アップロードされたファイルを保存するディレクトリ
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# 学習済みモデルのロード
model = load_model('mnist_model.h5')

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in {'png', 'jpg', 'jpeg', 'gif'}

def prepare_image(image_path):
    # 画像を読み込み、グレースケールに変換
    img = Image.open(image_path).convert('L')
    # 28x28にリサイズ
    img = img.resize((28, 28), Image.LANCZOS)
    # numpy配列に変換し、正規化
    img_array = np.asarray(img) / 255.0
    # モデルの入力形状に合わせるための次元追加
    img_array = img_array.reshape((1, 28, 28, 1))
    return img_array

@app.route('/')
def index():
    return render_template('index.html')
@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # 画像の前処理
        img_array = prepare_image(filepath)
        
        # 推論
        predictions = model.predict(img_array)
        result = np.argmax(predictions, axis=1)[0]  # 最も確率の高いクラス
        
        return render_template('result.html', result=result)

if __name__ == "__main__":
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)