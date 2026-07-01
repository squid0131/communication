# Webcam Posture Alert App

WindowsのWebカメラで姿勢悪化を検知する最小アプリです。

## 使い方

1. Pythonをインストール
2. このフォルダを開く
3. コマンドプロンプトで以下を実行

```bash
pip install -r requirements.txt
python app.py
```

## 操作

- `q` キーで終了

## 判定内容

- 鼻と肩中心のズレ
- 左右肩の高さ差
- 悪い姿勢が3秒続くと警告表示

## 注意

正面からWebカメラで映る前提です。横向きで猫背判定したい場合は、カメラを横に置いて耳・肩・腰の角度で判定する方式に変更できます。
