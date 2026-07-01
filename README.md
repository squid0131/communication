# Posture Background Blur Advanced

Webカメラで姿勢をバックグラウンド検知し、姿勢が悪くなるとPC画面全体にぼかしをかけるWindows向けデモアプリです。

## 特徴

- 自分のカメラ映像は表示しません
- Webカメラはバックグラウンドで姿勢検知
- 起動後5秒間で良い姿勢を自動キャリブレーション
- 正面姿勢と横向き姿勢を切り替えて評価
- 過度な覗き込み、猫背、肩の傾き、頭の下がりを減点
- 点数が低いほど画面全体のぼかしが強くなります

## インストール

```bash
pip install -r requirements.txt
```

## 起動

```bash
python app.py
```

## 終了

コマンドプロンプトで `Ctrl + C` を押してください。

ぼかし画面が出ている場合は `ESC` でも終了できます。

## 注意

このアプリはプレゼン・課題用のプロトタイプです。
Windowsの画面そのものを変更するのではなく、スクリーンショットをぼかした全画面オーバーレイを重ねる方式です。

## ファイル構成

```text
posture_background_blur_advanced/
├── First_webcamera //一番最初の色でわかるやつ
├── webcam_posture_app //グレーになったやつ
├── app.py
├── posture_analyzer.py
├── screen_blur_overlay.py
├── requirements.txt
└── README.md
```
