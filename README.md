# **PredExrate**

**PredExrate** は、為替レートの予測および金融モデリングを目的とした Python アプリケーションです。
本プロジェクトでは、ジャンプ拡散モデル、カルマンフィルタ、確率的ボラティリティモデルなどの手法を用いて為替データを解析します。
（※作成したモデルを精査せず追加していますのであまり深追いしないように...。学習データはyahoo finace等で取得しています。）

## **📌 主な機能**
- ✅ **為替レートの予測** （統計モデルを使用）
- ✅ **金融時系列データの解析**
- ✅ **ジャンプ拡散モデル・カルマンフィルタの実装**
- ✅ **結果の可視化**

---

## **📥 インストール**
### **🔧 必要なライブラリ**
以下の Python ライブラリが必要です。

```bash
pip install numpy pandas matplotlib scipy statsmodels
```

また、`python-can` や `asammdf` を使用する場合は、追加でインストールしてください。

```bash
pip install python-can asammdf
```

---

## **🚀 使い方**
### **1️⃣ 予測の実行**
以下のコマンドを実行すると、為替レートの予測が開始されます。

```bash
python run.py
```

---

## **📂 フォルダ構成**
```
PredExrate-main/
├── run.py                  # メインスクリプト
├── test.ipynb              # Jupyter Notebook（テスト用）
├── gpt_src/                # 金融モデルの実装
│   ├── IndexModel.py       # インデックスモデル
│   ├── JumpDiffusionModel.py # ジャンプ拡散モデル
│   ├── KalmanFilter.py     # カルマンフィルタ
│   ├── Model.py            # モデルベースクラス
│   ├── StochasticVolalityModel.py # 確率的ボラティリティモデル
├── out/utils/manzen/       # 出力データ（例: manzen.png）
├── ref/                    # 参照ファイル
│   ├── structure_class.pu  # クラス構造の定義？
└── README.md               # 本ドキュメント
```

---

## **⚠️ 注意点**
1. データセットは適宜追加してください。
2. モデルのパラメータは `gpt_src/` 内の各スクリプトで調整可能です。
3. `run.py` 実行前に、適切なライブラリがインストールされているか確認してください。

---

## **📜 ライセンス**
MIT License © 2025 Your Name / Your Organization

---

## **📩 お問い合わせ**
バグ報告や機能追加のリクエストは、GitHub の **Issues** に投稿してください！
🚀 Happy Coding! 🚀
