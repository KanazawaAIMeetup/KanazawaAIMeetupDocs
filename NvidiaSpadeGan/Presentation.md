# NVIDIAのSPADEと論文読むのに便利なツール紹介　（KAIM） 

この記事で引用・参照する文献は、非営利の勉強会で使用することを目的としております。

## 引用文献・参考文献

- Semantic Image Synthesis with Spatially-Adaptive Normalization (https://arxiv.org/abs/1903.07291) 参照日時：2019/05/19
- https://qiita.com/Phoeboooo/items/ad6c0461ab052aae8e89

## この論文はどんな論文？新規性は？
以前のpix2pixでは、セグメンテーションマップを入力とした画像生成において、セグメンテーションマップの情報を損失してしまう問題があった。SPADEではこれを解決した。SPADEは「Spatially-Adaptive Normalization」の略で、簡単にいうと、セグメンテーションマップの情報を考慮した特殊なバッチノーマライゼーションのようなものを導入することで、問題を解決した。

SPADEではGANのアルゴリズムを使用しており、正解データは実際の画像で、予測には教師データで用いる実際の画像から作成したセグメンテーションマップ（輪郭を描いたお絵かきの画像）を用いる。これは、人間が手で作成する。

実際に画像を生成するときには、Discriminatorのネットワークは使わず、セグメンテーションマスクと乱数を入力としたGeneratorのネットワークのみ使う。

SPADEと従来の手法の比較
![](https://i.imgur.com/bvEMohy.jpg)

## 変数の説明

![](https://i.imgur.com/Omqf3OA.png)　mは入力となるセグメンテーションマスク(semantic segmentation mask)　Lはセグメンテーションマップが書かれた画像。HはheightでWはwidth mは海、草原、空などのクラスを表すラベル。


![](https://i.imgur.com/OsmO2Nz.png)　i番目のレイヤーでの活性化関数の出力


![](https://i.imgur.com/YtpVkIr.png) i番目のレイヤーでのチャンネル数

![](https://i.imgur.com/ovkXie1.png)　

μは平均、σは分散　チャンネルCにおける活性化関数の出力の、平均と分散を計算する。

## ニューラルネットワークの構成（Generator）

![](https://i.imgur.com/eAt0SVP.png)

入力値は、ランダムベクトル（乱数）である。セグメンテーションマップ（お絵描きされた境界線の画像）は、ネットワークの中間層にある、SPADE ResBlkに注入する（渡す）。青のブロックはアップサンプリングを表すブロック。一番最初のレイヤーで乱数を入力とする理由は、生成される画像のバリエーションを豊かにするためである。また、モデルサイズが小さくて済むという利点もある。

各層で注入するセグメンテーションマップは、nearest neighbor down samplingで小さくしていく。逆に、生成する画像の方は、各層でnearest neighbor up samplingをして大きくしていく。

![](https://i.imgur.com/O42dgNj.png)

![](https://i.imgur.com/NSU4XBX.png)


## 中間層の活性化関数からの出力　（SPADE ResBlkの構造）
![](https://i.imgur.com/gTk0HLd.png)

ResBlkでは、各レイヤーでアップサンプリングしながら特徴量のマップを拡大する。最終的には、大きな画像が出力される。

Batch Normalizationに似ている。中間層からの活性化関数の出力、つまり特徴量マップを標準化した後に、γをかけ、バイアスβを足して次の層へ渡す。γやβはセグメンテーションマスクにCNNをかけた後の、平面のマップになる。γやβの計算にセグメンテーションマスクの情報を使用しているため、それ以外ではわざわざセグメンテーションマスクの画像を与えなくて良い

![](https://i.imgur.com/GDwreRH.png)

γ(m)と分数の掛け算のところにアダマール積を使っている。
![](https://i.imgur.com/3COLyK9.png)　この部分

γ(m)はセグメンテーションマップを入力として、CNNに通した後の特徴量マップのこと
β(m)はセグメンテーションマップを入力として、CNNに通した後の特徴量マップのこと
これらと、乱数を入力にしてSPADE ResBlkを通した後の中間層の値に対してアダマール積をとったり、足したりする。

batch normalizationの解説はこちら<br>
論文： https://arxiv.org/abs/1502.03167 <br>
Qiita: https://qiita.com/cfiken/items/b477c7878828ebdb0387



batch normalizationはただ出力を標準化している訳ではない。γとβをかけた後に出てくる特徴量マップは平均0、分散1ではない。

![](https://i.imgur.com/CfG6bL8.png)

iはの活性化関数を通した後の特徴量マップi番目のこと

hは乱数を入力としてSPADE ResBlkを何層も重ねた後の出力
mはセグメンテーションマップ（お絵描き）の画像

Batch Normalizationとの違いは、平均と分散の計算にセグメンテーションマップ（境界線のお絵描きの画像）を使用している点。

セグメンテーションマップ（お絵かきの画像）を、各レイヤーのサイズに合わせて縮小しながら渡す。

## Discriminator
{{:datascience:pasted:20190608-001022.png}}

- ネットワークの構成は、pix2pixHDやPatch GANと似た構造を採用。唯一の違いはinstance normalizationではなく、spectral normalizationを使用したこと。
{{:datascience:pasted:20190607-233910.png}}

- DiscriminatorはGeneratorによって生成された画像とセグメンテーションマスクを結合（Concatenate）したものを入力とする。

- LS-GANの損失関数ではなく、ヒンジ損失関数使った。
{{:datascience:pasted:20190607-233720.png}}
引用元：https://mathwords.net/hinge



## Encoder


- Encoder部分の役割は、生の画像から、平均と分散をニューラルネットワークを使って求めること。ここで求めた平均と分散は、Generator部分の入力となる乱数を生成するために使用する。乱数は、求められた平均と分散の分布から確率的にサンプリングする。
- 平均と分散をただしく出力できるように。誤差関数にKL Divergence Lossを使用する。(KL　Divergence Lossは確率分布の類似度を測る指標です。分布の重なった部分の面積をもとめるイメージ。今回は正規分布と求められた分布（平均と分散）が近くなるようすることが目的。逆に、正規分布に似ずに異なる分布になると、誤差が大きくなる。)
- 
{{:datascience:pasted:20190607-234250.png}}


## 豆知識
アダマール積・要素積を調べると良い。図の×、＋はアダマール積や足し算を表したもの。
アダマール積：
![](https://i.imgur.com/6CLniIW.png)　
引用元：https://ja.wikipedia.org/wiki/%E3%82%A2%E3%83%80%E3%83%9E%E3%83%BC%E3%83%AB%E7%A9%8D

よく考えるとCNNと似ている...

クロネッカー積は以下を参照（本論文とは関係ありません！）
https://ja.wikipedia.org/wiki/%E3%82%AF%E3%83%AD%E3%83%8D%E3%83%83%E3%82%AB%E3%83%BC%E7%A9%8D

## インストール方法(Windows)
### プロキシを通さなければいけない場合：
```
(base) C:\Users\tomohiro>notepad .condarc
```
![](https://i.imgur.com/qS0IXcO.png)

```
proxy_servers:
    http: http://www.sample.co.jp:8080
    https: https://www.sample.co.jp:8080
```

### anacondaのPythonでのコマンド（Anaconda Prompt上で実行する。）

```
conda create -n nvidia_spade python=3.5.6
conda activate nvidia_spade
conda install git

cd C:\
mkdir GithubClone
cd GithubClone
```
![](https://i.imgur.com/eX7wb2q.png)

```
git clone https://github.com/NVlabs/SPADE
cd SPADE

conda install -c pytorch pytorch
pip install -r requirements.txt
conda install tornado
```

![](https://i.imgur.com/9SMOAxV.png)

トレーニング済みモデルを以下からダウンロード

https://drive.google.com/uc?id=12gvlTbMvUcJewQlSEaZdeb2CdOB-b8kQ&export=download

ソースコードのルートフォルダ、つまりSPADEフォルダにcheckpointsフォルダを作成する。そのcheckpointsフォルダにダウンロードしたモデルをコピーする。

```
mkdir checkpoints
cd checkpoints
モデルをコピー（エクスプローラーを使うなどしてコピー）
tar xvf checkpoints.tar.gz
cd ../
```

### 解凍ソフトが無い方はダウンロード、checkpoints.tar.gzを解凍する
Lhaplus(脆弱性があったり、パスの長さの問題に対応していないため推奨ではありませんが、困った方はLhaplusインストールしてください。)

https://forest.watch.impress.co.jp/library/software/lhaplus/

### 正しくインストールできたかチェックするために、とりあえずテストを実行(C:\GithubClone\　などのパスの部分は適宜変更してください)
```
python test.py --name coco_pretrained --dataset_mode coco --dataroot C:\GithubClone\SPADE\datasets\coco_stuff
```

## GUIのツール
以下のソースは私がWindowsでも動くようにしたGUIのソースコードです。

- https://github.com/llDataSciencell/SmartSketchNvidiaSpadeForWindows

つい最近、もう一つ良さげなGUIのソースコードがGithubにあるのを発見しました。

- https://github.com/SDBagel/SPADE-GUI

# 論文読むのに便利なツールの紹介

1. pdfをダウンロードして、Dropboxにアップロードする。
2. Chromeの拡張機能である「iKnow!　ポップアップ辞書」や「Weblioポップアップ英和辞典」(https://chrome.google.com/webstore/detail/weblio%E3%83%9D%E3%83%83%E3%83%97%E3%82%A2%E3%83%83%E3%83%97%E8%8B%B1%E5%92%8C%E8%BE%9E%E5%85%B8/oingodpdjohhkelnginmkagmkbplgema?hl=ja)をインストールする。
3. Dropbox上でpdfを開き、わからない単語にカーソルを載せると、英単語の日本語訳や発音の仕方が表示される。



注意：英熟語にはあまり対応していません。
