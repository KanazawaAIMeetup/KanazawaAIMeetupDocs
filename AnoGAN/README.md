###### tags: `KAIM`

# AnoGANについて

# GANによる異常検知の直感的な理解

この記事ではGANを用いた異常検知やAutoEncoderを用いた異常検知のを初心者でもわかる直感的な説明で紹介しようと思います。GANを用いた異常検知は、教師データが少ない場合に威力を発揮します。なので、完全に教師なし学習として使われるケースはあまり無いと思います。

ただし、一部の異常検知の手法は異常である/ないといったラベル（教師データ）は必要ありません。これは、教師なしの異常検知と呼ばれます。(AIにデータを渡すと、勝手に異常と思われるデータを除外してくれるイメージです。もちろん、閾値は決めなければならないですし、正しい保証もありません。)詳しくは、時間があれば書き足します。

# 異常検知って
異常検知というと、AutoEncoderを私はすぐに思い浮かべます。しかし、近年GANを用いた手法なども盛んに開発されています。2018年に聞いた話なのですが、製造業の企業でGANを用いて既に異常検知をしている企業もあるという話を伺いました。

## Auto Encoderによる異常検知

Auto Encoderは、エンコーダとデコーダを用いることにより、入力画像と同じ画像を出力させることを目的としたディープラーニングの手法です。

![](https://i.imgur.com/6UVbxGX.png)

Auto Encoderは入力データ（今回は画像）をそのまま教師として学習させるネットワークですが、このネットワークを異常検知に使うこともできます。(現在はVariational Auto Encoder[^1]を用いた手法や、metric learningを用いた手法が主流のようですが、今回はAnoGANの説明のために紹介します)

アルゴリズムの流れを解説します。

1. 学習させたい画像のデータセットと、異常検知を行いたい画像を用意する
2. 入力画像(X)をそのまま教師データ(Y)とし、Auto Encoderを学習させる。
3. 異常検知を行いたい画像をAuto Encoderに入力し、画像を出力させる。
4. 異常検知を行いたいオリジナルの画像と、Auto Encoderから出力された画像の類似度を求め、類似度がある閾値より低ければ異常とみなす。

Auto Encoderを用いた異常検知は、今まで学習してきた画像と似ている画像であれば、上手くAuto Encoderで復元できるが、今まで見たことの無いような画像であれば上手く復元できないといったディープニューラルネットワークの性質を利用した異常検知の手法と言えます。

## DCGAN
今回は、DCGANを異常検知を目的としない、ただのGANとして説明します。

### DCGANのソースコード
引用元： https://github.com/eriklindernoren/Keras-GAN/blob/master/dcgan/dcgan.py 
```python
from __future__ import print_function, division

from keras.datasets import mnist
from keras.layers import Input, Dense, Reshape, Flatten, Dropout
from keras.layers import BatchNormalization, Activation, ZeroPadding2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Sequential, Model
from keras.optimizers import Adam

import matplotlib.pyplot as plt

import sys

import numpy as np

class DCGAN():
    def __init__(self):
        # Input shape
        self.img_rows = 28
        self.img_cols = 28
        self.channels = 1
        self.img_shape = (self.img_rows, self.img_cols, self.channels)
        self.latent_dim = 100

        optimizer = Adam(0.0002, 0.5)

        # Build and compile the discriminator
        self.discriminator = self.build_discriminator()
        self.discriminator.compile(loss='binary_crossentropy',
            optimizer=optimizer,
            metrics=['accuracy'])

        # Build the generator
        self.generator = self.build_generator()

        # The generator takes noise as input and generates imgs
        z = Input(shape=(self.latent_dim,))
        img = self.generator(z)

        # For the combined model we will only train the generator
        self.discriminator.trainable = False

        # The discriminator takes generated images as input and determines validity
        valid = self.discriminator(img)

        # The combined model  (stacked generator and discriminator)
        # Trains the generator to fool the discriminator
        self.combined = Model(z, valid)
        self.combined.compile(loss='binary_crossentropy', optimizer=optimizer)

    def build_generator(self):

        model = Sequential()

        model.add(Dense(128 * 7 * 7, activation="relu", input_dim=self.latent_dim))
        model.add(Reshape((7, 7, 128)))
        model.add(UpSampling2D())
        model.add(Conv2D(128, kernel_size=3, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Activation("relu"))
        model.add(UpSampling2D())
        model.add(Conv2D(64, kernel_size=3, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Activation("relu"))
        model.add(Conv2D(self.channels, kernel_size=3, padding="same"))
        model.add(Activation("tanh"))

        model.summary()

        noise = Input(shape=(self.latent_dim,))
        img = model(noise)

        return Model(noise, img)

    def build_discriminator(self):

        model = Sequential()

        model.add(Conv2D(32, kernel_size=3, strides=2, input_shape=self.img_shape, padding="same"))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(Conv2D(64, kernel_size=3, strides=2, padding="same"))
        model.add(ZeroPadding2D(padding=((0,1),(0,1))))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(Conv2D(128, kernel_size=3, strides=2, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(Conv2D(256, kernel_size=3, strides=1, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(Flatten())
        model.add(Dense(1, activation='sigmoid'))

        model.summary()

        img = Input(shape=self.img_shape)
        validity = model(img)

        return Model(img, validity)

    def train(self, epochs, batch_size=128, save_interval=50):

        # Load the dataset
        (X_train, _), (_, _) = mnist.load_data()

        # Rescale -1 to 1
        X_train = X_train / 127.5 - 1.
        X_train = np.expand_dims(X_train, axis=3)

        # Adversarial ground truths
        valid = np.ones((batch_size, 1))
        fake = np.zeros((batch_size, 1))

        for epoch in range(epochs):

            # ---------------------
            #  Train Discriminator
            # ---------------------

            # Select a random half of images
            idx = np.random.randint(0, X_train.shape[0], batch_size)
            imgs = X_train[idx]

            # Sample noise and generate a batch of new images
            noise = np.random.normal(0, 1, (batch_size, self.latent_dim))
            gen_imgs = self.generator.predict(noise)

            # Train the discriminator (real classified as ones and generated as zeros)
            d_loss_real = self.discriminator.train_on_batch(imgs, valid)
            d_loss_fake = self.discriminator.train_on_batch(gen_imgs, fake)
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

            # ---------------------
            #  Train Generator
            # ---------------------

            # Train the generator (wants discriminator to mistake images as real)
            # モデル２つを合体させたものに、ノイズzを入力として、Generatorで画像を生成。
            # そのまま画像をDiscriminatorに入力して、1.0（つまり正常な画像)として認識された場合、誤差が小さくなる(Discriminatorをだますほど自然な画像が生成できたということ)
            g_loss = self.combined.train_on_batch(noise, valid)

            # Plot the progress
            print ("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (epoch, d_loss[0], 100*d_loss[1], g_loss))

            # If at save interval => save generated image samples
            if epoch % save_interval == 0:
                self.save_imgs(epoch)

    def save_imgs(self, epoch):
        r, c = 5, 5
        noise = np.random.normal(0, 1, (r * c, self.latent_dim))
        gen_imgs = self.generator.predict(noise)

        # Rescale images 0 - 1
        gen_imgs = 0.5 * gen_imgs + 0.5

        fig, axs = plt.subplots(r, c)
        cnt = 0
        for i in range(r):
            for j in range(c):
                axs[i,j].imshow(gen_imgs[cnt, :,:,0], cmap='gray')
                axs[i,j].axis('off')
                cnt += 1
        fig.savefig("images/mnist_%d.png" % epoch)
        plt.close()


if __name__ == '__main__':
    dcgan = DCGAN()
    dcgan.train(epochs=4000, batch_size=32, save_interval=50)
```



## AnoGAN

AnoGAN[^2][^3]は通常のDCGANで学習を行ったGenerator部分のネットワークを用いて、少し工夫を施すことで異常検知にも使えるようにするという手法です。

![](https://i.imgur.com/QwxwRGI.png)


アルゴリズムの流れを説明します。

1. まず、トレーニングデータとテストデータ（画像など）を用意する。
1. 通常のDCGANで学習を行う。AnoGANは、このネットワークを再利用する。
1. テストデータ(正常な画像と異常な画像両方含まれる）を参考にGeneratorの入力となる最適と思われる乱数zを生成し、その乱数を元にテストデータと似たような画像を出力できるようにする。(注意：GeneratorやDiscriminatorの学習は一切行いません)
1. 乱数zをGeneratorに入力して、画像を出力してみる。(正常な画像、つまりDCGANを学習する段階で多く経験した似たような画像はテスト画像に似たような画像を上手く出力できるのに対し、異常画像を元に乱数を生成し、それを元にGeneratorから画像を出力すると上手く出力できない。)
1. 評価する。テスト画像と出力された画像の類似度を、画素ごとの引き算を行い絶対値をとったものなどで求める。正確には、それに加えて、テスト画像と、出力された画像それぞれを用いてDiscriminatorに通し、出力層の一個手前の層で出現した特徴量を引き算し、絶対値をとったものも類似度を測る指標として使う。

### 乱数zの生成方法

通常ディープラーニングでは、重みなどのパラメータの値を偏微分（バックプロパゲーション）により学習させますが、誤差が最小となる最適な入力値の値も偏微分で求めることができます。入力zはGeneratorネットワークの入力として使われます。アルゴリズムは、

1. まず、zとなる乱数をランダムに生成する。
2. ランダムに生成された入力値zを、偏微分することにより（微小な値zを大きく、または小さくして)誤差が減る方向に少しずつ修正していく。数百回程度繰り返す。

### AnoGANのイメージ

肝心のイメージですが、AutoEncoderを用いた異常検知と似ていると思います。今まで多く見てきた画像とテスト画像が似ていれば、少ない情報に変換して（zという乱数に変換して）再び復元しようとした場合、上手く復元できますが、今までAIが見てきた画像と大きく異なる画像の場合、一旦少ない情報に変換して(zという値に変換して)元のテスト画像を復元しようとしても上手くいきません。

上手く復元できたかどうか=今までAIが見てきた画像と似ている正常な画像かどうか　です。
似ているかどうかは、類似度を測って求めます。類似度の測り方については様々な手法があります。例えば、Cos類似度、KL情報量(ヒストグラムを書いて重なっている部分の面積が大きいか、小さいか)、単純に2枚の写真の同じ位置の画素同士を引き算する　などがあります。

## AnoGANの誤差関数

AnoGANでは、residual lossとdiscrimination lossの二つを割合を考えて足し合わせたものを誤差関数として用います。

テスト画像（判定したい画像）と乱数を元に生成された画像について、同じ位置の画素同士を引き算を行い絶対値をとったものの合計を、residual loss（累積的な誤差）と呼びます。

一方、discrimination lossはテスト画像と、テスト画像を元に生成された乱数を元に、復元された（生成された）画像両方をGANのdiscriminator部分に与え、出力層の一個手前の特徴量同士を引き算し、絶対値をとったものである。

![](https://i.imgur.com/56xml9L.png)


誤差関数の数式は以下のようになります。

```math
loss = (1-λ) × residual loss + λ × discrimination loss
```

やっていることは単純。AnoGANの論文ではλに0.1が採用された。つまり、residual lossに9割、discrimination lossは1割の重み付けを行う。residual lossの方が大事だと考えている。

## ハンズオン（力尽きた...）

「つくりながら学ぶ! PyTorchによる発展ディープラーニング[^4]」という本のサンプルコード[^5]がgithubで手に入ります。詳しくはこちらを参照してください。


## 教師なしの異常検知
教師ありの異常検知は皆さんどのように実装すれば良いか簡単に想像がつくと思いますが、教師なしの異常検知はパッと考えただけでは思いつかないと思います。

教師なしの異常検知の動作の仕組みを述べると、

- データの中から、外れ値のようなもの（例えば他の写真と大きく異なる）を探す
- 閾値を決めて大きく外れていると思われるデータは除外する

といった感じになります。詳しくは時間があれば追記します。

## GANの勉強にオススメの教材

### Keras-GAN ( GitHub)
- https://github.com/eriklindernoren/Keras-GAN

### Pytorch-GAN
- https://github.com/eriklindernoren/PyTorch-GAN

## 引用元、参考文献のリスト
[1] https://qiita.com/kenmatsu4/items/b029d697e9995d93aa24

[2] Thomas Schlegl, Philipp Seeböck, Sebastian M. Waldstein, Ursula Schmidt-Erfurth, Georg Langs. Unsupervised Anomaly Detection with Generative Adversarial Networks to Guide Marker Discovery(https://arxiv.org/abs/1703.05921)

[3] AnoGAN (https://link.springer.com/chapter/10.1007/978-3-319-59050-9_12)

[4] つくりながら学ぶ！PyTorchによる発展ディープラーニン(https://book.mynavi.jp/ec/products/detail/id=104855)

[5] YutaroOgawa/pytorch_advanced: 書籍「つくりながら学ぶ! PyTorchによる発展ディープラーニング」の実装コードを配置したリポジトリです(https://github.com/YutaroOgawa/pytorch_advanced/blob/master/6_gan_anomaly_detection/6-2_AnoGAN.ipynb)
[6]  (https://qiita.com/kenmatsu4/items/b029d697e9995d93aa24)

