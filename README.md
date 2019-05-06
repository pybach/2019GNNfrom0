# 課題提出物説明

## ファイルの概要
- `src/`ディレクトリ下に、`gnn1.py`、`gnn2.py`、`gnn3.py`、`gnn4.py`の4ファイル（Pythonスクリプト）が置かれています。番号がそれぞれ課題番号に対応します。例えば課題3は`gnn3.py`で行っています。

- 全てPython3で書かれています。動作確認はmacOS Sierra上にpyenvでインストールしたPython 3.7.0で行いました。インポートするパッケージはnumpy、math、matplotlibの3つです。

- 各ファイルを直接、`$ python gnn1.py`
のように実行すると各課題のテストプログラムが走ります。

- テストプログラムは、データセットがスクリプトから見て`../../datasets/`以下に置かれている（`datasets/`が、本ファイルの置かれているディレクトリの兄弟ディレクトリである）ことを想定して書かれています。

- `gnn4.py`は`gnn3.py`を、`gnn3.py`は`gnn2.py`を、`gnn2.py`は`gnn1.py`を、それぞれ内部でインポートします。

- コード内容の説明は、ソースコード中にコメントとして直接書かれています。

## 内容
- 本回答では、まず課題1でクラス`GNN1`を定義します。次いで課題2でそのサブクラス`GNN2`を、さらに課題3で`GNN3`、課題4で`GNN4`と、順に継承して機能を追加していきます。上記の依存関係があるのはそのためです。

- GNNクラスの他に、`gnn3.py`と`gnn4.py`では(momentum)SGDやAdamを実現するためのオプティマイザクラスを定義しています。

- 課題4では、Adamの実装および、集約2のニューラルネット多層化を行いました。

## 課題3
ここでは課題2で実装したクラス`GNN2`に機能を追加した派生クラス`GNN3`と、それと独立なオプティマイザクラス`SGD`および`MomentumSGD`を実装した。

`GNN3`に追加した主要な要素はメソッド`GNN3.fit(self,x,y,optimizer)`である。`GNN3.fit()`は学習データをランダムに並べ替えてミニバッチに分割し、ミニバッチごとにパラメータ集合`Theta=(W,A,b)`の勾配平均を求めて`Theta`を更新する。  

勾配平均と更新量の関係はSGDやmomentum SGDといった最適化アルゴリズムにより異なる。この部分は`GNN3.fit()`の引数`optimizer`に`Optimizer`派生型オブジェクトを渡すことで可変としている。

`Optimizer`はメソッド`update(self,grad)`のみを持つ基底クラスである。派生クラスにおいて、このメソッドは現在のミニバッチでの勾配平均を受け取り、必要に応じて内部状態（`MomentumSGD`では`w`。`SGD`には無い）を更新した後、更新量を返す。

以上のような構造とすることで、`GNN3`の構造・実装を極力簡易にするとともに、新しい最適化アルゴリズムにも容易に対応可能な拡張性を実現した。