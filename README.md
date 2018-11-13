# Adversarial Discriminative Domain Adaptation
A tensorflow implement of [ADDA](cpipc.chinadegrees.cn)
## Environment
* python 3
* tensorflow 1.09
* sklearn
* matplotlib 2.2.2
* numpy 1.14.2

## Network
Network achitecture referenced LeNet for mnist digits datasets.
* For encoder:
    * 3 x Conv + 1 x Linear
* For classifier:
    * softmax (note that there are no trainable variables.)
* For Discriminator:
    * 3 x Linear
* More detial can be seem in "adda.py". You can design your network according to your adaptation task.

## Usage
**Note**: This repository is still semi-finished. Dataset only MNIST and USPS are support now. 
* step:
    * Step 1 is training source network.
    ```
    python main.py --step=1 --epoch=20
    ```
    * Step 2 is training target encoder and discriminator.
    ```
    python main.py --step=2 --epoch=2000
    ```
    * Step 3 is evaluation of adda network
    ```
    python main.py --step=3
    ```
* epoch: the training epoch in the step.
  
# Result
||MNIST(Source)|USPS(Target)|
|:--|:--:|:--:|
|Source Encoder + Source Classifier|99.23%|78.52%|
|Target Encoder + Source Classifier|-|92.83%|
* Target accuracy: 78.52%(without adapting) vs 92.83%(Adapted)
# Visualization result using t-SNE
* Orign imageand feature map of encoder output after adaptation

![before](./result/Samples_before_adaptation.png)![after](./result/Samples_after_adaptation.png)
