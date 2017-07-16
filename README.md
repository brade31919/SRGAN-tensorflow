# SRGAN_tensorflow

### Introduction
This project is a tensorflow implementation of the impressive work  [Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network](https://arxiv.org/pdf/1704.02470v1.pdf). <br />
The result on BSD100, Set14, Set5 will be reported later. The code is highly inspired by the [pix2pix-tensorflow](https://github.com/affinelayer/pix2pix-tensorflow).
##### Some results:
* The result from the paper:
<table>
    <tr>
        <td>SRGAN result</td>
        <td>Original</td>
        <td>SRGAN result</td>
        <td>Original</td>
    </tr>
    <tr>
        <td><img src="./pic/SRGAN/comic_SRGAN-VGG54.png" width="50%"></td>
        <td><img src="./pic/SRGAN/comic_HR.png" width="50%"></td>
        <td><img src="./pic/SRGAN/baboon_SRGAN-VGG54.png" width="50%"></td>
        <td><img src="./pic/SRGAN/baboon_HR.png" width="50%"></td>
    </tr>
    <tr>
        <td>Our result</td>
        <td>Original</td>
        <td>Our result</td>
        <td>Original</td>
    </tr>
    <tr>
        <td><img src="./pic/images/img_005-outputs.png" width="50%"></td>
        <td><img src="./pic/images/img_005-targets.png" width="50%"></td>
        <td><img src="./pic/images/img_001-outputs.png" width="50%"></td>
        <td><img src="./pic/images/img_001-targets.png" width="50%"></td>
    </tr>
</table>

* The result from my implementation:
### Denpendency
* tensorflow (tested on r1.0, r1.2)
* TF slim library

### Recommended
* Ubuntu 16.04 with tensorflow GPU edition

### Getting Started

```bash
# clone the repository from github
git clone https://github.com/brade31919/SRGAN-tensorflow.git
cd SRGAN-tensorflow/

# [optional] download the training dataset

```