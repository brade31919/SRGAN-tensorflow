# SRGAN_tensorflow

### Introduction
This project is a tensorflow implementation of the impressive work  [Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network](https://arxiv.org/pdf/1704.02470v1.pdf). <br />
The result on BSD100, Set14, Set5 will be reported later. The code is highly inspired by the [pix2pix-tensorflow](https://github.com/affinelayer/pix2pix-tensorflow).
#### Some results:
* The comparison of some result form my implementation and the paper

<table >
    <tr >
    	<td><center>Inputs</center></td>
        <td><center>Our result</center></td>
        <td><center>SRGAN result</center></td>
        <td><center>Original</center></td>
    </tr>
    <tr>
    	<td>
    		<center><img src="./pic/SRGAN/comic_LR.png" height="280"></center>
    	</td>
    	<td>
    		<center><img src="./pic/images/img_005-outputs.png" height="280"></center>
    	</td>
        <td>
        	<center><img src="./pic/SRGAN/comic_SRGAN-VGG54.png" height="280"></center>
        </td>
        <td>
        	<center><img src="./pic/SRGAN/comic_HR.png" height="280"></center>
        </td>
    </tr>
    <tr>
    	<td><center>Inputs</center></td>
        <td><center>Our result</center></td>
        <td><center>SRGAN result</center></td>
        <td><center>Original</center></td>
    </tr>
    <tr>
    	<td>
    		<center><img src="./pic/SRGAN/baboon_LR.png" height="200"></center>
    	</td>
        <td>
        	<center><img src="./pic/images/img_001-outputs.png" height="200"></center>
        </td>
        <td>
        	<center><img src="./pic/SRGAN/baboon_SRGAN-VGG54.png" height="200"></center>
       </td>
       <td>
        	<center><img src="./pic/images/img_001-targets.png" height="200"></center>
        </td>
    </tr>
</table>

### Denpendency
* tensorflow (tested on r1.0, r1.2)
* TF slim library
* Download and extract the pre-trained model from my [google drive](https://drive.google.com/a/gapp.nthu.edu.tw/uc?id=0BxRIhBA0x8lHNDJFVjJEQnZtcmc&export=download)

### Recommended
* Ubuntu 16.04 with tensorflow GPU edition

### Getting Started

```bash
# clone the repository from github
git clone https://github.com/brade31919/SRGAN-tensorflow.git
cd SRGAN-tensorflow/

# Download the pre-trained model from the google-drive
# Go to https://drive.google.com/a/gapp.nthu.edu.tw/uc?id=0BxRIhBA0x8lHNDJFVjJEQnZtcmc&export=download
# Download the pre-trained model to SRGAN-tensorflow/
tar xvf SRGAN_pre-trained.tar

# Run the test mode
sh test_SRGAN.sh

#The result can be viewed at SRGAN-tensorflow/result/images/
```

### More result on benchmark

####Coming soon!!!