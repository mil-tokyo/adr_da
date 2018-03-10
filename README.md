<img src='imgs/fig2.png' align="right" width=384>

# Adversarial Dropout Regularization
This is the implementation of Adversarial Dropout Regularization in Pytorch.
The code is written by Kuniaki Saito.
#### Adversarial Dropout Reguralization: [[Project]]() [[Paper]](https://openreview.net/forum?id=HJIoJWZCZ).
<img src='imgs/picture_adr.png' width=900>

## Getting Started
### Installation
- Install PyTorch and dependencies from http://pytorch.org
- Install Torch vision from the source.
## Download Dataset
Download MNIST Dataset [here](). Resized image dataset is contained in the file.
Place it in the directory ./data.
SVHN Dataset and place it in !.
USPS dataset and place it in ~.

### ADR train/evaluation
For example, adaptation from svhn to mnist.
```
python main.py --source svhn --target mnist
```


## Citation
If you use this code for your research, please cite our papers.
```
@article{saito2017adversarial,
  title={Adversarial Dropout Regularization},
  author={Saito, Kuniaki and Ushiku, Yoshitaka and Harada, Tatsuya and Saenko, Kate},
  journal={arXiv preprint arXiv:1711.01575},
  year={2017}
}

```


