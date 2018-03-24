<img src='imgs/fig2.png' align="right" width=384>

# Adversarial Dropout Regularization
This is the implementation of Adversarial Dropout Regularization in Pytorch.
The code is written by Kuniaki Saito. The work was accepted by ICLR 2018.
#### Adversarial Dropout Reguralization: [[Project]](https://github.com/mil-tokyo/adr_da) [[Paper]](https://openreview.net/forum?id=HJIoJWZCZ).
<img src='imgs/picture_adr.png' width=900>

## Getting Started
### Installation
- Install PyTorch (Works on Version 0.2.0_3) and dependencies from http://pytorch.org.
- Due to the change of calculation of kl divergence, it may not work for newer version.
- Install Torch vision from the source.
- Install torchnet as follows
```
pip install git+https://github.com/pytorch/tnt.git@master
```
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

In case of version conflict of Pytorch, use the option --use_abs_diff, which will change the measurement from kl divergence to absolute difference.
```
python main.py --source svhn --target mnist --use_abs_diff
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

## License
MIT
