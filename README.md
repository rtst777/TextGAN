# TextGAN  
## Overview
Generative Adversarial Nets (GANs) face problems when dealing with the tasks of generating discrete data. This non-differentiability problem can be addressed byusing gradient estimators. In theory, the bias and variance of these estimators have been discussed, but there has not been much work done on testing them on GAN-Based Text Generation. We will be analyzing the bias and variance of two gradient estimators, Gumbel-Softmax and [REBAR](http://papers.nips.cc/paper/6856-rebar-low-variance-unbiased-gradient-estimates-for-discrete-latent-variable-models.pdf), on GAN-Based Text Generation. We propose two sets of experiments based on differing sentence length and vocabulary size to analyse bias and variance, respectively. In this work, we evaluate the biasand variance impact of the above two gradient estimators on GAN Text Generation problem. We also create a novel GAN-Based Text Generation model on top of [RelGAN](https://openreview.net/pdf?id=rJedV3R5tm) by replacing Gumbel-Softmax with [REBAR](http://papers.nips.cc/paper/6856-rebar-low-variance-unbiased-gradient-estimates-for-discrete-latent-variable-models.pdf). The performance of the newmodel is evaluated using BLEU score and compared with [RelGAN](https://openreview.net/pdf?id=rJedV3R5tm).

The project was originally forked from [here](https://github.com/williamSYSU/TextGAN-PyTorch). 

## Selected Experiment Results
### Bias-Variance Analysis 
RebarGAN has lower average bias than GumbelGAN for all the sequence lengths and vocabulary sizes we tested. However, at the same time, GumbelGAN has lower average log variance compared to RebarGAN for all tested values.
<br><br>
<img src = "https://github.com/rtst777/TextGAN/blob/master/image/bias_comparison_table.png" width="640" height="190"> 
<br><br>
<img src = "https://github.com/rtst777/TextGAN/blob/master/image/variance_comparison_table.png" width="600" height="188">
<br><br>
### Rebar-Based RelGAN Evalution 
We trained both RelGAN and ReLbarGAN on the Image COCO dataset with 5 MLE pretraining epochs, batch size of 16. 
<br><br>
<img src = "https://github.com/rtst777/TextGAN/blob/master/image/ReLBarGAN_comparison_plot.png" width="1200" height="260">
<br><br>
More details can be found in this [report](https://drive.google.com/open?id=1l2GRzoFl44E0orNzTpCwKOGCimymJ8EZ).
