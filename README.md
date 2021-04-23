# ISCL: Interdependent Self-Cooperative Learning for Unpaired Image Denoising #
This is an official tensorflow implementation of the [paper](https://arxiv.org/abs/2102.09858) by [Kanggeun Lee](https://scholar.google.com/citations?hl=ko&user=OvRs1iwAAAAJ) and [Won-Ki Jeong](https://scholar.google.com/citations?hl=ko&user=bnyKqkwAAAAJ).

IEEE Transactions on Medical Imaging (Under review on Second Special Issue on Machine Learning for Image Reconstruction)
![alt text](figure/Fig_2.png)
- With the advent of advances in self-supervised learning, paired clean-noisy data are no longer required in deep learning-based image denoising. However, existing blind denoising methods still require the assumption with regard to noise characteristics, such as zero-mean noise distribution and pixel-wise noise-signal independence; this hinders wide adaptation of the method in the medical domain. On the other hand, unpaired learning can overcome limitations related to the assumption on noise characteristics, which makes it more feasible for collecting the training data in real-world scenarios. 
In this paper, we propose a novel image denoising scheme, Interdependent Self-Cooperative Learning (ISCL), that leverages unpaired learning by combining cyclic adversarial learning with self-supervised residual learning. Unlike the existing unpaired image denoising methods relying on matching data distributions in different domains, the two architectures in ISCL, designed for different tasks, complement each other and boost the learning process. To assess the performance of the proposed method, we conducted extensive experiments in various biomedical image degradation scenarios, such as noise caused by physical characteristics of electron microscopy (EM) devices (film and charging noise), and structural noise found in low-dose computer tomography (CT). We demonstrate that the image quality of our method is superior to conventional and current state-of-the-art deep learning-based unpaired image denoising methods. 

## Environment Setup ## 


## Citation ##
Please use below bibtex if you cite this paper or use this code.

`@misc{lee2021iscl,
  title={ISCL: Interdependent Self-Cooperative Learning for Unpaired Image Denoising},
  url={https://arxiv.org/abs/2102.09858},
  author={Lee, Kanggeun and Jeong, Won-Ki},
  year={2021}}`