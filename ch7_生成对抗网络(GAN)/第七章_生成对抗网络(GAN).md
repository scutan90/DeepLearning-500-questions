# 第七章 生成对抗网络(GAN)

## GAN生成概率的本质是什么？
GAN的形式是：两个网络，G（Generator）和D（Discriminator）。Generator是一个生成图片的网络，它接收一个随机的噪声z，记做G(z)。Discriminator是一个判别网络，判别一张图片是不是“真实的”。它的输入是x，x代表一张图片，输出D（x）代表x为真实图片的概率，如果为1，就代表100%是真实的图片，而输出为0，就代表不可能是真实的图片。

GAN*生成*能力是学习*分布*，引入的latent variable的noise使习得的概率分布进行偏移。因此在训练GAN的时候，latent variable**不能**引入均匀分布（uniform distribution)，因为均匀分布的数据的引入并不会改变概率分布。

## VAE与GAN有什么不同？
1. VAE可以直接用在离散型数据。
2. VAE整个训练流程只靠一个假设的loss函数和KL Divergence逼近真实分布。GAN没有假设单个loss函数, 而是让判别器D和生成器G互相博弈，以期得到Nash Equilibrium。


## 有哪些优秀的GAN？

### DCGAN

### WGAN/WGAN-GP

WGAN及其延伸是被讨论的最多的部分，原文连发两文，第一篇(Towards principled methods for training generative adversarial networks)非常solid的提了一堆的数学，一作Arjovsky克朗所的数学能力果然一个打十几个。后来给了第二篇Wasserstein GAN，可以说直接给结果了，和第一篇相比，第二篇更加好接受。

然而Wasserstein Divergence真正牛逼的地方在于，几乎对所有的GAN，Wasserstein Divergence可以直接秒掉KL Divergence。那么这个时候就有个问题呼之欲出了：

**KL/JS Divergence为什么不好用？Wasserstein Divergence牛逼在哪里？**

**KL Divergence**是两个概率分布P和Q差别的**非对称性**的度量。KL Divergence是用来度量使用基于Q的编码来编码来自P的样本平均所需的额外的位元数。 而**JS Divergence**是KL Divergence的升级版，解决的是**对称性**的问题。即：JS Divergence是对称的。

KL Divergence:
$$D_{KL}(P||Q)=-\sum_{x\in X}P(x) log\frac{1}{P(x)}+\sum_{x\in X}p(x)log\frac{1}{Q(x)}=\sum_{x\in X}p(x)log\frac{P(x)}{Q(x)}$$

JS Divergence:

$$JS(P_1||P_2)=\frac{1}{2}KL(P_1||\frac{P_1+P_2}{2})$$

**Wasserstein Divergence**：如果两个分配P,Q离得很远，完全没有重叠的时候，KL Divergence毫无意义，而此时JS Divergence值是一个常数。这使得在这个时候，梯度直接消失了。

**WGAN从结果上看，对GAN的改进有哪些？**

1. 判别器最后一层去掉sigmoid
2. 生成器和判别器的loss不取log
3. 对更新后的权重强制截断到一定范围内，比如[-0.01，0.01]，以满足lipschitz连续性条件。
4. 论文中也推荐使用SGD，RMSprop等优化器，不要基于使用动量的优化算法，比如adam。

然而，就实际而言，最优的选择其实应该是

**如何理解Wass距离？**

### condition GAN

### InfoGAN
通过最大化互信息（c，c’）来生成同类别的样本。

$$L^{infoGAN}_{D,Q}=L^{GAN}_D-\lambda L_1(c,c')$$
$$L^{infoGAN}_{G}=L^{GAN}_G-\lambda L_1(c,c')$$

### CycleGAN

**CycleGAN与DualGAN之间的区别**

### StarGAN
目前Image-to-Image Translation做的最好的GAN。
## GAN训练有什么难点？
由于GAN的收敛要求**两个网络（D&G）同时达到一个均衡**，

## GAN与强化学习中的AC网络有何区别？
强化学习中的AC网络也是Dual Network，似乎从某个角度上理解可以为一个GAN。但是GAN本身
## GAN的可创新的点
GAN是一种半监督学习模型，对训练集不需要太多有标签的数据。

## 如何训练GAN？
判别器D在GAN训练中是比生成器G更强的网络

Instance Norm比Batch Norm的效果要更好。

## GAN如何解决NLP问题

GAN只适用于连续型数据的生成，对于离散型数据效果不佳，因此假如NLP方法直接应用的是character-wise的方案，Gradient based的GAN是无法将梯度Back propagation（BP）给生成网络的，因此从训练结果上看，GAN中G的表现长期被D压着打。
## Reference
### DCGAN部分：

### WGAN部分：
* Arjovsky, M., & Bottou, L. (2017). Towards principled methods for training generative adversarial networks. arXiv preprint arXiv:1701.04862.
* Arjovsky, M., Chintala, S., & Bottou, L. (2017). Wasserstein gan. arXiv preprint arXiv:1701.07875.
* Nowozin, S., Cseke, B., & Tomioka, R. (2016). f-gan: Training generative neural samplers using variational divergence minimization. In Advances in Neural Information Processing Systems (pp. 271-279).
* Wu, J., Huang, Z., Thoma, J., Acharya, D., & Van Gool, L. (2018, September). Wasserstein Divergence for GANs. In Proceedings of the European Conference on Computer Vision (ECCV) (pp. 653-668).

### CycleGAN
Zhu, J. Y., Park, T., Isola, P., & Efros, A. A. (2017). Unpaired image-to-image translation using cycle-consistent adversarial networks. arXiv preprint.
