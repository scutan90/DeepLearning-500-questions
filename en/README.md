Translated with Google Translate (corrections welcome)

# 1. Copyright statement
Please respect the author's intellectual property rights, copyright, piracy will be investigated. It is strictly forbidden to forward content without permission!
Please work together to maintain the results of your work and supervise. It is strictly forbidden to forward content without permission!
2018.6.27 TanJiyong

# 2. Overview

This project is to integrate the relevant knowledge of AI and brainstorm ideas to form a comprehensive and comprehensive collection of articles.


# 3. Join and document specifications
1. Seek friends, editors, and writers who are willing to continue to improve; if you are interested in cooperation, improve the book (become a co-author).
2. All contributors who submit content will reflect the contributor's personal information in the text (eg: Daxie-West Lake University)
3, in order to make the content more complete and thoughtful, brainstorming, welcome to Fork the project and participate in the preparation. Please note your name-unit (Dayu-Stanford University) while modifying the MD file (or direct message). Once adopted, the contributor's information will be displayed in the original text, thank you!
4. It is recommended to use the typora-Markdown reader: https://typora.io/  

Setting:
File->Preference
- Syntax Support
  - Inline Math
  - Subscript
  - Superscript
  - Highlight
  - Diagrams

Check these items on

Example:

```markdown
### 3.3.2 How to find the optimal value of the hyperparameter? (Contributor: Daxie - Stanford University)

There are always some difficult hyperparameters when using machine learning algorithms. For example, weight attenuation size, Gaussian kernel width, and so on. The algorithm does not set these parameters, but instead requires you to set their values. The set value has a large effect on the result. Common practices for setting hyperparameters are:

1. Guess and check: Select parameters based on experience or intuition, and iterate over.
2. Grid Search: Let the computer try to evenly distribute a set of values ​​within a certain range.
3. Random search: Let the computer randomly pick a set of values.
4. Bayesian optimization: Using Bayesian optimization of hyperparameters, it is difficult to meet the Bayesian optimization algorithm itself.
5. Perform local optimization with good initial guessing: this is the MITIE method, which uses the BOBYQA algorithm and has a carefully chosen starting point. Since BOBYQA only looks for the nearest local optimal solution, the success of this method depends largely on whether there is a good starting point. In the case of MITIE, we know a good starting point, but this is not a universal solution, because usually you won't know where the good starting point is. On the plus side, this approach is well suited to finding local optimal solutions. I will discuss this later.
6. The latest global optimization method for LIPO. This method has no parameters and is proven to be better than a random search method.
```

# 4. Contributions and Project Overview

Submitted MD version chapter: Please check MarkDown


# 5. More

1. Seek friends, editors, and writers who are willing to continue to improve; if you are interested in cooperation, improve the book (become a co-author).
  All contributors who submit content will reflect the contributor's personal information in the article (Dalong - West Lake University).

2. Contact: Please contact scutjy2015@163.com (the only official email); WeChat Tan:

   (Into the group, after the MD version is added, improved, and submitted, it is easier to enter the group and enjoy sharing knowledge to help others.)

   Into the "Deep Learning 500 Questions" WeChat group please add WeChat Client 1: HQJ199508212176 Client 2: Xuwumin1203 Client 3: tianyuzy

3. Markdown reader recommendation: https://typora.io/ Free and support for mathematical formulas is better.

4. Note that there are now criminals pretending to be promoters, please let the partners know!

5. Next, the MD version will be provided, and everyone will edit it together, so stay tuned! I hope to make suggestions and add changes!


# 6. Contents

**Chapter 1 Mathematical Foundation 1**

1.1 The relationship between scalars, vectors, and tensors 1  
1.2 What is the difference between tensor and matrix? 1  
1.3 Matrix and vector multiplication results 1  
1.4 Vector and matrix norm induction 1  
1.5 How to judge a matrix to be positive? 2  
1.6 Derivative Bias Calculation 3  
What is the difference between 1.7 derivatives and partial derivatives? 3  
1.8 Eigenvalue decomposition and feature vector 3  
1.9 What is the relationship between singular values ​​and eigenvalues? 4  
1.10 Why should machine learning use probabilities? 4  
1.11 What is the difference between a variable and a random variable? 4  
1.12 Common probability distribution? 5  
1.13 Example Understanding Conditional Probability 9  
1.14 What is the difference between joint probability and edge probability? 10  
1.15 Chain Law of Conditional Probability 10  
1.16 Independence and conditional independence 11  
1.17 Summary of Expectations, Variances, Covariances, Correlation Coefficients 11  

**Chapter 2 Fundamentals of Machine Learning 14**

2.1 Various common algorithm illustrations 14  
2.2 Supervised learning, unsupervised learning, semi-supervised learning, weak supervised learning? 15  
2.3 What are the steps for supervised learning? 16  
2.4 Multi-instance learning? 17  
2.5 What is the difference between classification networks and regression? 17  
2.6 What is a neural network? 17  
2.7 Advantages and Disadvantages of Common Classification Algorithms? 18  
2.8 Is the correct rate good for evaluating classification algorithms? 20  
2.9 How to evaluate the classification algorithm? 20  
2.10 What kind of classifier is the best? twenty two  
2.11 The relationship between big data and deep learning 22  
2.12 Understanding Local Optimization and Global Optimization 23  
2.13 Understanding Logistic Regression 24  
2.14 What is the difference between logistic regression and naive Bayes? twenty four  
2.15 Why do you need a cost function? 25  
2.16 Principle of the function of the cost function 25  
2.17 Why is the cost function non-negative? 26  
2.18 Common cost function? 26  
2.19 Why use cross entropy instead of quadratic cost function 28  
2.20 What is a loss function? 28  
2.21 Common loss function 28  
2.22 Why does logistic regression use a logarithmic loss function? 30  
How does the logarithmic loss function measure loss? 31  
2.23 Why do gradients need to be reduced in machine learning? 32  
2.24 What are the disadvantages of the gradient descent method? 32  
2.25 Gradient descent method intuitive understanding? 32  
2.23 What is the description of the gradient descent algorithm? 33  
2.24 How to tune the gradient descent method? 35  
2.25 What is the difference between random gradients and batch gradients? 35  
2.26 Performance Comparison of Various Gradient Descent Methods 37  
2.27 Calculation of the derivative calculation diagram of the graph? 37  
2.28 Summary of Linear Discriminant Analysis (LDA) Thoughts 39  
2.29 Graphical LDA Core Ideas 39  
2.30 Principles of the second class LDA algorithm? 40  
2.30 LDA algorithm flow summary? 41  
2.31 What is the difference between LDA and PCA? 41  
2.32 LDA advantages and disadvantages? 41  
2.33 Summary of Principal Component Analysis (PCA) Thoughts 42  
2.34 Graphical PCA Core Ideas 42  
2.35 PCA algorithm reasoning 43  
2.36 Summary of PCA Algorithm Flow 44  
2.37 Main advantages and disadvantages of PCA algorithm 45  
2.38 Necessity and purpose of dimensionality reduction 45  
2.39 What is the difference between KPCA and PCA? 46  
2.40 Model Evaluation 47  
2.40.1 Common methods for model evaluation? 47  
2.40.2 Empirical error and generalization error 47  
2.40.3 Graphic under-fitting, over-fitting 48  
2.40.4 How to solve over-fitting and under-fitting? 49  
2.40.5 The main role of cross-validation? 50  
2.40.6 k fold cross validation? 50  
2.40.7 Confusion Matrix 50  
2.40.8 Error Rate and Accuracy 51  
2.40.9 Precision and recall rate 51  
2.40.10 ROC and AUC 52  
2.40.11 How to draw ROC curve? 53  
2.40.12 How to calculate TPR, FPR? 54  
2.40.13 How to calculate Auc? 56  
2.40.14 Why use Roc and Auc to evaluate the classifier? 56  
2.40.15 Intuitive understanding of AUC 56  
2.40.16 Cost-sensitive error rate and cost curve 57  
2.40.17 What are the comparison test methods for the model 59  
2.40.18 Deviation and variance 59  
2.40.19 Why use standard deviation? 60  
2.40.20 Point Estimation Thoughts 61  
2.40.21 Point Estimation Goodness Principle? 61  
2.40.22 The connection between point estimation, interval estimation, and central limit theorem? 62  
2.40.23 What causes the category imbalance? 62  
2.40.24 Common Category Unbalance Problem Resolution 62  
2.41 Decision Tree 64  
2.41.1 Basic Principles of Decision Trees 64  
2.41.2 Three elements of the decision tree? 64  
2.41.3 Decision Tree Learning Basic Algorithm 65  
2.41.4 Advantages and Disadvantages of Decision Tree Algorithms 65  
2.40.5 Concept of entropy and understanding 66  
2.40.6 Understanding of Information Gain 66  
2.40.7 The role and strategy of pruning treatment? 67  
2.41 Support Vector Machine 67  
2.41.1 What is a support vector machine 67  
2.25.2 Problems solved by the support vector machine? 68  
2.25.2 Function of the kernel function? 69  
2.25.3 Dual Problem 69  
2.25.4 Understanding Support Vector Regression 69  
2.25.5 Understanding SVM (Nuclear Function) 69  
2.25.6 What are the common kernel functions? 69  
2.25.6 Soft Interval and Regularization 73  
2.25.7 Main features and disadvantages of SVM? 73  
2.26 Bayesian 74  
2.26.1 Graphical Maximum Likelihood Estimate 74  
2.26.2 What is the difference between a naive Bayes classifier and a general Bayesian classifier? 76  
2.26.4 Plain and semi-simple Bayesian classifiers 76  
2.26.5 Three typical structures of Bayesian network 76  
2.26.6 What is the Bayesian error rate 76  
2.26.7 What is the Bayesian optimal error rate? 76  
2.27 EM algorithm to solve problems and implementation process 76  
2.28 Why is there a dimensionality disaster? 78  
2.29 How to avoid dimension disasters 82  
2.30 What is the difference and connection between clustering and dimension reduction? 82  
2.31 Differences between GBDT and random forests 83  
2.32 Comparison of four clustering methods 84  

**Chapter 3 Fundamentals of Deep Learning 88**

3.1 Basic Concepts 88  
3.1.1 Neural network composition? 88  
3.1.2 What are the common model structures of neural networks? 90  
3.1.3 How to choose a deep learning development platform? 92  
3.1.4 Why use deep representation 92  
3.1.5 Why is deep neural network difficult to train? 93  
3.1.6 What is the difference between deep learning and machine learning? 94  
3.2 Network Operations and Calculations 95  
3.2.1 Forward Propagation and Back Propagation? 95  
3.2.2 How to calculate the output of the neural network? 97  
3.2.3 How to calculate the convolutional neural network output value? 98  
3.2.4 How do I calculate the output value of the Pooling layer output value? 101  
3.2.5 Example Understanding Back Propagation 102  
3.3 Superparameters 105  
3.3.1 What is a hyperparameter? 105  
3.3.2 How to find the optimal value of the hyperparameter? 105  
3.3.3 General procedure for hyperparameter search? 106  
3.4 Activation function 106  
3.4.1 Why do I need a nonlinear activation function? 106  
3.4.2 Common Activation Functions and Images 107  
3.4.3 Derivative calculation of common activation functions? 109  
3.4.4 What are the properties of the activation function? 110  
3.4.5 How do I choose an activation function? 110  
3.4.6 Advantages of using the ReLu activation function? 111  
3.4.7 When can I use the linear activation function? 111  
3.4.8 How to understand that Relu (<0) is a nonlinear activation function? 111  
3.4.9 How does the Softmax function be applied to multiple classifications? 112  
3.5 Batch_Size 113  
3.5.1 Why do I need Batch_Size? 113  
3.5.2 Selection of Batch_Size Values ​​114  
3.5.3 What are the benefits of increasing Batch_Size within a reasonable range? 114  
3.5.4 What is the disadvantage of blindly increasing Batch_Size? 114  
3.5.5 What is the impact of Batch_Size on the training effect? 114  
3.6 Normalization 115  
3.6.1 What is the meaning of normalization? 115  
3.6.2 Why Normalize 115  
3.6.3 Why can normalization improve the solution speed? 115  
3.6.4 3D illustration not normalized 116  
3.6.5 What types of normalization? 117  
3.6.6 Local response normalization  
Effect 117  
3.6.7 Understanding the local response normalization formula 117  
3.6.8 What is Batch Normalization 118  
3.6.9 Advantages of the Batch Normalization (BN) Algorithm 119  
3.6.10 Batch normalization (BN) algorithm flow 119  
3.6.11 Batch normalization and group normalization 120  
3.6.12 Weight Normalization and Batch Normalization 120  
3.7 Pre-training and fine tuning 121  
3.7.1 Why can unsupervised pre-training help deep learning? 121  
3.7.2 What is the model fine tuning fine tuning 121  
3.7.3 Is the network parameter updated when fine tuning? 122  
3.7.4 Three states of the fine-tuning model 122  
3.8 Weight Deviation Initialization 122  
3.8.1 All initialized to 0 122  
3.8.2 All initialized to the same value 123  
3.8.3 Initializing to a Small Random Number 124  
3.8.4 Calibrating the variance with 1/sqrt(n) 125  
3.8.5 Sparse Initialization (Sparse Initialazation) 125  
3.8.6 Initialization deviation 125  
3.9 Softmax 126  
3.9.1 Softmax Definition and Function 126  
3.9.2 Softmax Derivation 126  
3.10 Understand the principles and functions of One Hot Encodeing? 126  
3.11 What are the commonly used optimizers? 127  
3.12 Dropout Series Issues 128  
3.12.1 Choice of dropout rate 128  
3.27 Padding Series Issues 128  

**Chapter 4 Classic Network 129**

4.1 LetNet5 129  
4.1.1 Model Structure 129  
4.1.2 Model Structure 129  
4.1.3 Model characteristics 131  
4.2 AlexNet 131  
4.2.1 Model structure 131  
4.2.2 Model Interpretation 131  
4.2.3 Model characteristics 135  
4.3 Visualization ZFNet-Deconvolution 135  
4.3.1 Basic ideas and processes 135  
4.3.2 Convolution and Deconvolution 136  
4.3.3 Convolution Visualization 137  
4.3.4 Comparison of ZFNe and AlexNet 139  
4.4 VGG 140  
4.1.1 Model Structure 140  
4.1.2 Model Features 140  
4.5 Network in Network 141  
4.5.1 Model Structure 141  
4.5.2 Model Innovation Points 141  
4.6 GoogleNet 143  
4.6.1 Model Structure 143  
4.6.2 Inception Structure 145  
4.6.3 Model hierarchy 146  
4.7 Inception Series 148  
4.7.1 Inception v1 148  
4.7.2 Inception v2 150  
4.7.3 Inception v3 153  
4.7.4 Inception V4 155  
4.7.5 Inception-ResNet-v2 157  
4.8 ResNet and its variants 158  
4.8.1 Reviewing ResNet 159  
4.8.2 residual block 160  
4.8.3 ResNet Architecture 162  
4.8.4 Variants of residual blocks 162  
4.8.5 ResNeXt 162  
4.8.6 Densely Connected CNN 164  
4.8.7 ResNet as a combination of small networks 165  
4.8.8 Features of Paths in ResNet 166  
4.9 Why are the current CNN models adjusted on GoogleNet, VGGNet or AlexNet? 167  

**Chapter 5 Convolutional Neural Network (CNN) 170**

5.1 Constitutive layers of convolutional neural networks 170  
5.2 How does convolution detect edge information? 171  
5.2 Several basic definitions of convolution? 174  
5.2.1 Convolution kernel size 174  
5.2.2 Step size of the convolution kernel 174  
5.2.3 Edge Filling 174  
5.2.4 Input and Output Channels 174  
5.3 Convolution network type classification? 174  
5.3.1 Ordinary Convolution 174  
5.3.2 Expansion Convolution 175  
5.3.3 Transposition Convolution 176  
5.3.4 Separable Convolution 177  
5.3 Schematic of 12 different types of 2D convolution? 178  
5.4 What is the difference between 2D convolution and 3D convolution? 181  
5.4.1 2D Convolution 181  
5.4.2 3D Convolution 182  
5.5 What are the pooling methods? 183  
5.5.1 General Pooling 183  
5.5.2 Overlapping Pooling (OverlappingPooling) 184  
5.5.3 Spatial Pyramid Pooling 184  
5.6 1x1 convolution? 186  
5.7 What is the difference between the convolutional layer and the pooled layer? 187  
5.8 The larger the convolution kernel, the better? 189  
5.9 Can each convolution use only one size of convolution kernel? 189  
5.10 How can I reduce the amount of convolutional parameters? 190  
5.11 Convolution operations must consider both channels and zones? 191  
5.12 What are the benefits of using wide convolution? 192  
5.12.1 Narrow Convolution and Wide Convolution 192  
5.12.2 Why use wide convolution? 192  
5.13 Which depth of the convolutional layer output is the same as the number of parts? 192  
5.14 How do I get the depth of the convolutional layer output? 193  
5.15 Is the activation function usually placed after the operation of the convolutional neural network? 194  
5.16 How do you understand that the maximum pooling layer is a little smaller? 194  
5.17 Understanding Image Convolution and Deconvolution 194  
5.17.1 Image Convolution 194  
5.17.2 Image Deconvolution 196  
5.18 Image Size Calculation after Different Convolutions? 198  
5.18.1 Type division 198  
5.18.2 Calculation formula 199  
5.19 Step size, fill size and input and output relationship summary? 199  
5.19.1 No 0 padding, unit step size 200  
5.19.2 Zero fill, unit step size 200  
5.19.3 Not filled, non-unit step size 202  
5.19.4 Zero padding, non-unit step size 202  
5.20 Understanding deconvolution and checkerboard effects 204  
5.20.1 Why does the board phenomenon appear? 204  
5.20.2 What methods can avoid the checkerboard effect? 205  
5.21 CNN main calculation bottleneck? 207  
5.22 CNN parameter experience setting 207  
5.23 Summary of methods for improving generalization ability 208  
5.23.1 Main methods 208  
5.23.2 Experimental proof 208  
5.24 What are the connections and differences between CNN and CLP? 213  
5.24.1 Contact 213  
5.24.2 Differences 213  
5.25 Does CNN highlight commonality? 213  
5.25.1 Local connection 213  
5.25.2 Weight sharing 214  
5.25.3 Pooling Operations 215  
5.26 Similarities and differences between full convolution and Local-Conv 215  
5.27 Example Understanding the Role of Local-Conv 215  
5.28 Brief History of Convolutional Neural Networks 216  

**Chapter 6 Cyclic Neural Network (RNN) 218**

6.1 What is the difference between RNNs and FNNs? 218  
6.2 Typical characteristics of RNNs? 218  
6.3 What can RNNs do? 219  
6.4 Typical applications of RNNs in NLP? 220  
6.5 What are the similarities and differences between RNNs training and traditional ANN training? 220  
6.6 Common RNNs Extensions and Improvement Models 221  
6.6.1 Simple RNNs (SRNs) 221  
6.6.2 Bidirectional RNNs 221  
6.6.3 Deep(Bidirectional) RNNs 222  
6.6.4 Echo State Networks (ESNs) 222  
6.6.5 Gated Recurrent Unit Recurrent Neural Networks 224  
6.6.6 LSTM Netwoorks 224  
6.6.7 Clockwork RNNs (CW-RNNs) 225  

**Chapter 7 Target Detection 228**

7.1 Candidate-based target detector 228  
7.1.1 Sliding Window Detector 228  
7.1.2 Selective Search 229  
7.1.3 R-CNN 230  
7.1.4 Boundary Box Regressor 230  
7.1.5 Fast R-CNN 231  
7.1.6 ROI Pooling 233  
7.1.7 Faster R-CNN 233  
7.1.8 Candidate Area Network 234  
7.1.9 Performance of the R-CNN method 236  
7.2 Area-based full convolutional neural network (R-FCN) 237  
7.3 Single Target Detector 240  
7.3.1 Single detector 241  
7.3.2 Sliding window for prediction 241  
7.3.3 SSD 243  
7.4 YOLO Series 244  
7.4.1 Introduction to YOLOv1 244  
7.4.2 What are the advantages and disadvantages of the YOLOv1 model? 252  
7.4.3 YOLOv2 253  
7.4.4 YOLOv2 Improvement Strategy 254  
7.4.5 Training of YOLOv2 261  
7.4.6 YOLO9000 261  
7.4.7 YOLOv3 263  
7.4.8 YOLOv3 Improvements 264  

** Chapter 8 Image Segmentation 269**

8.1 What are the disadvantages of traditional CNN-based segmentation methods? 269  
8.1 FCN 269  
8.1.1 What has the FCN changed? 269  
8.1.2 FCN network structure? 270  
8.1.3 Example of a full convolution network? 271  
8.1.4 Why is it difficult for CNN to classify pixels? 271  
8.1.5 How do the fully connected and convolved layers transform each other? 272  
8.1.6 Why can the input picture of the FCN be any size? 272  
8.1.7 What are the benefits of reshaping the weight of the fully connected layer into a convolutional layer filter? 273  
8.1.8 Deconvolutional Understanding 275  
8.1.9 Skip structure 276  
8.1.10 Model Training 277  
8.1.11 FCN Disadvantages 280  
8.2 U-Net 280  
8.3 SegNet 282  
8.4 Dilated Convolutions 283  
8.4 RefineNet 285  
8.5 PSPNet 286  
8.6 DeepLab Series 288  
8.6.1 DeepLabv1 288  
8.6.2 DeepLabv2 289  
8.6.3 DeepLabv3 289  
8.6.4 DeepLabv3+ 290  
8.7 Mask-R-CNN 293  
8.7.1 Schematic diagram of the network structure of Mask-RCNN 293  
8.7.2 RCNN pedestrian detection framework 293  
8.7.3 Mask-RCNN Technical Highlights 294  
8.8 Application of CNN in Image Segmentation Based on Weak Supervised Learning 295  
8.8.1 Scribble tag 295  
8.8.2 Image Level Marking 297  
8.8.3 DeepLab+bounding box+image-level labels 298  
8.8.4 Unified framework 299  

**Chapter IX Reinforcement Learning 301**

9.1 Main features of intensive learning? 301  
9.2 Reinforced Learning Application Examples 302  
9.3 Differences between reinforcement learning and supervised learning and unsupervised learning 303  
9.4 What are the main algorithms for reinforcement learning? 305  
9.5 Deep Migration Reinforcement Learning Algorithm 305  
9.6 Hierarchical Depth Reinforcement Learning Algorithm 306  
9.7 Deep Memory Reinforcement Learning Algorithm 306  
9.8 Multi-agent deep reinforcement learning algorithm 307  
9.9 Strong depth  
Summary of learning algorithms 307  

**Chapter 10 Migration Learning 309**

10.1 What is migration learning? 309  
10.2 What is multitasking? 309  
10.3 What is the significance of multitasking? 309  
10.4 What is end-to-end deep learning? 311  
10.5 End-to-end depth learning example? 311  
10.6 What are the challenges of end-to-end deep learning? 311  
10.7 End-to-end deep learning advantages and disadvantages? 312  

**Chapter 13 Optimization Algorithm 314**

13.1 What is the difference between CPU and GPU? 314  
13.2 How to solve the problem of less training samples 315  
13.3 What sample sets are not suitable for deep learning? 315  
13.4 Is it possible to find a better algorithm than the known algorithm? 316  
13.5 What is collinearity and is there a correlation with the fit? 316  
13.6 How is the generalized linear model applied in deep learning? 316  
13.7 Causes the gradient to disappear? 317  
13.8 What are the weight initialization methods? 317  
13.9 In the heuristic optimization algorithm, how to avoid falling into the local optimal solution? 318  
13.10 How to improve the GD method in convex optimization to prevent falling into local optimal solution 319  
13.11 Common loss function? 319  
13.14 How to make feature selection? 321  
13.14.1 How to consider feature selection 321  
13.14.2 Classification of feature selection methods 321  
13.14.3 Feature selection purpose 322  
13.15 Gradient disappearance / Gradient explosion causes, and solutions 322  
13.15.1 Why use gradient update rules? 322  
13.15.2 Does the gradient disappear and the cause of the explosion? 323  
13.15.3 Solutions for Gradient Disappearance and Explosion 324  
13.16 Why does deep learning not use second-order optimization?  
13.17 How to optimize your deep learning system? 326  
13.18 Why set a single numerical evaluation indicator? 326  
13.19 Satisficing and optimizing metrics 327  
13.20 How to divide the training/development/test set 328  
13.21 How to Divide Development/Test Set Size 329  
13.22 When should I change development/test sets and metrics? 329  
13.23 What is the significance of setting the evaluation indicators? 330  
13.24 What is the avoidance of deviation? 331  
13.25 What is the TOP5 error rate? 331  
13.26 What is the human error rate? 332  
13.27 Can avoid the relationship between deviation and several error rates? 332  
13.28 How to choose to avoid deviation and Bayesian error rate? 332  
13.29 How to reduce the variance? 333  
13.30 Best estimate of Bayesian error rate 333  
13.31 How many examples of machine learning over a single human performance? 334  
13.32 How can I improve your model? 334  
13.33 Understanding Error Analysis 335  
13.34 Why is it worth the time to look at the error flag data? 336  
13.35 What is the significance of quickly setting up the initial system? 336  
13.36 Why should I train and test on different divisions? 337  
13.37 How to solve the data mismatch problem? 338  
13.38 Gradient Test Considerations? 340  
13.39 What is the random gradient drop? 341  
13.40 What is the batch gradient drop? 341  
13.41 What is the small batch gradient drop? 341  
13.42 How to configure the mini-batch gradient to drop 342  
13.43 Locally Optimal Problems 343  
13.44 Improving Algorithm Performance Ideas 346  

**Chapter 14 Super Parameter Adjustment 358**

14.1 Debugging Processing 358  
14.2 What are the hyperparameters? 359  
14.3 How do I choose a debug value? 359  
14.4 Choosing the right range for hyperparameters 359  
14.5 How do I search for hyperparameters? 359  

**Chapter 15 Heterogeneous Computing, GPU and Frame Selection Guide 361**


15.1 What is heterogeneous computing? 361  
15.2 What is a GPGPU? 361  
15.3 Introduction to GPU Architecture 361  
  15.3.1 Why use a GPU?  
  15.3.2 What is the core of CUDA?  
  15.3.3 What is the role of the tensor core in the new Turing architecture for deep learning?  
  15.3.4 What is the connection between GPU memory architecture and application performance?  
15.4 CUDA framework  
  15.4.1 Is it difficult to do CUDA programming?  
  15.4.2 cuDNN  
15.5 GPU hardware environment configuration recommendation  
  15.5.1 GPU Main Performance Indicators  
  15.5.2 Purchase Proposal  
15.6 Software Environment Construction  
  15.6.1 Operating System Selection?  
  15.6.2 Is the native installation still using docker?  
  15.6.3 GPU Driver Issues  
15.7 Frame Selection  
  15.7.1 Comparison of mainstream frameworks  
  15.7.2 Framework details  
  15.7.3 Which frameworks are friendly to the deployment environment?  
  15.7.4 How to choose the framework of the mobile platform?  
15.8 Other  
  15.8.1 Configuration of a Multi-GPU Environment  
  15.8.2 Is it possible to distribute training?  
  15.8.3 Can I train or deploy a model in a SPARK environment?  
  15.8.4 How to further optimize performance?  
  15.8.5 What is the difference between TPU and GPU?  
  15.8.6 What is the impact of future quantum computing on AI technology such as deep learning?  

**References 366**

Hey you look like a cool developer.  
Translate it to english.  
