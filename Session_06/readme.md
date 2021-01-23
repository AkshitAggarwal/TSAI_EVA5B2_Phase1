# Session 06: Batch Normalization and Regularization. 

This session was about how Batch Normalization and Ghost batch normalization work in a Deep Neural Network. All types of regularization techniques and how they affect a model were discussed. 

The notebook S6_Assignment.ipynb is an improvement of S5_Assignment, it consists of a lot of modularized code which can be reused. The modules created in this assignments are very dynamic and can be expanded for training multiple models without repeating any lines of code. 

In this assignment five different model iterations with different combinations of Batch Normalizations and Regularization techniques have been compared. The combinations are namely: 

1. L1 + BN
2. L2 + BN
3. L1 + L2 + BN
4. GBN
5. L1 + L2 + GBN

Here L1 is Lasso Regression; L2 is Ridge Regression; BN is Batch Normalization and GBN is Ghost batch normalization. 

## Validation Model Accuracies over all five models. 
![Validation Accuracies](https://github.com/AkshitAggarwal/TSAI_EVA5B2_Phase1/blob/main/Session_06/validation_accuracy.png)


## Validation Model Losses over all five models. 
![Validation Losses](https://github.com/AkshitAggarwal/TSAI_EVA5B2_Phase1/blob/main/Session_06/validation_loss.png)


## 25 Misclassified Images. 
![Misclassified Images](https://github.com/AkshitAggarwal/TSAI_EVA5B2_Phase1/blob/main/Session_06/misclassified.png)
