
# Session-05: Coding Drill Down

In this session we learned about the process through which all developers should go through in order to systematically make a good model. It is true that experimenting is the key with Deep Learning models to achieve good accuracies, but doing those experiments with no particular reason, or notes or motive other than increasing accuracy doesn't lead to much good. We looked at 10 different notebooks, each notebook was a step, an enhancement of the last step which eventually lead to a good model with a high accuracy and a reduced number of parameters. 

The aim of this assignment is to go through similar steps as taken in the lesson and achieve a high accuracy under 20k parameters, and document those steps as we move forward. The things that we need to take care of: 
  1. Reduce parameters under 20k.
  2. Test accuracy should be consistently over 99.40%
  3. Make a clean model pipeline to achieve this. 
  
### I will try to achieve the goal of this assignment in 4 progressive steps and document my experiments in 4 different notebooks and explain clearly the motive of each step taken here. 

## Experiment 1

### Target:
To create a simple model architecture under 20k parameters with no data augmentation or regularization. 
### Results:
Total parameters: 19,552 |  Maximum Train Accuracy: 99.07 |  Maximum Test Accuracy: 98.99
### Analysis:
-  Train and test accuracies are quite close. Model was underfitting most of the time while training except at the last. 
-  Still to make the model architecture less paramater heavy. 

===================================================================================================

## Experiment 2

### Target: 
To make a model will less than 10k parameters.
### Results:
Total parameters: 9,032 |  Maximum Train Accuracy: 98.98 |  Maximum Test Accuracy: 98.85
### Analysis:
-  Train and test accuracies are very close. 
-  Model is finally overfitting. Which means I can apply regularization. 
-  Model parameters are reduced under 10k, still model performs equally well. 

===================================================================================================


## Experiment 3

### Target: 
Add regularization to model(Batch Normalization and Dropout). 
### Results:
Total parameters: 9,192 |  Maximum Train Accuracy: 99.19 |  Maximum Test Accuracy: 99.29
### Analysis:
-  Without applying droupt the train/test accuracies were: 99.61/99.37
-  After applying batch normalization(Without dropout) the accuracy increased and the model was consistently overfitting by 0.2-0.3%
-  Accuracy slightly decreased after adding dropout but it solved the problem of overfitting. The model after adding dropout overfits by 0.07-0.08%
-  It looks like this is the best this model can do under 20 epochs and to increase the accuracy other measures needs to be taken. 

===================================================================================================


## Experiment 4

### Target: 
Experiment with Image Augmentations and Learning Rates. 
### Results:
Total parameters: 9,752 |  Maximum Train Accuracy: 98.86 |  Maximum Test Accuracy: 99.39
### Analysis:
-  Adding image augmentation increased testing accuracy slightly. 
-  LR Schedular did the real trick, with a weight decay of 0.75 times after 3 steps. 
-  Loss was going down steadily and model can perform even better after training for a few more epochs. 

===================================================================================================
