# Session-04: Architectural Ingredients. 

In this session we learned about different Architectural Ingredients, methods and techniques which are used to enhance our Model's performance so that we can achieve a higher accuracy rate and avoid the following problems:
  1. Model overfitting: When a model gets too familiar with the input data and starts giving unexpected levels of accuracy but in reality doesn't perform          so well on test or real life data. 
  2. Too many model parameters: Every operation and layer that we add to our model adds aditional number of parameters, which basically mean that our model will have to peform addition calculations to get the results. Some steps such as **Fully Connected Layers** add too many parameters in comparison to other methods. We try to make a model that has just enough parameters to get the desired(or Best possible) accuracy. 
  
  ### Methods that got introduced in this session:
    1. MaxPooling
    2. Batch Normalization
    3. Dropout
    4. Learning Rate
    5. Batch Size
    
   The methods mentioned above can be used to achieve a better accuracy, there is a lot of experimentation involved in this process, because once we've used the best possible architecture for our model these are the only parameters that we can tweak in order to achieve better results. The **tweaking** of these parameters is what is known as **Hyperparameter tuning**. 
   
   
## Aim of the assignement:
We are using the MNIST Hand-written digits dataset, our aim is to design a prediction model using a **Deep Neural Network** to achieve State-of-the-art accuracy with the model. With the current model which consists of 7 Convolution Layers and 2 Maxpooling layers the model reaches a test accuracy of 99%. Our goal in this assignment is to use the above methods and Hyperparameter Tuning to reach an accuracy of **99.4%** while keeping the total parameters under 20K. 

> In this document I will try to document all the experiments that I'm doing in order to achieve this accuracy and also, the final method(or experiement) which allows us to achieve this goal will be explained in detail, along with how it has affected our model.
   
