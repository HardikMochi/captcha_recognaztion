<h1 align='center'>CAPTCHA Recognition</h1>


Try it out byuploading the captcha image and it will predictt captcha present inside image.

<table><tr><td><img src='https://github.com/HardikMochi/captcha_recognaztion/blob/main/images/result1.PNG' width=500></td><td><img src='https://github.com/HardikMochi/captcha_recognaztion/blob/main/images/2.PNG' width=500></td></tr></table>



Captcha is computer generating text images used to distinguish interactions given by humans or machines. Normally, a captcha image consists of a fixed number of characters (e.g. digit, letter). These characters are not only distorted, scaled into multiple different sizes but also can be overlapped and crossed by multiple random lines.
<br>I build deep learning models solving this captcha recognition .model consists of convolutional layers to learn visual features. These features are then fed to GRU to compute the final captcha prediction.
## Table of Contents
<details open>
<summary>Show/Hide</summary>
<br>

1. [ File Descriptions ](#File_Description)
2. [ Technologies Used ](#Technologies_Used)    
3. [ Structure ](#Structure)
4. [ Executive Summary ](#Executive_Summary)
   * [ 1. Webscraping, Early EDA, and Cleaning ](#Webscraping_Early_EDA_and_Cleaning)
       * [ Webscraping ](#Webscraping)
       * [ Early EDA and Cleaning](#Early_EDA_and_Cleaning)
   * [ 2. Further EDA and Preprocessing ](#Further_EDA_and_Preprocessing) 
   * [ 3. Modelling and Hyperparameter Tuning ](#Modelling)
   * [ 4. Evaluation ](#Evaluation)
       * [ Future Improvements ](#Future_Improvements)
   * [ 5. Neural Network Modelling ](#Neural_Network_Modelling)
   * [ 6. Revaluation and Deployment ](#Revaluation)
</details>

## File Descriptions
<details>
<a name="File_Description"></a>
<summary>Show/Hide</summary>
<br>
    
* <strong>[ captcha_images_v2 ](https://github.com/awesomeahi95/Hotel_Review_NLP/tree/master/Data)</strong>: folder containing all captcha images
* <strong>[ Images ](https://github.com/awesomeahi95/Hotel_Review_NLP/tree/master/Images)</strong>: folder containing images used for README 
* <strong>[ Model ](https://github.com/awesomeahi95/Hotel_Review_NLP/tree/master/Models)</strong>: folder containing trained models
* <strong>[ Data ](https://github.com/awesomeahi95/Hotel_Review_NLP/tree/master/Models)</strong>: folder containing data which is used in app.py
* <strong>[ static ](https://github.com/awesomeahi95/Hotel_Review_NLP/tree/master/Models)</strong>: folder containing css and javascript file
* <strong>[ templates ](https://github.com/awesomeahi95/Hotel_Review_NLP/blob/master/1.Webscraping_Early_EDA_and_Cleaning.ipynb)</strong>: folder containing html files
* <strong>[ config.py ](https://github.com/awesomeahi95/Hotel_Review_NLP/blob/master/2.Further_EDA_and_Preprocessing.ipynb)</strong>: This file containing  config perameter
* <strong>[dataset.py ](https://github.com/awesomeahi95/Hotel_Review_NLP/blob/master/3.Modelling_and_Hyperparameter_Tuning.ipynb)</strong>: This file is used to build dataset for project
* <strong>[engine.py ](https://github.com/awesomeahi95/Hotel_Review_NLP/blob/master/4.Evaluation.ipynb)</strong>: this file contain training and evaluation function
* <strong>[model.py ](https://github.com/awesomeahi95/Hotel_Review_NLP/blob/master/5.Neural_Network_Modelling.ipynb)</strong>: this file is used to build model 
* <strong>[train.py ](https://github.com/awesomeahi95/Hotel_Review_NLP/blob/master/6.Revaluation_and_Deployment.ipynb)</strong>: this file contain script to train the model
* <strong>[ CAPTCHA_Recognition ](https://github.com/awesomeahi95/Hotel_Review_NLP/blob/master/Classification.py)</strong>: contains classes with classifcation methods
* <strong>[ app.py ](https://github.com/awesomeahi95/Hotel_Review_NLP/blob/master/Ensemble.py)</strong>: this file contain script to create web application

</details>

## Tecnologies Used:
<details>
<a name="Technologies_Used"></a>
<summary>Show/Hide</summary>
<br>
    
* <strong>Python</strong>
* <strong>Pandas</strong>
* <strong>Numpy</strong>
* <strong>Matplotlib</strong>
* <strong>Pytorch</strong>
* <strong>Flask</strong>
* <strong>albumentations</strong>
</details>


<a name="Executive_Summary"></a>
## Executive Summary


<a name="Dataset Description"></a>
### Dataset Description :
<details open>
<summary>Show/Hide</summary>
<br>

The dataset used for this project consists of 1070 .png images of text based CAPTCHA. The dataset has been taken from https://www.researchgate.net/publication/248380891_CAPTCHA_dataset. Each dataset image is of 5 character set and the character set is defined as all English small letters and digits from 0 to 9. Hence a total of 36 characters are present in the character set.
<br>The CAPTCHA images consist of noise in the form of lines and blurriness. The characters in images are also not straight and clear.

<h5>Images</h5>
<table><tr><td><img src='https://github.com/HardikMochi/captcha_recognaztion/blob/main/captcha_images_v2/23mdg.png' ></td><td><img src='https://github.com/HardikMochi/captcha_recognaztion/blob/main/captcha_images_v2/22d5n.png' ></td></tr></table>
</detail>

<a name="Data Preprocessing"></a>
### Data Preprocessing
<details open>
<summary>Show/Hide</summary>
<br>
    The images in the dataset have a filename same as the CAPTCHA present in the image.The images are first pre-processed by reading image and  Each  image is then scaled and reshaped to the size: height- 50, width-300 and the number of channels as 3.
   than we normlize the image and we transfrom it to Pytorch tensor.

</details>

<a name="Model Development"></a>
### Model Development:
<details open>
<summary>Show/Hide</summary>
<br>

The model developed for this CAPTCHA dataset uses Convolutional Neural Network and GRU. It comprising the input layer, convolutional layers, max pooling layers, dense layers, flatten layers dropout layers and GRU layers. 

<h5 align="center">Architecture Model</h5>
<p align="center">
  <img src="https://github.com/awesomeahi95/Hotel_Review_NLP/blob/master/Images/all_models.png" width=600>
</p>

Initially, I thought the validation accuracy was low for most of the models I created, but when considering these models were attempting to classify for 5 different classes, 0.45 and greater seems very reasonable (where 0.2 = randomly guessing correctly).

I have saved all the models using the pickle library's dump function and stored them in the Models folder.
</details>

<a name="Evaluation"></a>
### Evaluation
<details open>
<summary>Show/Hide</summary>
<br>

I focused on 3 factors of defining a good model:

1. Good Validation Accuracy
2. Good Training Accuracy
3. Small Difference between Training and Validation Accuracy

I chose the Stacking ensemble model ( (Adaboost with log_reg_2) stacked with log_reg_2 ) as my best model, because it has the highest validation accuracy with only around 3.5% drop from train to validation in accuracy. I wanted to minimise overfitting and make the model as reusable as possible. Stacking achieved a reasonable training accuracy as well, although it did not reach the level of some of the other ensemble techniques.

I next tested the best model with the earlier saved test data. The model managed to get a high test accuracy, similar to the validation data from the model training stage. This is very good, proving that prioritising a high validation score, and minimising the difference between train and validation accuracy, has helped it classify new review texts very well.

<h5 align="center">Test Results</h5>
<p align="center">
  <img src="https://github.com/awesomeahi95/Hotel_Review_NLP/blob/master/Images/test_results.png" width=600>
</p>

Looking at the precision, recall, and f1 score, I also noticed the scores were higher around scores of 1 and 5, lower for 2, 3, and 4. This shows that the models performs well on more extreme opinions on reviews than mixed opinions.

Looking into different metrics and deeper into my best model; Stacking, I learnt that most the False Postives came from close misses (e.g. predicting a score of 4 for a true score of 5). This is best shown by these two confusion matrixes (validation and test). 

<h5 align="center">Confusion Matrix for Validation and Test Data Predictions ( Validation (Left) and Test (Right) )</h5>
<table><tr><td><img src='https://github.com/awesomeahi95/Hotel_Review_NLP/blob/master/Images/validation_conf_matrix.png' width=500></td><td><img src='https://github.com/awesomeahi95/Hotel_Review_NLP/blob/master/Images/test_conf_matrix.png' width=500></td></tr></table>

The adjacent squares of the diagonal going across the confusion matrix, shows that the model's second highest prediction for a given class (review score) is always a review score that is +-1 the true score.
Very few reviews that have a score of 5, have been predicted to have a score of 1 or 2. This is very relieving to know, the majority of the error for the model, is no different to the error a human may make classifying a review to a score with a scale of 1-5.

- most errors were near misses (e.g. 5 predicted as 4)
- extreme scores (1 and 5) were relatively accurate
- comparable to human prediction
- reusable and consistent


Given the classifcation problem is 5 way multi-class one and the adjacent classes can have overlap in the english language even to humans, this model I have created can be deployed.

Applying this model will address the problem of not having a full understanding of public opinion of our hotel. We can apply this to new sources for opinions on our hotel and yield more feedback then we did had before.

<a name="Future_Improvements"></a>
#### Future Improvements

- Model using neural networks - see if better accuracy can be achieved
- Create a working application to test new reviews written by people
- Try a different pre-processing approach and see if model performances change
- Bring in new sources of data to see if there are significant differences on frequent words used

</details>

<a name="Neural_Network_Modelling"></a>
### Neural Network Modelling:
<details open>
<summary>Show/Hide</summary>
<br>
    
I experimented with different classifcation and ensemble methods to help classify hotel review scores. Some performed well, but there was definitely room for improvement, so I wanted to explore a deep learning approach. 
    
    
<h5 align="center">Neural Network Architecture</h5>
<p align="center">
  <img src="https://github.com/awesomeahi95/Hotel_Review_NLP/blob/master/Images/NN_architecture.png" width=600>
</p>
    
* Input Layer: 17317 Nodes (one for each word in training data + 4 extra for padding, unknown words, start of review, and unused words)
* Embedding Layer: takes 17317 unique items and maps them into a 16 dimensional vector space
* Global Average 1D Pooling Layer: scales down 16 dimensional layer
* Dense Hidden Layer: 16 Nodes (using relu activation function)
* Dense Output Layer: 5 nodes for each score (using sigmoid activation function)
    
</details>

<a name="Revaluation"></a>
### Revaluation and Deployment:
<details open>
<summary>Show/Hide</summary>
<br>

I tested the neural network model using the test data and achieved an accuracy of <strong>0.5710</strong> which is better than the stacking model accuracy of <strong>0.5077</strong>, by <strong>over 5%</strong>. 
    
I wanted to look at the confusion matrix, as this gives a better idea of how the model is performing over all 5 classes.
    
<h5 align="center">Neural Network Model Test Confusion Matrix</h5>
<p align="center">
  <img src="https://github.com/awesomeahi95/Hotel_Review_NLP/blob/master/Images/nn_conf_matrix.png" width=600>
</p>
    
The error is more contained within adjacent scores with the neural network model. Almost zero confusion between extreme scores 1 and 5, and minimal confusion with scores 2 and 4. Although a score of 3 can be harder to predict, there is definitely an improvement from the Stacking model. Around 97% of the time the model predicts at least the adjacent score to the actual score.

#### Deployment and Application
    
After seeing the improvements from the Stacking model, I was more confident about deploying the model for actionable use.
    
I planned on future improvements being the addition of the neural network model and then creating an application for the model, so as a next step I decided to make a working application to test out new reviews using streamlit. I have deployed the app using Heroku: https://hilton-hotel-app.herokuapp.com/. 
    
Using this model, we will learn more about our new and old customers, then we can improve Hilton Hotel's guest satisfaction, and as a result increase customer retention and bring in new travelers.
    
#### Future Development
    
* Create a webscraper spider for twitter, reddit, etc for further model assessment
    
</details>
