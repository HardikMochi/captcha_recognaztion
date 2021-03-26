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
   * [ 1. Dataset Description](#Dataset Description)
   * [ 2. Data Preprocessing ](#Data Preprocessing) 
   * [ 3. Model Development ](#Model Development)
   * [ 4. Results ](#Results)
   * [ 5. Prediction ](#Prediction)
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
<br>Thus, the model uses an image of dimension (3,75, 300) as input and gets output  having dimension 20.
<h5 align="center">Architecture Model</h5>
<p align="center">
  <img src="https://github.com/HardikMochi/captcha_recognaztion/blob/main/images/7.PNG" width=600>
</p>
The dataset consists of 1070 sample images out of which 970 images have been used for training and the remaining images for testing purpose. Further, for training the model, a validation split of 0.1 is used which splits the training set such that 90% of the training data is used for training and the remaining 10% for testing. The batch size used is 8
and the number of epochs used is 70.
</details>

<a name="Results"></a>
### Results
<details open>
<summary>Show/Hide</summary>
<br>

After training the above model for 70 epochs, the following graph was obtained for loss with respect to the number of epochs as shown in figure. We see that as the number of epoch’s increases, the loss decreases exponentially. The loss obtained on training set is 0.0388 while the loss on test set is 0.114.
<h5 align="center">Test Results</h5>
<p align="center">
  <img src="https://github.com/HardikMochi/captcha_recognaztion/blob/main/images/10.PNG" width=600>
</p>
</details>

<a name="Prediction"></a>
### Prediction:
<details open>
<summary>Show/Hide</summary>
<br>

Prediction on the test images. 
   
<p align="center">
  <img src="https://github.com/HardikMochi/captcha_recognaztion/blob/main/images/9.PNG" width=600>
</p>
as you seen in above picture model is correctly predict the captcha.
</details>

<a name="Application "></a>
### Deployme:
<details open>
<summary>Show/Hide</summary>
<br>

i created a web application which predict the captcha present in the image. 
    
<p align="center">
  <img src="https://github.com/HardikMochi/captcha_recognaztion/blob/main/images/3.PNG">
</p>
    
    
</details>
