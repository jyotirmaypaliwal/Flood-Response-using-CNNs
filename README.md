# Flood-Detection-using-CNNs

### Here I have come up with a **flood relief** solution with the help of modern day technologies such as **artificial intelligence** and **deep learning**. Basically what we do is, we first accumulate flood images data from various sources and then train an image classifier on them to detect wether an image contains a human or not.This forms the first half of our solution. In the second half, we integrate Twitter APIs to fetch flood images and passs them through our classifier. Once we detect that an image contain humans, we store the images name and its metadata in a csv file. This whole process enables us to automate detection of people stuck in floods through the use of Artificial Intelligence.

## Architecture

![architecture](https://github.com/jyotirmaypaliwal/Flood-Detection-using-CNNs/blob/main/architecture.png)

### Our image classification task was divided into 3 sub tasks which are - 

1. **Gathering images** - This task involved gathering images through various sources. In our project we collected images primarily from a European flood images dataset and secondarily from social media images.                
     * Our total data size was of around 2000 images among which 700 images contained humans.

2. **Data processing** - This task involved selection and processing of images and is further sub divided into two sub tasks which are:
    * Manual filtering of images to remove unrelated images.
    * Manual classification of images into two classes - 
      a) images containing humans
      b) images not containing humans
    * We also converted images to the size of 299 x 299 as the model we used to train our classifier only accepts images of that size. 
      
 3. **Image Classification** - This is the step where we get to actual model training and loading part. In this part we use CNN (Convolution Neural Networks) to train our model to recognize images containing humans and images not containing humans. In our case, we used a popular image classification model knows as **Inception v3** to train our image classifier. 

## Inception V3

![inception-v3](https://user-images.githubusercontent.com/27720480/136644979-7acad130-2bd9-4a28-a5bd-94026f4fd4e2.jpg)
Inception-v3 is a convolutional neural network architecture from the Inception family that makes several improvements including using Label Smoothing, Factorized 7 x 7 convolutions, and the use of an auxiliary classifer to propagate label information lower down the network (along with the use of batch normalization for layers in the sidehead).


Inception-v3 - https://paperswithcode.com/method/inception-v3


## Result
### We were able to acheive an accuracy of 80% on our testing data.

## Extending and refining our project
#### We can further extend and refine our project through various ways, some of which are - 
1. Accumulating additional images and using transfer learning to refine our model.
2. Integrating it with the Twitter API to create an whole flood relief system.


![twitter](https://github.com/jyotirmaypaliwal/Flood-Detection-using-CNNs/blob/main/Blank%20diagram.png)

Images are accumulated through the Twitter APIs and are passed through our classifier. The classifier classifies them and stores their metadata inside of a csv file. 

## Conclusion
Deep Learning techniques are now in a comfortable and safe position to be leveraged by various governments in disaster relief response and their ability to learn new things has immense potential. AI techniques should be incorporated into disaster relief and response as they are reliable and fast along with taking some load off disaster relief personnels.
