# **Traffic Sign Recognition** 

## Writeup


---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./logs/tensorboard_1.JPG "Tensorboard Graph screen"
[image2]: ./logs/tensorboard_Model.jpg "Model Display"
[image3]: ./logs/tensorboard_conv1.jpg "Convolution Layer 1"
[image33]: ./logs/tensorboard_conv1_details.jpg "Convolution Layer 1 Details"
[image4]: ./web_images/30.jpg "Traffic Sign 1"
[image5]: ./web_images/general_caution.jpg "Traffic Sign 2"
[image6]: ./web_images/priority.jpg "Traffic Sign 3"
[image7]: ./web_images/right.jpg "Traffic Sign 4"
[image8]: ./web_images/stop.jpg "Traffic Sign 5"






## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and is found in the zip file for this project

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* The size of training set is                            : 34799
* The size of the validation set is                      : 4410
* The size of test set is                                : 12630
* The shape of a traffic sign image is                   : (34799, 32, 32, 3)
* The number of unique classes/labels in the data set is : 43

#### 2. Include an exploratory visualization of the dataset.

During is an exploratory visualization of the data set, a histogram chart was used to show the training data set's class imbalance.(but no data augmentation used in the final version of the submission) 
Additional, Each of the Sign Class images were grouped together to form a gif image to better visualise the data set.



### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

As a first step, I decided to normalized the images data to range (0 to 1), 
then convert to grayscale for further simplefying the training process and reduce computation required.

Here is an example of a traffic sign image before and after grayscaling.



#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 Gray-Scale image   					| 
| Convolution 1 5x5     | 1x1 stride, Valid padding, outputs 28x28x6 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x6  				|
| Convolution 2 5x5     | 1x1 stride, Valid padding, outputs 10x10x16 	|
| RELU					|												|
| Max pooling 	    	| 2x2 stride,  outputs 5x5x16  					|
| Fully connected 1		| Input = 400, Output = 120 					|
| RELU					|												|
| DropOut 				|	Keep probabilty 0.5							|
| Fully connected 2		| Input = 120, Output = 84						|
| RELU					|												|
| Fully connected 3		| Input = 84, Output = 10						|
| Softmax				|												|
|						|												|
|						|												|

#####Visualization in Tensorboard

######Tensorboard Graph screen
![alt text][image1]

######Tensorboard Model Display
![alt text][image2]

######Tensorboard Model > Convolution Layer 1
![alt text][image3]

######Tensorboard Convolution Layer 1 details
![alt text][image33]

######Additional tensorboard as be run by cloning the repository locally using the file: [for_tensorboad.ipynb](./for_tensorboad.ipynb)

#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used an AWS g2.2x GPU instance with 
- Adam Optimizer, 
- Batch size          : 128, 
- Number of epochs    : 100, 
- Learning Rate       : 0.0005
- Drop out- Keep_prop : 0.5
#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of   : 99.9 
* validation set accuracy of : 96.6
* test set accuracy of       : 93.8

If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen: "Lenet architecheture was tried"
* What were some problems with the initial architecture: "Model seemed to be overfitting, observed due to high different in Training and Validation error"
* How was the architecture adjusted and why was it adjusted: "Dropout layer was added to avoid overfitting"

* Which parameters were tuned? How were they adjusted and why: "Batch size, Number of epochs, Learning Rate, Keep_prop were adjusted with trail and error"
* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model? : "Dropout layer held reduce overfitting the model to training data"

If a well known architecture was chosen:
* What architecture was chosen: 'Lenet architecheture was choose, with addtional dropout layer' 
* Why did you believe it would be relevant to the traffic sign application: 'It was choosen as it has a CNN related architecheture and was well know as a basic starting models for image classification task.'
* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well: 'as seen from above data stats model's accuracy on the training, validation and test set is similar. Hence model is well fit and not overfit nor underfit.'
 

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image4] ![alt text][image5] ![alt text][image6] 
![alt text][image7] ![alt text][image8]

The first image might be difficult to classify because resolution

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image					        |     Prediction	        					| 
|:-----------------------------:|:---------------------------------------------:| 
| Stop Sign      				| Stop sign   									| 
| General caution				| General caution 								|
| Priority road					| Priority road									|
| Speed limit (30km/h)			| Speed limit (30km/h)		 					|
| Dangerous curve to the right	| Vehicles over 3.5 metric tons prohibited		|
|								|												|

The model was able to correctly guess 5 of the 5 traffic signs, which gives an accuracy of 100%. This compares favorably to the accuracy on the test set .

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the cell of the Ipython notebook.

For the first image, the model is relatively sure that this is a stop sign (probability of 0.99), and the image does contain a stop sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 0.999        			| Stop Sign      								| 
| 1.0    				| General caution								|
| 1.0					| Priority road									|
| 0.996	      			| Speed limit (30km/h)							|
| 0.998				    | Dangerous curve to the right					|


