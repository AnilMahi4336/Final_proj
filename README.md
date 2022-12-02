TOPIC: SEMANTIC SEGMENTATION OF REAL TIME DATA

The goal of semantic image segmentation is to label each pixel of an image with a corresponding class of what is being represented. 
We're predicting for every pixel in the image, this task is commonly referred to as dense prediction.

1. Originality and Innovation:
	In terms of Innovation, we are presenting a new architecture that is trades between the speed and the performance. There have been many models that work with better performance but requires lot of processing power thereby taking lot of powerful processors for  running the program, Even there are basic models that run faster but giving less accuracy for real time applications, our models works between these models where we could obtain a acceptable performance and the speed required.
	In terms of Originality the architecture is a blend of different models that could speed up the process and mixing these different models to form a intermediate model.

2. Significance and Implication:
	It has applications that could be applicable to many other types of applications like lane detection for autonomous vehicles, for lane detection of maps so we could navigate without the Gps, Which is available only with internet connection, also it is useful when we need to detect a human in the photo and segment him to get different background.
	
3.Appropriateness of Research Methodology :
	•	This approach is used as it is processed in parallel so to increase the speed of processing where larger kernels,smaller Kernels are trained in parallel.
	•	This process does give feature vectors that are from 2 different models that could be more refined than the feature vectors from one model.
	•	Mobilenets are used as spatial tools in order to make sure that models is fast and could be used to run on even simpler devices like mobiles.
	•	Mobilenets and Resnets used are trained initially on imageNets so in order to make sure the training in required on lesser no of components like the roads and column poles and other 4 to 5 components and rest others are trained well.


		
ARCHITECTURE DESIGNED:
	•	In this architecture data is cropped to 640 * 640 shape as the input.
	•	MobileNets is used as the Spatial Path 
	•	Resnets 101 trained on the Imagenet is used as the Contextual Path
	•	Accuracy expeded is 75%
	•	Time taken of output is nearly 1.5 iterations/Sec

SPATIAL PATH
	•	It models the global features of the image .
	•	It is important to predict the detailed output
CONTEXT PATH
	•	It models the features at different scales to get better pixel prediction.
	•	Important for High Level features.
ATTENTION REFINEMENT MODULE
	•	It models the features extracted from each layer from context path and refines them.
FEATURE FUSION MODULE
	•	It models the features extracted from spatial and contextual paths and then refine them.
LOSS CALCULATION
	•	Loss is calculated in 3 different scales and log loss is used to calculated from these results. 

4.Quality of data and findings:
	The Cambridge-driving Labeled Video Database (CamVid) is the first collection of object class semantic labels. The database provides ground truth labels that associate each pixel with one of 32 semantic classes.

The database addresses the need for experimental data to quantitatively evaluate emerging algorithms. Data was captured from the perspective of a driving automobile.
