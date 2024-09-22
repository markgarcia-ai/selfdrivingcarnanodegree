# PROJECT RUBRIC: Model Training and Evaluation

## Criteria and Submission Requirements

### 1. Test at Least Two Pretrained Models (Other than EfficientNet)
- **Requirement**: Test at least two models other than EfficientNet.
- **Submission**: Update and submit the `pipeline.config` file and notebooks associated with all the pretrained models.

### 2. Choosing the Best Model for Deployment
- **Requirement**: Write a brief summary of your experiments and suggest the best model for this problem. Include the accuracy (mAP) values of the models you tried.
- **Discussion Points**:
  - How does the validation loss compare to the training loss?
  - Did you expect such behavior from the losses/metrics?
  - What can you do to improve the performance of the tested models further?

# Model Deployment

## Criteria and Submission Requirements

### 1. Deploy the Best Model and Run Inference
- **Requirement**: Deploy the best model in AWS by completing and running `2_deploy_model.ipynb`.
- **Submission**: Should be able to generate an output video with vehicle, pedestrian, and cyclist detections using the deployed model.





# HOW TO RUN THESE SCRIPTS
You need to create a virtual environment in CONDA and use next libraries. <br>
1-  conda create --name myenv python=3.11 #Create virtual env <br>
2 - conde activate myenv #Activate conda environment <br>
3-  conda install -c conda-forge opencv #Install OpenCV libraru <br> 
4-  conda install -c conda-forge imageio <br>
5-  conda install pytorch torchvision torchaudio cpuonly -c pytorch #Install pytorch CPU version <br>


# Project folders

**01_Udacity_Project_tensorflow**: Files from udacity <br>
**02_OPENCV**: Examples of moldes using open CV <br>
**03_PYTORCH**: Examples using Pytorch <br>

## Using PYTORCH
PyTorch is a deep learning framework. It's an open-source machine learning library primarily designed for building and training deep neural networks. PyTorch provides a flexible and intuitive interface that facilitates the creation and experimentation of complex deep learning models.

## Diferences with OpenCV
OpenCV is the go-to library for traditional computer vision and image processing tasks such as Image and Video Processing, Object Detection and Recognition, Feature Extraction and Matching, Augmented Reality and Robotics. Depending on your project requirements, you might choose one over the other or even use them together, leveraging OpenCV for pre-processing and PyTorch for nodel training and inference.

## MODELS TO DEPLOY

**Faster R-CNN (Region-based Convolutional Neural Network)** -> It's designed to identify and localize objects within an image <br>
**ResNet (Residual Networks)** -> It detects upto 80 objects. <br>
**VGG (Visual Geometry Group)** -> It's more for image clasification than object detection with 1000 different categories. <br>
**DenseNet (Densely Connected Convolutional Networks):** -> It's designed for image classification tasks with 1000 categories <br>


# MODELS PERFORMANCE
<table>
  <tr>
    <td align="center"><img src="https://github.com/markgarcia-ai/selfdrivingcarnanodegree/blob/main/01_ObjectDetection/results/video.gif" alt="Image 1" width="400" /> Original Video</td>
    <td align="center"><img src="https://github.com/markgarcia-ai/selfdrivingcarnanodegree/blob/main/01_ObjectDetection/results/Fasterrcnn.gif" alt="Image 1" width="400" /> Faster R-CNN</td>    
    <td align="center"><img src="https://github.com/marcjesus/udacity/blob/main/01_ObjectDetection/OPENCV_output_gif.gif" alt="Image 2" width="400" /> ResNet (Residual Networks)</td>
    <td align="center"><img src="https://github.com/marcjesus/udacity/blob/main/01_ObjectDetection/output.gif" alt="Image 1" width="400" /> VGG (Visual Geometry Group) </td>
    <td align="center"><img src="https://github.com/marcjesus/udacity/blob/main/01_ObjectDetection/OPENCV_output_gif.gif" alt="Image 2" width="400" />DenseNet (Densely Connected Convolutional Networks)</td>    
  </tr>
</table>


## Project submition

You need to submit the following files or the Github Repository containing these files:

**1_train_model.ipynb** - TODO 1 and TODO 2 should be complete. See the Project Instructions page for more details. 
CREATED 

**pipeline.config** files associated with all the different pretrained models that you have tried (at least 2)
**2_deploy_model.ipynb** - TODO 3 should be complete.

Submission: Your project will be reviewed based on the criteria in the project rubric below:

**Model training and Evaluation**: 

Using google colab notebooks create a Model. 

1 - Test at least two pretrained models (other than EfficientNet) 
  - Tried two models other than EfficientNet.
  - Update and submit the pipeline.config file and notebooks associated with all the pretrained models.
2 - Choosing the best model for deployment : Write a brief summary of your experiments and suggest the best model for this problem. This should include the accuracy (mAP) values of the models you tried. Also, discuss the following:
  - How does the validation loss compare to the training loss?
  - Did you expect such behavior from the losses/metrics?
  - What can you do to improve the performance of the tested models further?

**Model Deployment**:


To deploy a model use Docker and render.
In under 

1 - Deploy the best model and run inference.
  - Deploy the best model in AWS by completing and running 2_deploy_model.ipynb.
  - Should be able to generate an output video with vehicle, pedestrian, and cyclist detections using the deployed model.


# Notes:

WIP : Working to get done Object detection using ResNet, VGG and DenseNet. Also, working to create a Docker file to be able to upload image and counts number of objects. 
1) Under folder 03_PYTROCH working with file PYTORCH_ImageObjectDetection_v3.py and website in folder "server"
2) Docker instructions:
   Build docker image -> docker build -t object_detector_app .
   Run docker image in internal server -> docker run -p 5001:5000 object_detector_app 

3) Issues : The scripts don't put boxes around images anymore.




