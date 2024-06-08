# HOW TO RUN THESE SCRIPTS
You need to create a virtual environment in CONDA and use next libraries. <br>
1-  conda create --name myenv python=3.11 #Create virtual env <br>
2 - conde activate myenv #Activate conda environment <br>
3-  conda install -c conda-forge opencv #Install OpenCV libraru <br> 
4-  conda install -c conda-forge imageio <br>
5-  conda install pytorch torchvision torchaudio cpuonly -c pytorch #Install pytorch CPU version <br>


# COMPUTER VISION
<table>
  <tr>
    <td align="center"><img src="https://github.com/marcjesus/udacity/blob/main/01_ObjectDetection/output.gif" alt="Image 1" width="400" /> Original Video</td>
    <td align="center"><img src="https://github.com/marcjesus/udacity/blob/main/01_ObjectDetection/OPENCV_output_gif.gif" alt="Image 2" width="400" />Using OpenCV</td>
  </tr>
</table>

## Using PYTORCH
PyTorch is a deep learning framework. It's an open-source machine learning library primarily designed for building and training deep neural networks. PyTorch provides a flexible and intuitive interface that facilitates the creation and experimentation of complex deep learning models.


## Training a model



## Using a trained model    



## Project submition

You need to submit the following files or the Github Repository containing these files:


**1_train_model.ipynb** - TODO 1 and TODO 2 should be complete. See the Project Instructions page for more details.
**pipeline.config** files associated with all the different pretrained models that you have tried (at least 2)
**2_deploy_model.ipynb** - TODO 3 should be complete.

Submission: Your project will be reviewed based on the criteria in the project rubric below:

**Model training and Evaluation**: 

1 - Test at least two pretrained models (other than EfficientNet) 
  - Tried two models other than EfficientNet.
  - Update and submit the pipeline.config file and notebooks associated with all the pretrained models.
2 - Choosing the best model for deployment : Write a brief summary of your experiments and suggest the best model for this problem. This should include the accuracy (mAP) values of the models you tried. Also, discuss the following:
  - How does the validation loss compare to the training loss?
  - Did you expect such behavior from the losses/metrics?
  - What can you do to improve the performance of the tested models further?

**Model Deployment**:

1 - Deploy the best model and run inference.
  - Deploy the best model in AWS by completing and running 2_deploy_model.ipynb.
  - Should be able to generate an output video with vehicle, pedestrian, and cyclist detections using the deployed model.

