# Model Training

1 - Prepare dataset : Collect and annotate images with cars and persons using COCO format or a custom dataset format. <br>

2 - Create the Dataset and DataLoader : Defoine a custom dataset class that inhertis from 'torch.utils.data.Dataset.' Use 'DataLoader' to handle batch processing and shuffling. <br>

https://medium.com/datatorch/how-to-create-a-custom-coco-dataset-from-scratch-7cd28f0a2c88

3 - Load a Pre-trained Model : Load a pre-trained object detection model (e.g, Faster R-CNN) from torchversion. Modify the model if necessary (e.g, change the number of classes). <br>

4 - Define the training Loop : Set up the optimizer and learning rate scheduler. Define the training loop, including forward and backward passes, loss calculation, and optimization steps.

5 - Evaluate the Model : Use a validation set to evaluate the model performance. Calculate metrics sucg as mAP (mean Average Precision).
