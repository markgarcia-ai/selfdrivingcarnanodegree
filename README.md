# TODO

Working to train my own model. Steps: <br>

**STEP 1: Creating a website** where you can upload the Image and it counts number of objects and then as a human you can introduce your inputs to calculate (mAP). 
App will run in http://localhost:8000/ or any port of your selection<br>
Running under server folder using docker instructions to run in my computer -> 'docker build -t app . ' and run 'docker -run -p 8000:8000 app' open docker desktop and in containers check it's the new one running<br>
Some tips for errors trying to run webflask because ports are used -> Check ports usage for example : 'lsof -i :5000' and then kill the process 'kill -9 PID' <br>
Website is host in render <br>

**STEP 2: Train own model** to detect objects not present from previous model. <br>
I will be adding trees detection to my new model based on R-CNN ResNet-50 model. <br>





**STEP 3: Update website** to show analysis with base model and analysis with new model <br>

