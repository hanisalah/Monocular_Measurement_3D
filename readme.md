# Monocular Measurement 3D

## 1. Features
1. Detect object(s) in images.
2. Draw 3D boxes around detected objects.
3. Measure real world dimensions of detected objects in meters.
4. Identify the smallest locker / crate (from a list of predefined dimensions) that can be used to box the detected object.

## 2. Installation
1. The project is using Streamlit as its frontend interface.
2. The project can run on streamlit cloud (for inference only).
3. For training the models, the project must be downloaded to a local machine with proper NVIDIA GPU.

### 2.1 Local Installation
#### 2.1.1 Platform Independent Steps
1. Install cuDNN from <href>https://developer.nvidia.com/cudnn</href> (Follow cuDNN installation requirements)
2. Create a new conda environment using ```conda create -n <env_name> python=<python_version>``` and activate it.
3. Install proper pytorch version with GPU support from <href>https://www.pytorch.org</href> (Pytorch official website)
4. Install all dependencies using ```pip install -r requirements_local.txt```
5. Install ninja from <href>https://www.ninja-build.org</href> and add it to path.

#### 2.1.2 Windows OS Steps
1. Install 'Command Line C/C++ Compiler for Microsoft' CL.exe and add it to path. Refer to <href>https://learn.microsoft.com/en-us/cpp/build/reference/compiler-options?view=msvc-170</href>
2. Browse to ```src/iou3d_win``` and run ```python setup.py install``` to compile and install IOU3D on your system.

#### 2.1.3 Linux OS Steps
1. Install GCC from <href>https://gcc.gnu.org/</href>
2. Browse to ```src/iou3d_unix``` and run ```python setup.py install``` to compile and install IOU3D on your system.

#### 2.1.4 Run the program
1. Browse to the root of the project and run ```streamlit run main.py```. The program will open in a web browser.

### 2.2 Streamlit Cloud Deployment
#### 2.2.1 Upload Project to Github
1. Signup / Signin to your Github account <href>https://www.github.com</href>
2. Upload the project to Github. You can follow instructions on <href>https://docs.github.com/en/get-started/importing-your-projects-to-github/importing-source-code-to-github/adding-locally-hosted-code-to-github</href>

#### 2.2.2 Link the Github repo to Streamlit Cloud
1. Signup / Signin to your Streamlit account <href>https://streamlit.io/</href>
2. Follow on screen instructions to link your Github account and repo.