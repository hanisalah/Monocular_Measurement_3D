from setuptools import setup
import sys
import os
import ctypes

def get_CUDA_details():
    is_cuda = True
    CUDA_SUCCESS = 0
    result = ctypes.c_int()
    error_str = ctypes.c_char_p()
    version_dll = ctypes.c_int()
    version='0.0'

    libnames = ("libcuda.so", "libcuda.dylib", "cuda.dll", "nvcuda.dll") #first 3 are used in linux/mac, nvcuda.dll is windows
    for libname in libnames:
        try:
            cuda = ctypes.CDLL(libname)
        except OSError:
            continue
        else:
            break
    else:
        print('No CUDA driver or library found!')
        is_cuda=False

    if is_cuda:
        result = cuda.cuInit(0)
        if result != CUDA_SUCCESS:
            cuda.cuGetErrorString(result, ctypes.byref(error_str))
            print("cuInit failed with error code %d: %s" % (result, error_str.value.decode()))
        if cuda.cuDriverGetVersion(ctypes.byref(version_dll)) == CUDA_SUCCESS:
            ver = version_dll.value #The version is returned as (1000 major + 10 minor). For example, CUDA 9.2 would be represented by 9020
            major= int(ver//1000)
            minor= int((ver-(major*1000))//10)
            version = str(major)+'.'+str(minor)
            print("CUDA DRIVER {}".format(version))
    return {'is_cuda':is_cuda, 'version':version}


CUDA_details = get_CUDA_details()
is_cuda = CUDA_details['is_cuda'] #Presence of cuda also indicates that app is not deployed on streamlit cloud, as streamlit cloud has no GPU available.
cuda_version = CUDA_details['version']

cuda_to_pytorch_ver = {
                        '11.6':{'torch':'1.12.1+cu116', 'torchvision':'0.13.1+cu116', 'extra':'--extra-index-url https://download.pytorch.org/whl/cu116'},
                        '11.3':{'torch':'1.12.1+cu113', 'torchvision':'0.13.1+cu113', 'extra':'--extra-index-url https://download.pytorch.org/whl/cu113'},
                        '10.2':{'torch':'1.12.1+cu102', 'torchvision':'0.13.1+cu102', 'extra':'--extra-index-url https://download.pytorch.org/whl/cu102'},


                        'cpu' :{'torch':'1.12.1+cpu', 'torchvision':'0.13.1+cpu', 'extra':'--extra-index-url https://download.pytorch.org/whl/cpu'}}

install_requires=[]
if not torch_installed:
    if is_cuda:
        install_requires.append(cuda_to_pytorch_ver[cuda_version]['torch'])
        install_requires.append(cuda_to_pytorch_ver[cuda_version]['torchvision'])
    else:
        install_requires.append(cuda_to_pytorch_ver['cpu']['torch'])
        install_requires.append(cuda_to_pytorch_ver['cpu']['torchvision'])
if is_cuda:
    install_requires.append('opencv-python')
    install_requires.append('streamlit-drawable-canvas==0.9.0')
else:
    install_requires.append('opencv-python-headless')
    install_requires.append('streamlit-drawable-canvas')

setup(name='packs',
        version='1.0.0',
        author = 'Hani Salah',
        packages=[],
        scripts=[],
        description='dummy pack to control vers',
        install_requires=install_requires
        )