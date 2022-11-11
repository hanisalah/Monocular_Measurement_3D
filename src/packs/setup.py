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

add_pkgs=[]
if is_deployed:
    add_pkgs=['pytorch', 'opencv-python-headless','streamlit-drawable-canvas']
else:
    add_pkgs =['pytorchGPU','opencv-python', 'streamlit-drawable-canvas==0.9.0']


import torch
try:
    import iou3d_cuda
except:
    import platform
    if platform.system() == 'Windows':
        subprocess.run([sys.executable,'-m', 'pip', 'install', './src/iou3d_win'])
    else:
        subprocess.run([sys.executable,'-m', 'pip', 'install', './src/iou3d_unix'])
    import importlib
    importlib.invalidate_caches()
    import sys
    import torch
    import iou3d_cuda



from torch.utils.cpp_extension import BuildExtension, CUDAExtension
setup(
    name='iou3d',
    ext_modules=[
        CUDAExtension('iou3d_cuda', [
            'src/iou3d.cpp',
            'src/iou3d_kernel.cu',
        ],
        extra_compile_args={'cxx': ['-g'],
                            'nvcc': ['-O2']})
    ],
    cmdclass={'build_ext': BuildExtension})

setup(name='packs',
        version='1.0.0',
        author = 'Hani Salah',
        packages=[],
        scripts=[],
        description='dummy pack to control vers',
        install_requires=
        ['numpy','opencv-python']
        )
