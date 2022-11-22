# Adaptations
The program is an adaptation from <href>https://github.com/Banconxuan/RTM3D</href>. The source requires multiple external installs which we explain below.

## DCNv2
1. The source program references an implementation of DCNv2 which requires compilation.
2. The source <href>https://github.com/liyier90/pytorch-dcnv2/blob/master/dcn.py</href> implements DCNv2 in pure python that doesn't require compilation and has been used in the program with a standard definition of a python class.

## soft_nms_39
1. The source program references an external C implementation of soft_nms_39.
2. To avoid multiple compilations, the implementation is converted to pure python and encapsulated in a python function. Impact to performance is negligible.

## IOU3D
IOU3D is written to run on Unix based machines. To make it run on Windows machines the following is needed. Since development is done on a Windows machine, there was no room to test a Unix compilation. However, some of the below is believed to be also applicable to a Unix installation. The program will be delivered with both sources (Unix and Windows), and user will have to decide which alternative to build. Refer to installation instructions.
1. In file iou3d_kernel.cu perform the following:
    1. Comment out instruction <code>const float EPS = 1e-8;</code> to be <code>//const float EPS = 1e-8;</code>
    2. Add instruction <code>#define EPS 1e-8</code>
2. In file iou3d.cpp perform the following:
    1. Before line <code>#define CHECK_CUDA(x) AT_CHECK(x.type().is_cuda(), #x, " must be a CUDAtensor ")</code> add the following three lines:<br>
        <code>#ifndef AT_CHECK</code><br><code>#define AT_CHECK TORCH_CHECK</code><br><code>#endif</code>
    2. Modify line <code>#define CHECK_CUDA(x) AT_CHECK(x.type().is_cuda(), #x, " must be a CUDAtensor ")</code><br> with <code>#define CHECK_CUDA(x) AT_CHECK(x.is_cuda(), #x, " must be a CUDAtensor ")</code>
    3. In functions <code>boxes_overlap_bev_gpu</code> and <code>boxes_iou_bev_gpu</code> modify:<br>
        <code>const float * boxes_a_data = boxes_a.data\<float\>();</code> with <code>const float * boxes_a_data = boxes_a.data_ptr\<float\>();</code><br>
        <code>const float * boxes_b_data = boxes_b.data\<float\>();</code> with <code>const float * boxes_b_data = boxes_b.data_ptr\<float\>();</code><br>
        <code>float * ans_overlap_data = ans_overlap.data\<float\>();</code> with <code>float * ans_overlap_data = ans_overlap.data_ptr\<float\>();</code>
    4. In functions <code>nms_gpu</code> and <code>nms_normal_gpu</code> modify:<br>
        <code>const float * boxes_data = boxes.data\<float\>();</code> with <code> const float * boxes_data = boxes.data_ptr\<float\>();</code><br>
        <code>long * keep_data = keep.data\<long\>();</code> with <code>long long * keep_data = keep.data_ptr\<long long\>();</code><br>
        <code>unsigned long long remv_cpu[col_blocks];</code> with <code>unsigned long long *remv_cpu = new unsigned long long [col_blocks];</code>
3. Open a terminal and run command <code>python setup.py install</code> which will build and install IOU3D
4. IOU3D can now be imported as <br><code>import torch</code><br><code>import iou3d_cuda</code><br>Take care you must import <code>torch</code> before you import <code>iou3d_cuda</code>

## Simplifications
1. The above referenced program has been simplified to match the needs, and the current implementation does not equate exactly to the referenced project.
