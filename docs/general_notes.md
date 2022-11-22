# General Notes
## Notes for Camera Calibration
1. Prepare a chessboard shape as shown in the above image. The chessboard should be printed and fixed on a plain surface. It could also be displayed on a tablet. Take note of the following:
    1. Number of inner corners of the chessboard (in the example below it is 6x9). First count number of inner corners in one column (vertical), then count number of inner corners in one row (horizontal).
    2. The size of one square in the chessboard in millimeter.
2. Take multiple photos with your camera of the chessboard image. Make sure that camera auto focus is turned off before you shoot your photos. Take 10 to 20 images by moving around in the camera frame. Save the photos at a known folder.

## Notes for Shooting Images
1. Take photos of objects of interest using the calibrated camera. For KITTI dataset, there are 7000+ training images for 3 objects of interest. So take lots of images. Make sure autofocus in all cases is turned off.
2. Image files shall be all named in sequential 6-digit number order. For example, first image shall be named 000000.png, next one is 0000001.png and so on.

## Notes for Annotating Images
1. The program also contains an annotation tool that can be used to annotate the datasets (refer to the page Label Dataset), other annotator tools can also be used (e.g. <href>https://alpslabel.wordpress.com/2017/01/26/alt/</href>)

2. Each image must be annotated according to the KITTI dataset format shown below.

| Num elements | Parameter name | Description | Type | Range | Example | Remark |
|:------------:|:--------------:|:-----------:|:----:|:-----:|:-------:|:------:|
| 1 | Class names | Class to which the object belongs | String | N/A | Person, Car | Use unique class names for objects |
| 1 | Truncation | How much of the object has left image boundaries. | Float | 0.0, 1.0 | 0.0 | Not used. Can be set to 0.0 |
| 1 | Occlusion | Occlusion state (0=fully visible, 1=partly visible, 2=largely occluded, 3= unknown.) | Integer | [0,3] | 2 | Not Used. Can be set to 0 |
| 1 | Alpha | Observation Angle of object | Float | [-pi, pi] | 0.146 | Not used. Can be set to 0.0 |
| 4 | Bounding box coordinates: [xmin, ymin, xmax, ymax] | Location of the object in the image | Float(0 based index) | [0 to image width],[0 to image_height], [top_left, image_width], [bottom_right, image_height] | 100 120 180 160 | See next section for possible ways to get this info |
| 3 | 3D dimension | Height, width, length of the object (in meters) | Float | N/A | 1.65, 1.67, 3.64 | Measure the object |
| 3 | Location | 3D object location x, y, z in camera coordinates (in meters) | Float | N/A | -0.65,1.71, 46.7 | See next section for possible ways to get this info |
| 1 | Rotation_y | Rotation ry around the Y-axis in camera coordinates | Float | [-pi, pi] | -1.59 | See next section for possible ways to get this info |

3. For each image, there should exist a text file with the same 6-digit image name (in different folder). In each text file, there will exist one line for each object in the image, each line contain 15 space separated entries reflecting the above. Below is a sample:<br> <code>Pedestrian 0.00 0 -0.20 712.40 143.00 810.73 307.92 1.89 0.48 1.20 1.84 1.47 8.41 0.01</code>

## Notes for program settings
1. The settings are mainly related to defining and training the model.
2. The settings that require user attention are already available via streamlit through the user friendly interface.
3. Other advanced settings are available in the file '''src/opts.py'''.
