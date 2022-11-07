import cv2
import numpy as np

def order_corners(corners,board_size):
    """
    function orders the corners obtained from chessboard calibration to match
    the same order of points identified in real world coordinates.

    inputs:
        corners: set of corners obtained from chessboard / cornersubpix functions
        board_size: number of inner squares of chessboard as tuple

    outputs:
        x_ordered: ordered corners in the same order identified in read-world coordinates
        actual: actual direction of ordered points H7V9 or H9V7
    """
    mn = min(board_size[0],board_size[1])
    mx = max(board_size[0],board_size[1])
    p0 = corners[0,0,:]
    p6 = corners[mn-1,0,:] #6
    p8 = corners[mx-1,0,:] #8
    actual='TBD'
    if abs(p0[0]-p6[0])>abs(p0[1]-p6[1]): #if dx>dy for first seven points, then they are horizontal
        #horizontal
        if ((((p0[0]-p6[0])**2) + ((p0[1]-p6[1])**2))**0.5) > ((((p0[0]-p8[0])**2) + ((p0[1]-p8[1])**2))**0.5): #if len1-7 > len1-9 7 points are in single line
            #9x7
            x = corners.reshape(board_size[1],board_size[0],2).copy() #'V9H7'
            tempx = np.array([x[:,i,:] for i in range(x.shape[1])])
            x = tempx.copy()
            actual = 'H7V9'
        else:
            #7x9
            x = corners.reshape(board_size[0],board_size[1],2).copy() #'V7H9'
            tempx = np.array([x[:,i,:] for i in range(x.shape[1])])
            x = tempx.copy()
            actual = 'H9V7'
    else:
        #vertical
        if ((((p0[0]-p6[0])**2) + ((p0[1]-p6[1])**2))**0.5) > ((((p0[0]-p8[0])**2) + ((p0[1]-p8[1])**2))**0.5): #if len1-7 > len1-9 7 points are in single line
            #9x7
            x = corners.reshape(board_size[1],board_size[0],2).copy()
            actual ='H9V7'
        else:
            #7x9
            x = corners.reshape(board_size[0],board_size[1],2).copy()
            actual='H7V9'

    p1=x[0,0,:] #first corner of chessboard as big outlined rectangle
    p2=x[0,-1,:] #second corner of chessboard as big outlined rectangle
    p3=x[-1,0,:] #third corner of chessboard as big outlined rectangle
    p4=x[-1,-1,:] #fourth corner of chessboard as big outlined rectangle
    direction = 'TBD'

    x_ordered = x.copy()

    if (actual == 'V9H7') or (actual == 'V7H9'): #outer order is vertical, inner order is horizontal
        if p1[0]-p2[0]>0: #if first horizontal line is ordered right to left invert to be left to right, else leave as is
            x_ordered = x_ordered[:,::-1,:]
        if p1[1]-p4[1]>0: #if first vertical line is ordered down to up invert to be up to down, else leave as is
            x_ordered = x_ordered[::-1,:,:]

    else: #outer order is horizontal, inner order is vertical
        if p1[1]-p2[1]>0: #if first vertical line is ordered down to up invert to be up to down, else leave as is
            x_ordered = x_ordered[:,::-1,:]
        if p1[0]-p4[0]>0: #if first horizontal line is ordered right to left invert to be left to right, else leave as is
            x_ordered = x_ordered[::-1,:,:]

    return x_ordered.reshape(board_size[0]*board_size[1],1,2) , actual

def calibrate_camera(imgs, chessboardSize, chessboardSizemm):
    # Termination criteria for refining the detected corners
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 1000000, 0.000001)

    zeros = np.zeros((chessboardSize[0]*chessboardSize[1],1,3),np.float32)
    objp_V9H7 = np.mgrid[0:chessboardSize[1],0:chessboardSize[0]].T #9x7x2 inner line is hor 7p (l->R) moving vertical down
    objp_H9V7 = np.mgrid[0:chessboardSize[1],0:chessboardSize[0]].T[:,:,::-1] #9x7x2 inner line is ver 7p (U->D) moving horizontal right
    objp_V7H9 = np.mgrid[0:chessboardSize[0],0:chessboardSize[1]].T #7x9x2 inner line is hor 9p (l->R) moving vertical down
    objp_H7V9 = np.mgrid[0:chessboardSize[0],0:chessboardSize[1]].T[:,:,::-1] #7x9x2 inner line is ver 9p (U->D) moving horizontal right
    objp_V9H7 = objp_V9H7 * chessboardSizemm
    objp_H9V7 = objp_H9V7 * chessboardSizemm
    objp_V7H9 = objp_V7H9 * chessboardSizemm
    objp_H7V9 = objp_H7V9 * chessboardSizemm

    img_ptsL = []
    obj_pts = []
    dirL = 'TBD'

    print("Calculating camera parameters ... ")

    for i in range(len(imgs)):
        imgL = imgs[i].copy()
        imgL_gray = cv2.cvtColor(imgL,cv2.COLOR_BGR2GRAY)
        outputL = imgL_gray.copy()
        retL, cornersL = cv2.findChessboardCorners(outputL,chessboardSize,None)

        if retL:
            cornersL=cv2.cornerSubPix(outputL,cornersL,(11,11),(-1,-1),criteria)
            cv2.drawChessboardCorners(outputL,chessboardSize,cornersL,retL)

            cornersL, dirL = order_corners(cornersL, (chessboardSize[1],chessboardSize[0]))

            if dirL != 'TBD':
                img_ptsL.append(cornersL)
            if dirL=='V9H7':
                objp = zeros.copy()
                objp[:,:,:2]=objp_V9H7.reshape(-1,1,2)
                obj_pts.append(objp)
                dmns=(chessboardSize[1],chessboardSize[0])
            elif dirL=='H9V7':
                objp = zeros.copy()
                objp[:,:,:2]=objp_H9V7.reshape(-1,1,2)
                obj_pts.append(objp)
                dmns=(chessboardSize[1],chessboardSize[0])
            elif dirL=='V7H9':
                objp = zeros.copy()
                objp[:,:,:2]=objp_V7H9.reshape(-1,1,2)
                obj_pts.append(objp)
                dmns=(chessboardSize[0],chessboardSize[1])
            elif dirL=='H7V9':
                objp = zeros.copy()
                objp[:,:,:2]=objp_H7V9.reshape(-1,1,2)
                obj_pts.append(objp)
                dmns=(chessboardSize[0],chessboardSize[1])
            dirL='TBD'

    # Calibrating camera
    retL, new_mtxL, distL, rvecsL, tvecsL = cv2.calibrateCamera(obj_pts,img_ptsL,imgL_gray.shape[::-1],None,None,criteria=criteria)
    noRot = np.concatenate([np.eye(3),np.array([[0],[0],[0]])],axis=1)
    calib_np = np.matmul(new_mtxL,noRot)

    print("Saving parameters ...")
    s = 'P0: \nP1: \nP2: '
    calib = sum(calib_np.tolist(),[]) #convert array to list and flatten it
    calib = ' '.join(str(i) for i in calib)
    s += calib+'\n'
    s += 'P3: \n'
    return s, calib_np
