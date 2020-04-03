# BallTracking
Track ball trajectories from videos using traditional image processing techniques

## Required Packages
- Python3.7
  - OpenCV
  - numpy
  - imutils
  - matplotlib

## Instructions
### BallTracking.py
Contains function *BallTracking(vidDir, show=)* 
- Inputs: 
  - video directory (String)
  - show video frame on/off (Bool)
- Return: 
  - Nx2 position np array containing coordinates of ball in each frame (x in first column, y in second column), -1      denotes ball not found in frame.
  - A list of image for overlay
          
### getHSV.py
get mean, max, min of HSV values of a image
