#IMPORTING THE LIBRARIES
import cv2
import numpy as np
import math

# Checks if a matrix is a valid rotation matrix.
def isRotationMatrix(R) :
	Rt = np.transpose(R)
	shouldBeIdentity = np.dot(Rt, R)
	I = np.identity(3, dtype = R.dtype)
	n = np.linalg.norm(I - shouldBeIdentity)
    	return n < 1e-6

# Calculates rotation matrix to euler angles
		# The result is the same as MATLAB except the order
		# of the euler angles ( x and z are swapped ).
def rotationMatrixToEulerAngles(R) :
 
   	 assert(isRotationMatrix(R))
     
    	 sy = math.sqrt(R[0,0] * R[0,0] +  R[1,0] * R[1,0])
     
    	 singular = sy < 1e-6
 
    	 if  not singular :
         	x = math.atan2(R[2,1] , R[2,2])
        	y = math.atan2(-R[2,0], sy)
        	z = math.atan2(R[1,0], R[0,0])
    	 else :
        	x = math.atan2(-R[1,2], R[1,1])
        	y = math.atan2(-R[2,0], sy)
        	z = 0
 
    	 return np.array([57.3*x, 57.3*y, 57.3*z])

# thresholding of blue square in the marker

def blue_square(hsv):
	lower_blue = np.array([55,50,50])
        upper_blue = np.array([130,255,255])
	blue_mask = cv2.inRange(hsv,lower_blue,upper_blue)                                           # threshold of the range specified above
	image,contours,hierarchy = cv2.findContours(blue_mask,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
	cnt=sorted(contours, key=cv2.contourArea, reverse=True)[0]                                   # sorting in descending order and using largest contour         
	epsilon = 0.1*cv2.arcLength(cnt,True)
	approx = cv2.approxPolyDP(cnt,epsilon,True)
	if(cv2.contourArea(cnt)>10000 and len(approx)==4):
		#cv2.drawContours(img,[approx],-1,(0,255,0),3)
		return(approx)
	
# thresholding the red circle and finding the center of circle

def red_circle(hsv):
	lower_red = np.array([160,70,50]) #
        upper_red = np.array([180,255,255]) 
	red_mask = cv2.inRange(hsv,lower_red,upper_red) # threshold of this range
	image,contours,hierarchy = cv2.findContours(red_mask,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
	cnt=sorted(contours, key=cv2.contourArea, reverse=True)[0]                                    # largest contour
	M = cv2.moments(cnt)
	if M["m00"] != 0:
    		cx = int(M['m10']/M['m00'])
    		cy = int(M['m01']/M['m00'])
        	#cv2.circle(img,(cx,cy),2, (255,255,255),3)
		return(cx,cy)


# thresholding the green circle and finding the center of circle

def green_circle(hsv):
	lower_green = np.array([40,50,50])
        upper_green = np.array([70,255,255])
	green_mask = cv2.inRange(hsv,lower_green,upper_green) # threshold of this range
	image,contours,hierarchy = cv2.findContours(green_mask,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
	cnt=sorted(contours, key=cv2.contourArea, reverse=True)[0]                                    # largest contour
	M = cv2.moments(cnt)
	if M["m00"] != 0:
    		cx = int(M['m10']/M['m00'])
    		cy = int(M['m01']/M['m00'])
        	#cv2.circle(img,(cx,cy),2, (255,255,255),3)
		return(cx,cy)

#finding the nearest point to the circles by finding the minimum distance between the point and center of circles

def min_dist(a,approx):

	min = np.linalg.norm(approx[0][0]-a)   # basically using vector ,normalising the output vector, it is an easy way to calculate dist rather than using sqrt
	index=0
	for i in range(4):
		b=np.linalg.norm(approx[i][0]-a)
		if (min>b):
			min = b
			index = i
	
	return approx[index][0],index
	#return(min)

def main():
	# VIDEO CAPTURE
	video=cv2.VideoCapture(0)
	while(True):
		ret,img=video.read()
		hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)  # BGR to HSV CONVERSION
		approx=  blue_square(hsv)                   # storimg the four points in approx
		new_imgpts = []                             # initialising for new image points ,well not compulsary
		red_center = red_circle(hsv)                # calling function
		green_center = green_circle(hsv)            # calling function
		b,c = min_dist(red_center,approx)           # calling function
		a,d = min_dist(green_center,approx)         # calling function
		# LOGIC FOR THE OTHER TWO POINTS
		# INDEX IS CLOCKWISE ALSO ACCORDING TO WEBCAM
`		
		
		if(c!=3):
			blue = approx[c+1][0]
			
		else:
			blue = approx[0][0]
			
		if(d!=3):
			yellow = approx[d+1][0]
			
		else:
			yellow = approx[0][0]
			
	
		new_imgpts = np.array([[[yellow[0],yellow[1]]],[[b[0],b[1]]],[[blue[0],blue[1]]],[[a[0],a[1]]]],dtype=np.float32) # inserting the coordinates of the image points in the format which is 	                                                                                                                                      required for solvePnp
		
		cv2.circle(img,(a[0],a[1]),2, (0,255,0),3)              # point near green circle idicated by green
		cv2.circle(img,(b[0],b[1]),2, (0,0,255),3)              # point near red circle idicated by red
		cv2.circle(img,(blue[0],blue[1]),2, (255,0,0),3)        # point in  between  idicated by blue
		cv2.circle(img,(yellow[0],yellow[1]),2, (0,255,255),3)  # point in between idicated by yellow
		cv2.imshow('img',img)
		k=(cv2.waitKey(1) & 0xFF)
		if k==27:
			break	

		# DEFINING THE ORIGIN AS THE CENTER OF THE MARKER 
		# SETTING THE THE RED CIRCLE TO BE IN THE FIRST QUADRANT HENCE POINT NEAR IT HAS BOTH X AND Y AS POSITIVE
		# THE CAMERA MATRIX AND DISTORTION COEFFICIENT WERE OBTAINED CALIBERATING IT WITH CHESS BOARD 
		objectPoints = np.array([[[-9,9,0]],[[9,9,0]],[[9,-9,0]],[[-9,-9,0]]],dtype=np.float64)
		cameraMatrix= np.array([ 6.4456330098188391e+02, 0, 3.0692409639227708e+02, 0,
       6.4478109440362607e+02, 2.4241059798021956e+02, 0, 0, 1])
		cameraMatrix=cameraMatrix.reshape(3,3)                                 
		distCoeffs=np.array([ -1.4433997821822588e-02, -6.1436406199815635e-02,
       5.5276557117156597e-03, -5.7851858669865849e-03,
       4.8367215921456952e-01 ])
		retval, rvec,tvec=cv2.solvePnP(objectPoints, new_imgpts, cameraMatrix, distCoeffs)
		dst,jacobian=cv2.Rodrigues(rvec)
		isRotationMatrix(dst)
		b = rotationMatrixToEulerAngles(dst)
		print(b)
		

if __name__=="__main__":
	main()
	
