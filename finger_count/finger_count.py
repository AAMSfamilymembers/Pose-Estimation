# IN THIS , WE ARE GOING TO DETECT THE NUMBER OF FINGERS IN YOUR HAND
# THIS WORKS FOR UNICOLOR ENVIRONMENTS AND OF COLOUR DIFFERENT FROM SKIN
# IMPORTING LIBRARIES
import numpy as np
import cv2
import math

#LOADING HAND CASCADE
hand_cascade = cv2.CascadeClassifier('/home/abhay/palm.xml')

# VIDEO CAPTURE
video = cv2.VideoCapture(0)
# INITIALIZING RETURN LIST TO STORE OTSU THRESHOLD VALUE
ret_list=list()
while 1:
	ret, img = video.read()

	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 
	
	hand = hand_cascade.detectMultiScale(gray, 1.3, 5) 
	
	for (x,y,w,h) in hand: 
		cv2.rectangle(img,(x,y),(x+w,y+h), (122,122,0), 2) 
		
		roi_gray=gray[y:y+h,x:x+h]  
		retval2,thresh1 = cv2.threshold(roi_gray,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU) 
		ret_list.append(retval2)
		if retval2 == 0:
			break
			
	
	ret3 = np.sum(ret_list)/len(ret_list)  
	retval2,thresh1 = cv2.threshold(gray,ret3,255,cv2.THRESH_BINARY_INV)  	
	final = cv2.GaussianBlur(thresh1,(7,7),0)	                        
	image,contours, hierarchy = cv2.findContours(final, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)         	

	if len(contours) > 0:
		cnt=contours[0]
		hull = cv2.convexHull(cnt, returnPoints=False)
		print(hull)
		# finding convexity defects
		defects = cv2.convexityDefects(cnt, hull)
		count_defects = 0
		# applying Cosine Rule to find angle for all defects (between fingers)
		# with angle > 90 degrees and ignore defect
		if type(defects)!= type(None):                                           
			for i in range(defects.shape[0]):
				p,q,r,s = defects[i,0]
				finger1 = tuple(cnt[p][0])
				finger2 = tuple(cnt[q][0])
				dip = tuple(cnt[r][0])
				
				a = math.sqrt((finger2[0] - finger1[0])**2 + (finger2[1] - finger1[1])**2)
				b = math.sqrt((dip[0] - finger1[0])**2 + (dip[1] - finger1[1])**2)
				c = math.sqrt((finger2[0] - dip[0])**2 + (finger2[1] - dip[1])**2)
				# apply cosine rule here
				angle = math.acos((b**2 + c**2 - a**2)/(2*b*c)) * 57.29         
				if angle <= 90:
				    count_defects += 1
		# define actions required
		if count_defects == 0:
			cv2.putText(img,"THIS IS 1", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, 2)
		elif count_defects == 1:
			cv2.putText(img,"THIS IS 2", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, 2)
		elif count_defects == 2:
			cv2.putText(img, "THIS IS 3", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, 2)
		elif count_defects == 3:
			cv2.putText(img,"This is 4", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, 2)
		elif count_defects == 4:
			cv2.putText(img,"THIS IS 5", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, 2)	
		
	cv2.imshow('img',thresh1)
	cv2.imshow('img1',img)

	k = cv2.waitKey(30) & 0xff
	if k == 27:
		break
video.release()
cv2.destroyAllWindows()







