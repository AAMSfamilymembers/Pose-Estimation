# Pose-Estimation

Pose estimation of a custom marker

I have designed a blue rectangular-marker with a red and green circle inside it.

STEPS INVOLVED:-

1. Marker segmentation from background.
2. Extracting the corners of the marker.
3. Labelling the corners of the marker corresponding to the world-cordinates
4. Calculation of Rotation and translation using solvePnP

# SubTask Involved
To detect hand and count the number of fingers shown 

STEPS INVOLVED:-

1. Hand detection using Haar cascade.
2. Contour of hand and convex hull.
3. Find the Convexity defects.
4. Count the number of fingers with the help of available defects.
