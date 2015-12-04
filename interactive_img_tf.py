# Interactive Image transform 
from Tkinter import Tk
from tkFileDialog import askopenfilename
import numpy as np
import cv2
import cv

a, b, x, c, d, y ,v1, v2, e = (10,)*9
M = np.array([
		[ 1,  0, 0],
		[ 0,  1, 0],
		[ 0,  0, 1]
		], dtype = np.float64)

def tf_and_display():
	global tf_img
	global original_img
	global M
	tf_img = cv2.warpPerspective(original_img, M, (700,700))
	disp_img = tf_img.astype(np.uint8)
	cv2.imshow("Matrix Image Manipulation", disp_img)
	print M

def update_tf_matrix():
	# can be called after global matrix element variables are changed in order
	# to update the transformation matrix
	global M, a, b, c, d, x, y, v1, v2
	M = np.array([
		[ a,  b, x],
		[ c,  d, y],
		[v1, v2, e]
		], dtype = np.float64)

def aChange(value):
	global a
	a = (value-10)/50.0
	update_tf_matrix()
	tf_and_display()

def bChange(value):
	global b
	b = (value-10)/50.0
	update_tf_matrix()
	tf_and_display()

def cChange(value):
	global c
	c = (value-10)/50.0
	update_tf_matrix()
	tf_and_display()

def dChange(value):
	global d
	d = (value-10)/50.0
	update_tf_matrix()
	tf_and_display()

def xChange(value):
	global x
	x = (value-10)*25
	update_tf_matrix()
	tf_and_display()

def yChange(value):
	global y
	y = (value-10)*25
	update_tf_matrix()
	tf_and_display()

def v1Change(value):
	global v1
	v1 = (value-10)/10000.0
	update_tf_matrix()
	tf_and_display()

def v2Change(value):
	global v2
	v2 = (value-10)/10000.0
	update_tf_matrix()
	tf_and_display()

def eChange(value):
	global e
	e = (value-10)/50.0
	update_tf_matrix()
	tf_and_display()


print a, b
Tk().withdraw() # we don't want a full GUI, so keep the root window from appearing
filename = askopenfilename() # show an "Open" dialog box and return the path to the selected file
print(filename)

cv2.namedWindow("Transformation Matrix Controls")
cv2.createTrackbar("a", "Transformation Matrix Controls", a, 20, aChange)
cv2.createTrackbar("b", "Transformation Matrix Controls", b, 20, bChange)
cv2.createTrackbar("c", "Transformation Matrix Controls", c, 20, cChange)
cv2.createTrackbar("d", "Transformation Matrix Controls", d, 20, dChange)
cv2.createTrackbar("x", "Transformation Matrix Controls", x, 20, xChange)
cv2.createTrackbar("y", "Transformation Matrix Controls", y, 20, yChange)
cv2.createTrackbar("v1", "Transformation Matrix Controls", v1, 20, v1Change)
cv2.createTrackbar("v2", "Transformation Matrix Controls", v2, 20, v2Change)
cv2.createTrackbar("e", "Transformation Matrix Controls", e, 20, eChange)

b, c, x, y, v1, v2 = (0,)*6
a, d = 1, 1
print M
original_img = cv2.imread(filename, cv2.CV_LOAD_IMAGE_GRAYSCALE)
original_img = original_img.astype(np.float64)
tf_img = np.zeros_like(original_img, np.float64)

# print M.shape
# print M.dtype
# print original_img.dtype
cv2.waitKey(0)