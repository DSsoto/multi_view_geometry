import cv2
import numpy as np
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as p3

def euclidean_dist(vec1, vec2):
	# function returns the euclidean distance between two inhomogeenous n-dimensional points
	if vec1.shape != vec2.shape:
		print "Error: cannot calculate euclidean distance between two vectors of differing size"
		return
	diff = vec1 - vec2
	# print "diff", diff, vec1, vec2
	square_diffs = np.dot(diff, diff)
	# print "square_diffs", square_diffs
	dist = np.sqrt(np.sum(square_diffs))
	return dist



def normalize_homogenous_coordinate(array):
    # the input np.array should have columns that are homogenous 4-vectors
    # this function will divide all of the elements in a column by the 
    # last element of that column
    for col in range(np.shape(array)[1]):
        array[:, col] = array[:, col] / float(array[3, col])
    return array

def tf_and_display():
	global tf_matrix, org_pts, tf_pts, fig, ax, colors
	print "Transform:\n", tf_matrix
	tf_pts = np.dot(tf_matrix, org_pts)
	tf_pts = normalize_homogenous_coordinate(tf_pts)
	# print "TF Points:\n", tf_pts
	points = ax.scatter(tf_pts[0,:], tf_pts[1,:], tf_pts[2,:], color=colors, marker='o')
	for pt1 in range(8):
		for pt2 in range(8):
			if pt2 < pt1:
				continue
			else:
				color = (
					(colors[pt1][0] + colors[pt2][0]) / 2.0,
					(colors[pt1][1] + colors[pt2][1]) / 2.0,
					(colors[pt1][2] + colors[pt2][2]) / 2.0
					)
				length = euclidean_dist(tf_pts[0:3,pt1],tf_pts[0:3,pt2])
				# print "length", length
				original_length = euclidean_dist(org_pts[0:3,pt1],org_pts[0:3,pt2])
				strain = (length - original_length) / original_length
				color_coeff = strain / 10 - 0.1
				if color_coeff < 0:
					color_coeff = 0.0
				if color_coeff > 0.8:
					color_coeff = 0.8
				# print "color_coeff", color_coeff
				ax.plot(tf_pts[0,[pt1,pt2]],tf_pts[1,[pt1,pt2]],tf_pts[2,[pt1,pt2]], color=plt.cm.hot(color_coeff) )

	plt.draw()
	points.remove()
	for i in range(36):
		ax.lines.pop(0)


def cb1(value):
	global tf_matrix
	print "cb1"
	tf_matrix[0,0] = ( value - (slider_max*0.5)) / 10.0
	tf_and_display()

def cb2(value):
	global tf_matrix
	tf_matrix[0,1] = ( value - (slider_max*0.5)) / 10.0
	tf_and_display()

def cb3(value):
	global tf_matrix
	tf_matrix[0,2] = ( value - (slider_max*0.5)) / 10.0
	tf_and_display()

def cb4(value):
	global tf_matrix
	tf_matrix[0,3] = ( value - (slider_max*0.5)) / 10.0
	tf_and_display()

def cb5(value):
	global tf_matrix
	tf_matrix[1,0] = ( value - (slider_max*0.5)) / 10.0
	tf_and_display()

def cb6(value):
	global tf_matrix
	tf_matrix[1,1] = ( value - (slider_max*0.5)) / 10.0
	tf_and_display()

def cb7(value):
	global tf_matrix
	tf_matrix[1,2] = ( value - (slider_max*0.5)) / 10.0
	tf_and_display()

def cb8(value):
	global tf_matrix
	tf_matrix[1,3] = ( value - (slider_max*0.5)) / 10.0
	tf_and_display()

def cb9(value):
	global tf_matrix
	tf_matrix[2,0] = ( value - (slider_max*0.5)) / 10.0
	tf_and_display()

def cb10(value):
	global tf_matrix
	tf_matrix[2,1] = ( value - (slider_max*0.5)) / 10.0
	tf_and_display()

def cb11(value):
	global tf_matrix
	tf_matrix[2,2] = ( value - (slider_max*0.5)) / 10.0
	tf_and_display()

def cb12(value):
	global tf_matrix
	tf_matrix[2,3] = ( value - (slider_max*0.5)) / 10.0
	tf_and_display()

def cb13(value):
	global tf_matrix
	tf_matrix[3,0] = ( value - (slider_max*0.5)) /50.0
	tf_and_display()

def cb14(value):
	global tf_matrix
	tf_matrix[3,1] = ( value - (slider_max*0.5)) /50.0
	tf_and_display()

def cb15(value):
	global tf_matrix
	tf_matrix[3,2] = ( value - (slider_max*0.5)) /50.0
	tf_and_display()

def cb16(value):
	global tf_matrix
	tf_matrix[3,3] = 1 + ( value - (slider_max*0.5)) /50.0
	tf_and_display()


tf_matrix = np.array([		          	 # 4x4 homogenous tf matrix
	[1, 0, 0, 0],
	[0, 1, 0, 0],
	[0, 0, 1, 0],
	[0, 0, 0, 1.0]
	])

org_pts = np.array([		             # original homogenous point coordinates
	[1, -1,  1,  1, -1,  1, -1,   -1],   # x
	[1,  1, -1,  1, -1, -1,  1,   -1],   # y
	[1,  1,  1, -1,  1, -1, -1,   -1],   # z
	[1,  1,  1,  1,  1,  1,  1,  1.0]    # homogenous coordinate
	])

tf_pts = np.ones_like(org_pts)

slider_max = 100
slider_start = slider_max / 2
cv2.namedWindow("Transformation Matrix Controls")
cv2.createTrackbar("m1", "Transformation Matrix Controls", slider_start, slider_max, cb1)
cv2.createTrackbar("m2", "Transformation Matrix Controls", slider_start, slider_max, cb2)
cv2.createTrackbar("m3", "Transformation Matrix Controls", slider_start, slider_max, cb3)
cv2.createTrackbar("m4", "Transformation Matrix Controls", slider_start, slider_max, cb4)
cv2.createTrackbar("m5", "Transformation Matrix Controls", slider_start, slider_max, cb5)
cv2.createTrackbar("m6", "Transformation Matrix Controls", slider_start, slider_max, cb6)
cv2.createTrackbar("m7", "Transformation Matrix Controls", slider_start, slider_max, cb7)
cv2.createTrackbar("m8", "Transformation Matrix Controls", slider_start, slider_max, cb8)
cv2.createTrackbar("m9", "Transformation Matrix Controls", slider_start, slider_max, cb9)
cv2.createTrackbar("m10", "Transformation Matrix Controls", slider_start, slider_max, cb10)
cv2.createTrackbar("m11", "Transformation Matrix Controls", slider_start, slider_max, cb11)
cv2.createTrackbar("m12", "Transformation Matrix Controls", slider_start, slider_max, cb12)
cv2.createTrackbar("m13", "Transformation Matrix Controls", slider_start, slider_max, cb13)
cv2.createTrackbar("m14", "Transformation Matrix Controls", slider_start, slider_max, cb14)
cv2.createTrackbar("m15", "Transformation Matrix Controls", slider_start, slider_max, cb15)
cv2.createTrackbar("m16", "Transformation Matrix Controls", slider_start, slider_max, cb16)

# Attaching 3D axis to the figure
plt.ion()
fig = plt.figure()
ax = p3.Axes3D(fig)
colors = [tuple(np.random.rand(3)) for i in range(8)]

# Setting the axes properties
ax.set_xlim3d([-10.0, 10.0])
ax.set_xlabel('X')

ax.set_ylim3d([-10.0, 10.0])
ax.set_ylabel('Y')

ax.set_zlim3d([-10.0, 10.0])
ax.set_zlabel('Z')

ax.set_title('Real-Time Interactive Homogenous Matrix Transformations')
# mpl_animation_thread = threading.Thread(target=mpl_animation)
# mpl_animation_thread.start()
cv2.waitKey(0)