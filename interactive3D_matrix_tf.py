import sys
import cv2
import numpy as np
from numpy import linalg as LA
from math import isinf
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as p3

def random_quaternion(rand=None):
    """Return uniform random unit quaternion.
    rand: array like or None
        Three independent random variables that are uniformly distributed
        between 0 and 1.
    >>> q = random_quaternion()
    >>> q = random_quaternion(numpy.random.random(3))
    >>> len(q.shape), q.shape[0]==4
    (1, True)
    """
    if rand is None:
        rand = np.random.rand(3)
    else:
        assert len(rand) == 3
    r1 = np.sqrt(1.0 - rand[0])
    r2 = np.sqrt(rand[0])
    pi2 = np.pi * 2.0
    t1 = pi2 * rand[1]
    t2 = pi2 * rand[2]
    return np.array([np.cos(t2)*r2, np.sin(t1)*r1,
                        np.cos(t1)*r1, np.sin(t2)*r2])


def quaternion_about_axis(angle, axis):
    """Return quaternion for rotation about axis.
    >>> q = quaternion_about_axis(0.123, [1, 0, 0])
    """
    q = np.array([0.0, axis[0], axis[1], axis[2]])
    qlen = LA.norm(q)
    if qlen > (np.finfo(float).eps * 4.0):
        q *= np.sin(angle/2.0) / qlen
    q[0] = np.cos(angle/2.0)
    if angle == 0:
        q[0] = 0
    return q


def quaternion_matrix(quaternion):
    """Return homogeneous rotation matrix from quaternion.
    >>> M = quaternion_matrix([0.99810947, 0.06146124, 0, 0])
    >>> M = quaternion_matrix([1, 0, 0, 0])
    """
    q = np.array(quaternion, dtype=np.float64, copy=True)
    n = np.dot(q, q)
    if n < (np.finfo(float).eps * 4.0):
        return np.identity(4)
    q *= np.sqrt(2.0 / n)
    q = np.outer(q, q)
    return np.array([
        [1.0-q[2, 2]-q[3, 3],     q[1, 2]-q[3, 0],     q[1, 3]+q[2, 0], 0.0],
        [    q[1, 2]+q[3, 0], 1.0-q[1, 1]-q[3, 3],     q[2, 3]-q[1, 0], 0.0],
        [    q[1, 3]-q[2, 0],     q[2, 3]+q[1, 0], 1.0-q[1, 1]-q[2, 2], 0.0],
        [                0.0,                 0.0,                 0.0, 1.0]])

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



def normalize_homogenous_coordinate(np_array):
    # the input np.array should have columns that are homogenous n-vectors
    # this function will divide all of the elements in a column by the 
    # last element of that column
    last_row = np_array.shape[0] - 1
    for col in range(np.shape(np_array)[1]):
        if np_array[last_row, col] == 0:      # Skip points at infinity
            continue
        np_array[:, col] = np_array[:, col] / float(np_array[last_row, col])
        if np_array[last_row, col] == 0:
            print np_array[:,col]
    return np_array

def is_normalized(np_array):
    # Returns true if the last element of every column is equal to 1
    last_row = np_array.shape[0] - 1
    cols = np_array.shape[1]
    all_ones = True
    for col in range(cols):
        if np_array[last_row, col] != 1:
            all_ones = False
    return all_ones


def update_camera_matrix(display=True):
    # This function will calculate a camera matrix for a camera aiming at the origin from a given position
    global camera_matrix, camera_x, camera_y, camera_z, f, px, py, camera_pose_visual_pts, cube_pts, ax
    cam_center = np.array([[camera_x, camera_y, camera_z]])
    print 'Camera Pose: ', cam_center
    z_axis = np.array([[0,0,1]])
    rot_axis = np.cross(z_axis, cam_center) # DBG removed cam_center negative
    # print np.vdot(z_axis, cam_center)
    # print LA.norm(cam_center)
    cam_origin_dist = LA.norm(cam_center) # used to eliminate singularity in rotation angle calculation
    rot_angle = 0
    if cam_origin_dist != 0:
        rot_angle = np.arccos(np.vdot(z_axis, cam_center) / cam_origin_dist)
    # print "rot_axis", rot_axis
    # print "rot_angle", rot_angle
    # print "quaternion", quaternion_about_axis(rot_angle, rot_axis[0])
    # print "identity quat" , quaternion_about_axis(0, [0,0,0])

    # TODO: determine why the quaternion selected  below negates the displacement magnitude in the z-direction
    cam_R = quaternion_matrix(quaternion_about_axis(rot_angle, rot_axis[0]))

    # create the points for a on screen graphic showing camera pose
    perspective_deform_matrix = np.array([
        [0.1,   0,   0, 0],
        [  0, 0.1,   0, 0],
        [  0,   0, 0.2, 0],
        [  0,   0, 0.9, 1] 
        ])
    camera_translation_matrix = np.array([
        [1, 0,   0, camera_x],
        [0, 1,   0, camera_y],
        [0, 0,   1, camera_z],
        [0, 0,   0,        1]
        ])
    # transform the original cube points and save the intermediate stages
    camera_visual_tf_stages = {
        'original_cube': cube_pts,
        'perspective_deformed': np.dot(perspective_deform_matrix, cube_pts),
        'orientation_adjusted': np.dot(cam_R, np.dot(perspective_deform_matrix, cube_pts)),
        'final_cube_visual'   : np.dot(camera_translation_matrix, np.dot(cam_R, np.dot(perspective_deform_matrix, cube_pts)))
        }
    camera_pose_visual_pts = camera_visual_tf_stages['final_cube_visual']
    camera_pose_visual_pts = normalize_homogenous_coordinate(camera_pose_visual_pts)
    # camera_pose_visual_pts[2,:] = -camera_pose_visual_pts[2,:] # DBG (Experimental) flip the camera visual about Z
    # print 'camera_pose_visual_pts: \n', camera_pose_visual_pts
    # for i in range(camera_pose_visual_pts.shape[2]):
    # camera_visual_plot = ax.scatter(camera_pose_visual_pts[0,:], camera_pose_visual_pts[1,:], camera_pose_visual_pts[2,:], color='r', marker='.')
    # plt.draw()
    # camera_visual_plot.remove()
    cam_R = cam_R[0:3,0:3]
    minusRC = -(np.dot(cam_R,cam_center.T))
    X_cam = np.concatenate((cam_R, minusRC), 1)
    X_cam = np.concatenate((X_cam, np.array([[0, 0, 0, 1]])))
    K =  np.array([
        [f, 0, px],
        [0, f, py],
        [0, 0,  1]
        ])
    K0 = np.concatenate((K, np.zeros((3, 1))), 1)
    camera_matrix = np.dot(K0, X_cam)
    # print 'camera_matrix: \n', camera_matrix
    if display == True:
        display3D()
    perspective_camera()


def tf_cube_points(display=True):
    global tf_matrix, cube_pts, cube_tf_pts, pts_at_infinity, pts_at_inf_tf, ax, colors
    # Apply transform to cube
    cube_tf_pts = np.dot(tf_matrix, cube_pts)
    # Make cube points homogenous
    cube_tf_pts = normalize_homogenous_coordinate(cube_tf_pts)
    # Apply transform to points at infinity
    pts_at_inf_tf = np.dot(tf_matrix, pts_at_infinity)
    # Make points at infinity homogenous
    for col in range(pts_at_inf_tf.shape[1]):     # avoid division by zero when making homogenous
        if pts_at_inf_tf[3, col] == 0:
            pts_at_inf_tf[3, col] = 0.00001
    pts_at_inf_tf = normalize_homogenous_coordinate(pts_at_inf_tf)
    # print "pts_at_inf_tf\n", pts_at_inf_tf
    if display == True:
        display3D()
    perspective_camera()


def display3D():
    global tf_matrix, cube_pts, cube_tf_pts, pts_at_infinity, pts_at_inf_tf, ax, colors
    points = None
    at_inf_plt_pts = None
    # print "cube_tf_pts[0,:]   ", cube_tf_pts[0,:]
    # print "cube_tf_pts[1,:]   ", cube_tf_pts[1,:]
    # print "cube_tf_pts[2,:]   ", cube_tf_pts[2,:]
    # print "pts_at_inf_tf[0,:] ", pts_at_inf_tf[0,:]
    # print "pts_at_inf_tf[1,:] ", pts_at_inf_tf[1,:]
    # print "pts_at_inf_tf[2,:] ", pts_at_inf_tf[2,:]
    # print "camera_pose_visual_pts[2,:] ", camera_pose_visual_pts[0,:]
    # print "camera_pose_visual_pts[1,:] ", camera_pose_visual_pts[1,:]
    # print "camera_pose_visual_pts[2,:] ", camera_pose_visual_pts[2,:]
    try:
        # Plot Transformed Cube Points and Points at infinity
        points = ax.scatter(cube_tf_pts[0,:], cube_tf_pts[1,:], cube_tf_pts[2,:], color=colors, marker='o')
        at_inf_plt_pts = ax.scatter(pts_at_inf_tf[0,:], pts_at_inf_tf[1,:], pts_at_inf_tf[2,:], color='r', marker='D')
        # plt.draw()
        # Plot cube lines
        num_lines_plotted = 0
        for pt1 in range(8):
            for pt2 in range(8):
                if abs(euclidean_dist(cube_pts[0:3,pt1], cube_pts[0:3,pt2]) - 2.0) > 0.1:
                    continue
                else:
                    # color = (
                    #     (colors[pt1][0] + colors[pt2][0]) / 2.0,
                    #     (colors[pt1][1] + colors[pt2][1]) / 2.0,
                    #     (colors[pt1][2] + colors[pt2][2]) / 2.0
                    #     )

                    # Determine color based on strech of transformed cube edge
                    length = euclidean_dist(cube_tf_pts[0:3,pt1],cube_tf_pts[0:3,pt2])
                    original_length = euclidean_dist(cube_pts[0:3,pt1],cube_pts[0:3,pt2])
                    strain = (length - original_length) / original_length
                    color_coeff = strain / 10 - 0.1
                    if color_coeff < 0:
                        color_coeff = 0.0
                    if color_coeff > 0.8:
                        color_coeff = 0.8
                    # print "color_coeff", color_coeff

                    # Plot Lines
                    tf_cube_lines = ax.plot(cube_tf_pts[0,[pt1,pt2]],cube_tf_pts[1,[pt1,pt2]],cube_tf_pts[2,[pt1,pt2]], color=plt.cm.hot(color_coeff) )
                    num_lines_plotted = num_lines_plotted + 1
        # print "plotted cube lines"

        # Plot the camera pose visual
        camera_visual_plot = ax.scatter(camera_pose_visual_pts[0,:], camera_pose_visual_pts[1,:], camera_pose_visual_pts[2,:], color='r', marker='.')
        # print "num_lines_plotted", num_lines_plotted
        for pt1 in range(8):
            for pt2 in range(8):
                if abs(euclidean_dist(cube_pts[0:3,pt1], cube_pts[0:3,pt2]) - 2.0) > 0.1:
                    continue
                else:
                    tf_cube_lines = ax.plot(camera_pose_visual_pts[0,[pt1,pt2]],camera_pose_visual_pts[1,[pt1,pt2]],camera_pose_visual_pts[2,[pt1,pt2]], color=(0, 1, 0))
                    num_lines_plotted = num_lines_plotted + 1
        # print "num_lines_plotted", num_lines_plotted
    except:
        print "Unexpected error while plotting:", sys.exc_info()[0]
        cv2.waitKey()
    try: 
        perspective_camera()
        plt.draw()
    except:
        print "Unexpected error:", sys.exc_info()[0], "in display3D()"
        # if True:
        #     blahblah =  ax.scatter(cube_tf_pts[0,:], cube_tf_pts[1,:], cube_tf_pts[2,:], color=colors, marker='o')
        # blahblah.remove()
        # points.remove()
        # at_inf_plt_pts.remove()
        # camera_visual_plot.remove()
        # while len(ax.lines) != 0:
        #     ax.lines.pop(0)
    finally:
        points.remove()
        at_inf_plt_pts.remove()
        camera_visual_plot.remove()
        while len(ax.lines) != 0:
            ax.lines.pop(0)
    


def perspective_camera():
    global camera_matrix, cube_pts, cube_tf_pts, pts_at_inf_tf, colors, px, py
    cube_img_pts = np.dot(camera_matrix, cube_tf_pts)
    if not is_normalized(cube_img_pts):
        cube_img_pts = normalize_homogenous_coordinate(cube_img_pts)
    # for i in range(cube_img_pts.shape[1]):
    #     cube_img_pts[0,i] = cube_img_pts[0,i] / cube_img_pts[2,i]
    #     cube_img_pts[1,i] = cube_img_pts[1,i] / cube_img_pts[2,i]
    # print "img points after for loop\n", cube_img_pts[:,0]
    cube_img_pts = cube_img_pts[[0,1],:]
    # print cube_img_pts
    # print "shape", cube_img_pts.shape
    img_shape = (200,200)
    img = np.zeros(img_shape)
    for i in range(cube_img_pts.shape[1]):
        # print "color", colors[i]
        x = cube_img_pts[0, i]
        y = cube_img_pts[1, i]
        if isinf(x) or isinf(y):
            continue
        # img_point = (10*(-px + int(x)), 10*(-py + int(y)))
        img_point = (int(100 * x) + img_shape[1]/2, int(100 * y) + img_shape[0]/2)
        print "img_point", img_point
        cv2.circle(img, img_point, 3, (255,255,255), -1) #colors[i]
    cv2.imshow("Camera Position Control and Image", img)



def cb1(value):
    global tf_matrix
    print "cb1"
    tf_matrix[0,0] = ( value - (slider_max*0.5)) / 10.0
    tf_cube_points()

def cb2(value):
    global tf_matrix
    tf_matrix[0,1] = ( value - (slider_max*0.5)) / 10.0
    tf_cube_points()

def cb3(value):
    global tf_matrix
    tf_matrix[0,2] = ( value - (slider_max*0.5)) / 10.0
    tf_cube_points()

def cb4(value):
    global tf_matrix
    tf_matrix[0,3] = ( value - (slider_max*0.5)) / 10.0
    tf_cube_points()

def cb5(value):
    global tf_matrix
    tf_matrix[1,0] = ( value - (slider_max*0.5)) / 10.0
    tf_cube_points()

def cb6(value):
    global tf_matrix
    tf_matrix[1,1] = ( value - (slider_max*0.5)) / 10.0
    tf_cube_points()

def cb7(value):
    global tf_matrix
    tf_matrix[1,2] = ( value - (slider_max*0.5)) / 10.0
    tf_cube_points()

def cb8(value):
    global tf_matrix
    tf_matrix[1,3] = ( value - (slider_max*0.5)) / 10.0
    tf_cube_points()

def cb9(value):
    global tf_matrix
    tf_matrix[2,0] = ( value - (slider_max*0.5)) / 10.0
    tf_cube_points()

def cb10(value):
    global tf_matrix
    tf_matrix[2,1] = ( value - (slider_max*0.5)) / 10.0
    tf_cube_points()

def cb11(value):
    global tf_matrix
    tf_matrix[2,2] = ( value - (slider_max*0.5)) / 10.0
    tf_cube_points()

def cb12(value):
    global tf_matrix
    tf_matrix[2,3] = ( value - (slider_max*0.5)) / 10.0
    tf_cube_points()

def cb13(value):
    global tf_matrix
    tf_matrix[3,0] = ( value - (slider_max*0.5)) /50.0
    tf_cube_points()

def cb14(value):
    global tf_matrix
    tf_matrix[3,1] = ( value - (slider_max*0.5)) /50.0
    tf_cube_points()

def cb15(value):
    global tf_matrix
    tf_matrix[3,2] = ( value - (slider_max*0.5)) /50.0
    tf_cube_points()

def cb16(value):
    global tf_matrix
    tf_matrix[3,3] = 1 + ( value - (slider_max*0.5)) /50.0
    tf_cube_points()


def cam_cbx(value):
    global camera_x
    camera_x = (value-50)/5.0
    update_camera_matrix()

def cam_cby(value):
    global camera_y
    camera_y = (value-50)/5.0
    update_camera_matrix()

def cam_cbz(value):
    global camera_z
    camera_z = (value-50)/5.0
    update_camera_matrix()




# Global Geometric Objects
tf_matrix = np.array([                   # 4x4 homogenous tf matrix
    [1, 0, 0, 0],
    [0, 1, 0, 0],
    [0, 0, 1, 0],
    [0, 0, 0, 1.0]
    ])
cube_pts = np.array([                    # original homogenous point coordinates
    [1, -1,  1,  1, -1,  1, -1,   -1],   # x
    [1,  1, -1,  1, -1, -1,  1,   -1],   # y
    [1,  1,  1, -1,  1, -1, -1,   -1],   # z
    [1,  1,  1,  1,  1,  1,  1,  1.0]    # homogenous coordinate
    ])
pts_at_infinity = np. array([
    [1, 0, 0],
    [0, 1, 0],
    [0, 0, 1],
    [0, 0, 0]
    ])
camera_pose_visual_pts = np.ones_like(cube_pts)
cube_tf_pts = np.ones_like(cube_pts)
pts_at_inf_tf = np.ones_like(pts_at_infinity)


# User Inteface
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
cv2.namedWindow("Camera Position Control and Image")
cv2.createTrackbar("camera_x", "Camera Position Control and Image", slider_start, slider_max, cam_cbx)
cv2.createTrackbar("camera_y", "Camera Position Control and Image", slider_start, slider_max, cam_cby)
cv2.createTrackbar("camera_z", "Camera Position Control and Image", slider_start, slider_max, cam_cbz)

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

# camR = quaternion_matrix(random_quaternion())
camera_x = 5; camera_y = 5; camera_z = 0
camera_matrix = np.zeros((4, 4))
f = 1    # focal length
px = 0  # principal point offset in x
py = 0  # principal point offset in y
update_camera_matrix(False)
tf_cube_points(False)
display3D()

cv2.waitKey(0)