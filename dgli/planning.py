from math import ceil, floor
import numpy as np
import copy

from dgli import loadData
from dgli.dGLI_class import dGLI_class, includeEdges

def get_linear_interpolation(featureVector_1, featureVector_2, num_steps=10):
    # featureVector_i[0] does not change. We only need to linearly interpolate FV[1]
    # Ensure the feature vectors are from the same example
    if (featureVector_1[0] != featureVector_2[0]).all():
        raise ValueError("Dude! Corners don't match!")
    interpolatered_edges = interpolate_arrays_circular(featureVector_1[1], featureVector_2[1], num_steps)
    linear_interpolation = []
    for i in range(len(interpolatered_edges)):
        featureVector = copy.deepcopy(featureVector_1)
        featureVector[1] = interpolatered_edges[i]
        linear_interpolation.append(featureVector)
    return linear_interpolation

def get_distance(loc_0, loc_1):
    if (loc_1 > loc_0):
        return loc_1 - loc_0
    else:
        return 2*np.pi - (loc_0 - loc_1)

def get_interpolation(point_0, point_1, num_steps, counterClockwise=True):
    t_values = np.linspace(0, 1, num_steps)
    
    if counterClockwise:
        if point_1 > point_0:
            interpolation = (1 - t_values) * point_0 + t_values * point_1
        else:
            point_1 += 2 * np.pi
            interpolation = (1 - t_values) * point_0 + t_values * point_1
    else:
        if point_1 < point_0:
            interpolation = (1 - t_values) * point_0 + t_values * point_1
        else:
            point_1 -= 2 * np.pi
            interpolation = (1 - t_values) * point_0 + t_values * point_1
    
    return np.mod(interpolation, 2 * np.pi)

def interpolate_arrays_circular(fold_1, fold_2, num_steps=10): #Returns a list of arrays
    # normalize the folds to [0,2pi]
    fold_1 %= (2*np.pi)
    fold_2 %= (2*np.pi)
    # get length of fold_1
    length_fold_1 = get_distance(fold_1[0], fold_1[1])
    
    # check movement. If lower has covered more than length_fold_1 change direction of lower
    movement_1  =  get_distance(fold_1[1], fold_2[1])
    movement_0  =  get_distance(fold_1[0], fold_2[0])
    
    # choose direction
    # The direction in which movement is minimum will be chosen
    total_movement = (movement_0 + movement_1)
    if total_movement<(4*np.pi-total_movement):
        # default direction is anti-clockwise
        if movement_0 - movement_1 > length_fold_1:
            print("0 changes direction")
            path_0 = get_interpolation(fold_1[0], fold_2[0], num_steps, False)
            path_1 = get_interpolation(fold_1[1], fold_2[1], num_steps, True)
        elif movement_1 - movement_0 > 2*np.pi-length_fold_1:
            print("1 changes direction")
            path_0 = get_interpolation(fold_1[0], fold_2[0], num_steps, True)
            path_1 = get_interpolation(fold_1[1], fold_2[1], num_steps, False)
        else:
            path_0 = get_interpolation(fold_1[0], fold_2[0], num_steps, True)
            path_1 = get_interpolation(fold_1[1], fold_2[1], num_steps, True)
        
        
        
    else:
        print("we move clockwise")
        movement_0 -= 2*np.pi
        movement_1 -= 2*np.pi
        if movement_0 - movement_1 > length_fold_1:
            print("1 changes direction")
            path_1 = get_interpolation(fold_1[1], fold_2[1], num_steps, True)
            path_0 = get_interpolation(fold_1[0], fold_2[0], num_steps, False)
        elif movement_1 - movement_0 > 2*np.pi-length_fold_1:
            print("1 changes direction")
            path_1 = get_interpolation(fold_1[1], fold_2[1], num_steps, False)
            path_0 = get_interpolation(fold_1[0], fold_2[0], num_steps, True)
        else:
            path_1 = get_interpolation(fold_1[1], fold_2[1], num_steps, False)
            path_0 = get_interpolation(fold_1[0], fold_2[0], num_steps, False)
        
        
    
    return np.array([path_0, path_1]).T
# inter  = interpolate_arrays_circular(np.array([0.28443517, 2.70626886]), np.array([5.84312693, -1.85132905]))
# print(inter)
        
def interpolate_arrays_circular__(arr1, arr2, num_steps=10): #Returns a list of arrays
    # These are circular arrays so we need to go the next point such that it's easy to reach it considering we are on a circle
    # For now see if shortest path works - Check if this is a sufficient condition
    arr1 %= (2*np.pi)
    arr2 %= (2*np.pi)
    diff = (arr2-arr1)
    if is_clockwise(arr1)==is_clockwise(arr2):
        pass
        # If same orientation prefer shortest path always
        # Prefer shortest path
        for i in range(len(arr1)):
            if abs(diff[i])>np.pi:
                print(f"you need to change direction for {i}")
                if (diff[i]>0):
                    arr2[i] -= 2*np.pi
                else:
                    arr2[i] += 2*np.pi
        print(arr1, arr2)
        # However, if orientation of one of the vectors change, it is better to go the longer way!
        # check for orientation
    else:
        # distance must keep increasing. So check which diff is larger
        print("distance must keep increasing")
        if (diff[0]-diff[1])>0:
            idx = 1 # this changes direction
        else:
            idx = 0
            
        if arr2[idx]-arr1[idx] > 0: #clockwise
            # make it anti-clockwise
            arr2[idx] -= 2*np.pi
        else:
            # make it clockwise
            arr2[idx] += 2*np.pi
               
    return [(1 - t) * arr1 + t * arr2 for t in np.linspace(0, 1, num_steps)]
    

def is_clockwise(array):
    return (True if((array[1]-array[0]<=np.pi and array[1]-array[0]>=0) or array[1]-array[0]<-np.pi) else False)

def get_foldPoints(border, fold_line): #points in 3d space where the fold starts and ends
    # Get the indices of the points on the border along which to rotate
    loc1 = fold_line[0]/(2*np.pi)*len(border)
    loc2 = fold_line[1]/(2*np.pi)*len(border)
    #points in 3d space where the fold starts and ends
    p1 = border[floor(loc1)%len(border)] + loc1%1*(border[ceil(loc1)%len(border)] - border[floor(loc1)%len(border)])
    p2 = border[floor(loc2)%len(border)] + loc2%1*(border[ceil(loc2)%len(border)] - border[floor(loc2)%len(border)])
    return p1, p2, loc1, loc2

def fold(border, fold_line):    

    p1, p2, loc1, loc2 = get_foldPoints(border, fold_line)
    m, c = getPlane(p1, p2)

    idx1 = ceil(loc1)%len(border)
    idx2 = ceil(loc2)%len(border)

    # fold (reflect)
    if idx1>idx2:
        idx2 += len(border)
        # idx1, idx2 = idx2, idx1
        # print(idx1, idx2)
    for i in range(idx1, idx2):
        # print(i%len(border))
        # if (i%len(border)==0):
        #     continue
        reflect_3d_array(border[i%len(border)], m, c)
    
    return np.array([p1, p2])

def fold_(border, fold_line):    

    p1, p2, loc1, loc2 = get_foldPoints(border, fold_line)
    m, c = getPlane(p1, p2)

    idx1 = ceil(loc1)%len(border)
    idx2 = ceil(loc2)%len(border)

    # fold (reflect)
    if idx1>idx2:
        idx2 += len(border)
    for i in range(idx1, idx2):
        if (i%len(border)==0):
            continue
        reflect_3d_array(border[i%len(border)], m, c)
    
    return np.array([p1, p2])

def get_affected_corners(vec, folds):
    vec = vec.tolist()
    folds = folds.tolist()
    if folds[0] < folds[1]:
        folded_corners = [v for v in vec if folds[0] < v < folds[1]]
    else:
        folded_corners = list(set(vec) - {v for v in vec if folds[1] < v < folds[0]})
    return (np.array(folded_corners))

def getPlane(p1, p2):
    m = (p2[1]-p1[1])/(p2[0]-p1[0]) # y2-y1 / x2-x1
    c = p2[1] - m*p2[0] # y2-m*x2
    # print("m = ", m, ", c = ", c)
    return m,c

def reflect_3d_array(array, m, c):
    if array.shape[0] != 3:
        raise ValueError("Input array must have shape (3, )")
    
    x, y = array[0], array[1]
    
    # Calculate reflection using formula
    denom = 1 + m**2
    array[0] = ((1 - m**2) * x + 2 * m * (y - c)) / denom
    array[1] = ((m**2 - 1) * y + 2 * m * x + 2 * c) / denom
    
# Functions for calculating area of trapezoid
import math

def sort_points_counterclockwise(points):
    """
    Sorts the four points in counterclockwise order based on the centroid.
    """
    # Compute centroid (average of x and y coordinates)
    cx = sum(x for x, y in points) / 4
    cy = sum(y for x, y in points) / 4
    
    # Sort points based on angle from centroid
    points.sort(key=lambda p: math.atan2(p[1] - cy, p[0] - cx))
    return points

def trapezoid_area(points):
    """
    Calculate the area of a quadrilateral using the Shoelace Theorem.
    :param points: List of four (x, y) tuples
    :return: Enclosed area
    """
    if len(points) != 4:
        raise ValueError("Four points are required to form a quadrilateral.")

    # Ensure points are sorted counterclockwise
    sorted_points = sort_points_counterclockwise(points)

    x1, y1 = sorted_points[0]
    x2, y2 = sorted_points[1]
    x3, y3 = sorted_points[2]
    x4, y4 = sorted_points[3]

    # Apply Shoelace formula
    area = 0.5 * abs(
        (x1 * y2 + x2 * y3 + x3 * y4 + x4 * y1) -
        (y1 * x2 + y2 * x3 + y3 * x4 + y4 * x1)
    )
    
    return area

def get_control_points_and_goal(filePath, foldLine, same_category):
    borders,_ = loadData.loadAllFrames(filePath)
    Eb = loadData.get_border_mesh_mapping(filePath) # Mapping of border to mesh

    dGLI_ = dGLI_class(includeEdges.ALL, borders)
    start = dGLI_.get_frames() -1
    dGLI_.set_start_frame(start)
    # foldLine = np.array([2.66928204, 6.42257836])

    # The code starts here...
    if same_category:
        # unfold and fold
        print("same")
        folded_border = copy.deepcopy(borders[-1]) #Copy the last border 
        dGLI_.add_border(folded_border)
        fold_points = fold(folded_border, foldLine)
    else:
        print("not same")
        folded_border = copy.deepcopy(borders[-1]) #Copy the last border 
        dGLI_.add_border(folded_border)
        fold_points = fold(folded_border, foldLine)

    # Get corners to be affected by the fold
    corners = dGLI_.get_corners()

    # Corners that will be affected by the fold
    toFoldCorners = get_affected_corners(corners, foldLine)

    # Choose which corner(s) to fold
    # The way you choose the corners is by choosing 2 points on the corner such that the area of the trapezoid is maximum
    # So I need to project the 3d points on the xy plane (plane of the table) and then calculate the area

    def get_corners_to_manipulate(toFoldCorners, fold_points):
        if len(toFoldCorners)<=2:
            return toFoldCorners
        else:
            # Return corners that subtend max area
            max_area=0
            points = fold_points[:, :-1]
            p1 = points[0].tolist()
            p2 = points[1].tolist()
            for i in range(len(toFoldCorners)):
                for j in range(i+1, len(toFoldCorners)):
                    a = dGLI_.get_border(start)[dGLI_.get_vertex_index(toFoldCorners[i])]
                    a = a[:-1].tolist()
                    b = dGLI_.get_border(start)[dGLI_.get_vertex_index(toFoldCorners[j])]
                    b = b[:-1].tolist()
                    area = trapezoid_area([a, b, p1, p2])
                    if max_area < area:
                        max_area = area
                        cornersIdx = [i,j]
            return [toFoldCorners[cornersIdx[0]], toFoldCorners[cornersIdx[1]]]
            
    # planning.manipulateCorners(toFoldCorners, )
    man_corners = get_corners_to_manipulate(toFoldCorners, fold_points)

    goals = []
    controlPoints = []
    for i in range(len(man_corners)):
        controlPoints.append(Eb[(dGLI_.get_vertex_index(man_corners[i])+1)%dGLI_.get_edges()])
        goals.append(folded_border[(dGLI_.get_vertex_index(man_corners[i])+1)%dGLI_.get_edges()])

    return goals, controlPoints