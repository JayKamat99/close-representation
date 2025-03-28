import numpy as np
from numpy.linalg import norm
from enum import Enum
from sklearn.cluster import DBSCAN
from dgli.curveFitting import fit_poly_curve, circular_euclidean_distance

class includeEdges(Enum):
  ALL = "all"
  ALTERNATE = "alternate"
  CORNERS = "corners"
  
class dGLI_class:
  def __init__(self, includeEdges, borders=None):
    """Initialize the class"""
    self.__borders = borders if borders is not None else []
    self.__frames = len(self.__borders)
    self.__includeEdges = includeEdges
    self.__edges = self.__countEdges()
    self.__add_noise = False
    self.__noise_mean = 0
    self.__noise_sd = 0.007
    self.__threshold = 2
    self.__startFrame = 0 # Unless changed this remains as 0

  ## Public Methods ##
  
  def get_dGLICoords(self, curveIdx=-1, sideSize = 20):
    if self.__includeEdges is includeEdges.ALTERNATE:
      subCurveEdges= self.__getAlternateEdges(self.__borders[curveIdx])
    elif self.__includeEdges is includeEdges.ALL:
      subCurveEdges= self.__getAllEdges(self.__borders[curveIdx])
    else:
      raise NotImplementedError("This method has been depritiated")
    return self.__dGLICoords(subCurveEdges)
  
  def get_edges(self):
    return self.__edges
  
  def get_frames(self):
    return self.__frames
  
  def get_border(self, idx):
    return self.__borders[idx]
  
  def get_location(self, i, j):
    i, j = self.__ensure_ij(i, j) 
    return (i*self.__edges - i*(i+1)//2 + j -i - 1)
  
  def get_polar_coordinates(self, i, j):
    i, j = self.__ensure_ij(i, j)
    return self.__get_radius(i, j), self.__get_theta(i, j) 
    # raise NotImplementedError()
  
  def get_coordinates(self, idx):
    sum = int(self.__edges*(self.__edges-1)//2)
    i=2
    sum = sum-1 #This is the last element
    if (idx>sum):
        raise IndexError("Index out of bounds")
    while (idx<sum):
        sum -= i
        i+=1
    return int(self.__edges-i), int(self.__edges-i+idx-sum+1)
  
  def get_dGLI_diff(self,endFrame, startFrame, withSign = False):
    difference = np.abs(np.abs(self.get_dGLICoords(endFrame))-np.abs(self.get_dGLICoords(startFrame)))
    if (withSign):
      signed_diff = np.abs(np.array(self.get_dGLICoords(endFrame)) - np.array(self.get_dGLICoords(startFrame)))
      # get indices of diff_array
      indices = np.array(self.__get_indices_above_threshold(difference))
      for idx in indices:
        signed_diff[idx] = 0
      return signed_diff
      # return (np.abs(np.array(self.get_dGLICoords(endFrame)) - np.array(self.get_dGLICoords(startFrame))) - difference)
    return difference
  
  def get_corners(self): # gives corners location in radians. This functaion is called by get_featureVector
    borderCells = np.zeros(self.__edges)
    dGLI_coords_start = self.get_dGLICoords(self.__startFrame)
    for i in range(self.__edges-1):
      borderCells[i] = dGLI_coords_start[i*self.__edges - ((i*(i+1))//2)]
    borderCells[self.__edges-1] = dGLI_coords_start[self.__edges-2]

    # Count edges and save their position in radians
    corners = []
    self.__calc_threshold(dGLI_coords_start) # updates the threshold for the given array
    for i in range(len(borderCells)):
      if abs(borderCells[i]) > self.__threshold:
        corners.append((((i+0.5)/len(borderCells))*(2*np.pi))%(2*np.pi))
    
    return np.array(corners)
  
  def get_vertex_index(self, angle):
    idx = int(angle*self.__edges/(2*np.pi) - 0.5)
    return idx
  
  def get_fold(self, endFrame, startFrame):
    # get the curve
    curves = self.__get_fold_curves(endFrame, startFrame)
    # foldBorder = self.__get_fold_end_points(endFrame, startFrame)
    # oriented_endPoints = self.__get_orientedEndPoints(foldBorder, endFrame, startFrame)
    # verify and get orientation
    self.__get_oriented_curves(curves, endFrame, startFrame)
    oriented_endPoints = np.array([(curves[0][1][-1])%(2*np.pi), (curves[1][1][-1])%(2*np.pi)])
    return oriented_endPoints #numpy.ndarray
  
  def add_border(self, border):
    # Before adding, verify if it belongs to the cloth by matching the edges
    if self.__edges is not self.__countEdges(border):
      raise TypeError("Number of edges do not match")
    if self.__add_noise: #If you are supposed to add noise, do so
      self.__add_noise_to_border(border)
    self.__borders.append(border)
    self.__frames += 1

  def add_noise(self, mean=0, sd=0.007):
    self.__add_noise=True
    self.__noise_mean = mean
    self.__noise_sd = sd
    self.__add_noise_to_borders()
    
  def get_high_value_points(self, array, scaling_factor = 1):
    brightPoints = []
    self.__calc_threshold(array)
    for idx in range(len(array)):
      if array[idx]>self.__threshold*scaling_factor:
        i,j = self.get_coordinates(idx)
        point = self.get_polar_coordinates(i,j)
        brightPoints.append(point)
    return brightPoints
    
  def get_threshold(self):
    return self.__threshold
  
  def get_max_point(self, endFrame, startFrame):
    i, j = self.get_coordinates(np.argmax(self.get_dGLI_diff(endFrame,startFrame, True)))
    return self.get_polar_coordinates(i,j)
  
  def get_feature_vector(self, endFrame, startFrame):
    feature_vector = [self.get_corners()]
    feature_vector.append(self.get_fold(endFrame, startFrame))
    return feature_vector
  
  def get_indexFraction_from_polar(self, angle):
    loc = (angle/(2*np.pi))*len(self.get_border(self.__startFrame))
    return loc
  
  def set_start_frame(self, startFrame):
    self.__startFrame = startFrame
    
  def set_end_frame(self, endFrame):
    self.__endFrame = endFrame
  
  ## Private Methods##
  
  def __get_indices_above_threshold(self, array):
    indices = []
    self.__calc_threshold(array)
    for idx in range(len(array)):
      if array[idx]>self.__threshold:
        indices.append(idx)
    return indices
  
  def __calc_threshold(self, array):
    # threshold = np.std(array)
    mean = np.mean(array)
    std_dev = np.std(array)
    # Compute Z-scores
    z_scores = (array - mean) / std_dev
    # Keep values within ±3 standard deviations
    filtered_data = array[np.abs(z_scores) < 7]
    threshold = np.std(filtered_data)
    self.__threshold=threshold
  
  def __ensure_ij(self, i, j):
    # Ensure i and j are in the required range
    i = i%self.__edges
    j = j%self.__edges
    if i==j:
      raise ValueError("Invalid Query, values of i and j are the same")  
    # Ensure i<j
    if j<i:
      i,j = j,i
    return i, j

  def __add_noise_to_borders(self):
    for border in self.__borders:
      self.__add_noise_to_border(border)
  
  def __add_noise_to_border(self, border):
    for idx in range(len(border)):
      border[idx] = self.__add_gaussian_noise(border[idx])
  
  def __add_gaussian_noise(self, arr):
    noise = np.random.normal(self.__noise_mean, self.__noise_sd, arr.shape)
    return arr + noise
  
  def __get_diff(self, i, j):
    diff = min(j-i, self.__edges+i-j)
    return diff
  
  def __get_radius(self, i, j):
    num_levels = int(self.__edges/2)
    diff = self.__get_diff(i,j)
    # r = (2*num_levels-(2*diff-1))/(2*num_levels)
    r = (num_levels-(diff-1))/num_levels
    return r
  
  def __get_theta(self, i, j):
    offset = 0 #This is done to match the image
    cell = i+j
    if j-i > self.__get_diff(i,j):
      cell += self.__edges
    theta = (cell) * np.pi / self.__edges + offset
    return theta%(2*np.pi) # Returns value in radians (make sure this is within [0,2*pi))
  
  def __get_fold_end_points(self, endFrame, startFrame):
    diff_points = self.get_high_value_points(self.get_dGLI_diff(endFrame, startFrame), scaling_factor=1)

    X = np.array(diff_points)
    X[:,1] = X[:,1]%(2*np.pi)

    dbscan = DBSCAN(eps=0.3, min_samples=int(self.get_edges()/2), metric=circular_euclidean_distance)  # Adjust eps based on dataset
    labels = dbscan.fit_predict(X)

    # Find unique cluster labels (excluding noise, which is labeled as -1)
    unique_labels = set(labels)  # Includes noise (-1) if present
    unique_labels.discard(-1)  # Remove noise if you want only valid clusters

    # Separate X into clusters as a list of numpy arrays
    clusters = [X[labels == label] for label in unique_labels]
    
    # Take the last layer / last few layers from the cluster and find the mean. use that theta
    # For now, I take 2 
    
    max_second_max_points = [
      cluster[np.isin(cluster[:, 1], np.unique(cluster[:, 1])[-2:])]
      for cluster in clusters
    ]
    
    max_points = [cluster[cluster[:, 1] == cluster[:, 1].max()] for cluster in clusters]

    
    foldBorder = []
    
    for blob in max_points:
      if abs(blob[0,1]%(2*np.pi)-np.pi)>(np.pi-np.pi/6): #check only the first value
        # add np.pi to all values
        for i in range(len(blob)):
          blob[i][1] += np.pi
          blob[i][1] %= (2*np.pi)
        av = np.mean(blob, axis=0)
        av[1] = av[1] - np.pi
        foldBorder.append((av[1]%(2*np.pi)))
      else:
        av = np.mean(blob, axis=0)
        foldBorder.append(np.array([1,(av[1]%(2*np.pi))]))
    
    return foldBorder
  
  def __get_fold_curves(self, endFrame, startFrame):
    diff_points = self.get_high_value_points(self.get_dGLI_diff(endFrame, startFrame), scaling_factor=1)

    X = np.array(diff_points)
    X[:,1] = X[:,1]%(2*np.pi)

    dbscan = DBSCAN(eps=0.3, min_samples=int(self.get_edges()/4), metric=circular_euclidean_distance)  # Adjust eps based on dataset
    labels = dbscan.fit_predict(X)

    # Find unique cluster labels (excluding noise, which is labeled as -1)
    unique_labels = set(labels)  # Includes noise (-1) if present
    unique_labels.discard(-1)  # Remove noise if you want only valid clusters

    # Separate X into clusters as a list of numpy arrays
    clusters = [X[labels == label] for label in unique_labels]

    # Fit polynomial curve of degree
    degree = 1
    curves = []
    for i, cluster in enumerate(clusters):
      curve = fit_poly_curve(cluster, degree=degree)
      curves.append(curve)
         
    return curves
  
  def __get_orientedEndPoints(self, endPoints, endFrame, startFrame):
    # How many end curves do I get? If not 2 just exit!
    if len(endPoints) != 2:
      raise ValueError("detected wrong numer of curves: ", len(endPoints))

    # get the rotated side. TO know the rotated side we need to get the difference between the dGLI_end and the dGLI_diff
    max_point = self.get_max_point(endFrame, startFrame)
      
    points = np.array(endPoints)
    # print(points)

    # check if the max_point lies between the two points on our curve
    # For this I only need to check for the thetas
    theta0 = points[0][1]
    theta1 = points[1][1]
    theta = max_point[1]
    if (theta1 < theta0):
      if (theta<theta1):
        theta += 2*np.pi
      theta1 += 2*np.pi

    # Now check if theta lies between theta0 and theta1
    if (theta0<theta) and (theta<theta1):
      pass #This is the right arrangement
    else:
      endPoints[0], endPoints[1] = endPoints[1], endPoints[0]
      # print("inverted!")
    return(np.array([endPoints[0][1], endPoints[1][1]]))
  
  def __get_oriented_curves(self, curves, endFrame, startFrame):
    # How many end curves do I get? If not 2 just exit!
    if len(curves) != 2:
      raise ValueError("detected wrong numer of curves: ", len(curves))

    # get the rotated side. TO know the rotated side we need to get the difference between the dGLI_end and the dGLI_diff
    max_point = self.get_max_point(endFrame, startFrame)

    # get the radius of the max_point (usually it is one but we never know)
    r = max_point[0]
    idx = round(r*len(curves[0][0]))-1 # We do this to go to the level of the max_point

    points = []
    for curve in curves:
      points.append(np.array([curve[0][idx], curve[1][idx]]))
      
    points = np.array(points)
    # print(points)

    # check if the max_point lies between the two points on our curve
    # For this I only need to check for the thetas
    theta0 = points[0][1]
    theta1 = points[1][1]
    theta = max_point[1]
    if (theta1 < theta0):
      if (theta<theta1):
        theta += 2*np.pi
      theta1 += 2*np.pi

    # Now check if theta lies between theta0 and theta1
    if (theta0<theta) and (theta<theta1):
      pass #This is the right arrangement
    else:
      curves[0], curves[1] = curves[1], curves[0]
      # print("inverted!")
  
  def __countEdges(self, curve=None):
    if curve is None:
      curve = self.__borders[0]
    if self.__includeEdges == includeEdges.ALTERNATE:
      return int(len(curve)//2)
    elif self.__includeEdges == includeEdges.ALL:
      return len(curve)
    else: #It will be Corners
      return 8
    
  def __getAllEdges(self, curve):
    outCurve = []
    # close curve
    if (curve[0] != curve[-1]).all():
      curve.append(curve[0])
    if self.__edges != len(curve)-1:
      curve.append(curve[0])
    for ind in range(len(curve)-1):
      outCurve.append([curve[ind], curve[ind+1]])
    return outCurve
 
  def __getAlternateEdges(self, curve):
    outCurve = []
    # close curve
    if not (curve[0] == curve[-1]).all():
      curve.append(curve[0])
    for ind in range(int((len(curve)-1)/2)):
      outCurve.append([curve[2*ind], curve[2*ind+1]])
    return outCurve
  
  # edgesCurve contains a list of edges, that is, tuples {p1,p2} of 3D points
  def __dGLICoords(self, edgesCurve):
    numP1 = len(edgesCurve)
    numP2 = len(edgesCurve)
    
    vector_res = []

    for i in range(numP1):
      for j in range(i, numP2):
        if i!=j:
          dgli = self.__dGLI_calc(edgesCurve[i][0], edgesCurve[i][1], edgesCurve[j][0], edgesCurve[j][1])
          vector_res.append(dgli)

    return np.array(vector_res)
    
  def __dGLI_calc(self, ap1, ap2, ap3, ap4):
    epsilon = 0.00001
    def perturb_coords(a1, a2, a3, a4):
      p1 = np.add(a1 ,[0, 0, 0])
      p2 = np.add(a2, [0, 0, epsilon])
      p3 = np.add(a3, [0, 0, 0])
      p4 = np.add(a4, [0, 0, epsilon])
      return p1, p2, p3, p4
    
    p1, p2, p3, p4 = perturb_coords(ap1, ap2, ap3, ap4)
    ap1, ap2, ap3, ap4 = p1, p2, p3, p4
    p1, p2, p3, p4 = perturb_coords(ap1, ap2, ap3, ap4)
    
    # # if we are evaluating the same segment
    # if np.linalg.norm(np.subtract(ap1,ap3)) < 0.01 and np.linalg.norm(np.subtract(ap2,ap4)) < 0.01:
    #   ret = 10
    # else: #In my experience we never evaluate the same segment
    gli_ = self.__evalWritheTwoSegments(ap1, ap2, ap3, ap4)
    
    gli_Perturbed_ = self.__evalWritheTwoSegments(p1, p2, p3, p4)
    
    ret = (gli_Perturbed_ - gli_) / epsilon
    
    if np.isnan(ret):
      ret = -10
    
    return ret #Use the above code if you want to cap the dGLI
  
  def __evalWritheTwoSegments(self, p1, p2, p3, p4):
    n1 = np.cross(np.subtract(p3,p1), np.subtract(p4,p1)) / norm(np.cross(np.subtract(p3,p1), np.subtract(p4,p1)))
    n2 = np.cross(np.subtract(p4,p1), np.subtract(p4,p2)) / norm(np.cross(np.subtract(p4,p1), np.subtract(p4,p2)))
    n3 = np.cross(np.subtract(p4,p2), np.subtract(p3,p2)) / norm(np.cross(np.subtract(p4,p2), np.subtract(p3,p2)))
    n4 = np.cross(np.subtract(p3,p2), np.subtract(p3,p1)) / norm(np.cross(np.subtract(p3,p2), np.subtract(p3,p1)))
    sgn = np.sign(np.dot(np.cross(np.subtract(p4,p3),np.subtract(p2,p1)), np.subtract(p3,p1)))
    ret = np.arcsin(np.dot(n1,n2)) + np.arcsin(np.dot(n2,n3)) + np.arcsin(np.dot(n3,n4)) + np.arcsin(np.dot(n4,n1))
    return sgn*ret