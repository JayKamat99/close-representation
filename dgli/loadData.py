import xml.etree.ElementTree as ET
import scipy.io
from scipy.sparse import csc_matrix
import os

# Loading Mesh
def loadAllFrames(filePath):
    extension = os.path.splitext(filePath)[1]
    if extension == ".xml":
        return loadAllFrames_xml(filePath)
    elif extension == ".mat":
        return loadAllFrames_mat(filePath)
    else:
        # Ideally should shoot an error but you just run the xml file with warning
        Warning("No filePath argument passed, using the defalut xml example")
        return loadAllFrames_xml(filePath)
    
def loadAllFrames_xml(filePath = "Dataset/VR_framework_dataset/Dataset/Dataset_02/01_2PM_2PCM_09.xml"):
  tree = ET.parse(filePath)
  root = tree.getroot()
  # writting the 400 vertexs when initializes the Manipulation
  numFrames = len(root[2])
  meshes = []
  borders = []
  for j in range(numFrames):
    mesh_started_frame_coordinates = root[2][j][11][0][0]
    mesh = []
    numVertices=len(mesh_started_frame_coordinates)
    for i in range(numVertices):
      x=float(mesh_started_frame_coordinates[i][0].attrib['x'])
      y=float(mesh_started_frame_coordinates[i][0].attrib['y'])
      z=float(mesh_started_frame_coordinates[i][0].attrib['z'])
      mesh.append([x,y,z])
    meshes.append(mesh)
    
    sideSize = 20
    s1 = list(range(0, sideSize-1))
    s2 = [(i+1)*sideSize-1 for i in range(0, sideSize-1)]
    s3 = [sideSize*sideSize-i-1 for i in range(0, sideSize-1)]
    s4 = list(reversed([(i+1)*sideSize for i in range(0, sideSize-1)]))
    borderInd = s1 + s2 + s3 + s4
    borders.append([mesh[i] for i in borderInd])

  return borders, meshes

def loadAllFrames_mat(filePath = "Dataset/MATLAB_dataset/DiskSimulation.mat"):
  frames = []
  # Load the .mat file
  data = scipy.io.loadmat(filePath)

  # Extract variables
  phiPositions = data['phiPositions']
  Eb = data['Eb']
  # Iterate through all timesteps:
  for timestep in range(phiPositions.size):
    sparse_matrix = csc_matrix(phiPositions[0, timestep])
    X_timestep = sparse_matrix.toarray()
    frames.append(X_timestep)
  
  # Now get the boundary
  # Coordinates of the first edge of the boundary
  border = []
  for timestep in range(phiPositions.size):
    border_points = []
    for i in range(Eb.shape[0]):
      border_points.append(frames[timestep][Eb[i,0]-1])
    border.append(border_points)
  
  return border, frames

def get_border_mesh_mapping(filePath = "Dataset/MATLAB_dataset/DiskSimulation.mat"):
  # Load the .mat file
  data = scipy.io.loadmat(filePath)
  Eb = data['Eb']
  return Eb[:,0]