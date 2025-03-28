import argparse
import numpy as np
from dgli import loadData
from dgli.dGLI_class import dGLI_class, includeEdges
from dgli.semanticLabeling import get_semantic_label

############################# Priliminaries ########################
parser = argparse.ArgumentParser()
parser.add_argument("--filePath", type=str, default="Dataset/dataset_MATLAB/bottomTopHalf_Tshirt_03.mat", help="Path to the file")
parser.add_argument("--includeEdges", type=lambda v: includeEdges[v.upper()], default=includeEdges.ALL, help="whether to use 'alternate', 'all' or 'corners'")
parser.add_argument("--addNoise", action="store_true", help="Enable noise addition")

# Parse arguments
input_args = parser.parse_args()
includeEdges_ = input_args.includeEdges

borders,_ = loadData.loadAllFrames(input_args.filePath)  
dGLI_ = dGLI_class(includeEdges_, borders)
if input_args.addNoise:
  dGLI_.add_noise(0, 0.005)

# Update the start and the goal as per your requirements. However, this must work in most cases
start = 0
end = -1 

################################### Get the feature vector for labeling the cloth #########################################

featureVector = dGLI_.get_feature_vector(end, start)
print(f"{get_semantic_label(featureVector)}")