import argparse
import ast
from dgli import planning
from dgli.planning import get_linear_interpolation
import copy
import os
import dgli.loadData as loadData
from dgli.dGLI_class import dGLI_class
from dgli.dGLI_class import includeEdges
from dgli.plot import plot, FigTypes
import matplotlib.pyplot as plt
import numpy as np

###################################### Visualize Continuity of the CloSE representation ############################################
# Define the function to handle parsing the list input
def parse_array(string):
    try:
        # Convert the string [a,b] into an actual Python list
        return np.array(ast.literal_eval(string))
    except Exception as e:
        raise argparse.ArgumentTypeError("Invalid input format. Use --goal=[a,b]")

# Define argument with default value as described
parser = argparse.ArgumentParser()
parser.add_argument("--filePath", type=str, default="Dataset/dataset_MATLAB/half_tshirt_01.mat", help="Path to the file")
parser.add_argument("--includeEdges", type=lambda v: includeEdges[v.upper()], default=includeEdges.ALL, help="whether to use 'alternate', 'all' or 'corners'")
parser.add_argument("--start", type=parse_array, default=np.array([5.7, 1.6]), help="Input start fold location array in the format --start=[x, y]")
parser.add_argument("--end", type=parse_array, default=np.array([0.22, 3.45]), help="Input end fold location array in the format --end=[x, y]")

# Parse arguments
input_args = parser.parse_args()
filePath = input_args.filePath
includeEdges_ = input_args.includeEdges

start = 0
borders_,_ = loadData.loadAllFrames(filePath)
borders = []
borders.append(borders_[start])
dGLI_ = dGLI_class(includeEdges_, borders)

CloSE_rep_1 = [dGLI_.get_corners(), input_args.start]		# Define Start Edge Location
CloSE_rep_2 = [dGLI_.get_corners(), input_args.end]			# Define End Edge Location

# Get Interpolated CloSE representations
interpolated_CloSE_reps = get_linear_interpolation(CloSE_rep_1, CloSE_rep_2, 90)

def get_filename():
  return f"{os.path.splitext(os.path.basename(filePath))[0]}"

path=f"Results/Continuity/{get_filename()}"
os.makedirs(path, exist_ok=True)

for i in range(len(interpolated_CloSE_reps)):
	CloSE_rep = interpolated_CloSE_reps[i]
	plot_ =  plot()
	fig = plt.figure(figsize=(12, 6))

	ax = fig.add_subplot(1, 2, 1, projection='polar')
	plot_.polar_plot(ax, CloSE_rep[0], color = 'orange')
	plot_.polar_plot(ax, CloSE_rep[1], color = 'red', with_orientation=True)
	
	border = copy.deepcopy(dGLI_.get_border(0))
	foldLine = CloSE_rep[1]
	fold_points = planning.fold_(border, foldLine)

	# 3D plot on the left
	ax1 = fig.add_subplot(1, 2, 2, projection='3d')
	ax1.scatter(fold_points[:, 0], fold_points[:, 1], fold_points[:, 2], color='red', s=50, label="New Points")
	ax1.plot(fold_points[:, 0], fold_points[:, 1], fold_points[:, 2], linestyle='dotted', color='black')
	plot_.static_plot(ax1, border, dGLI_.get_edges(), FigTypes.BORDER, highlightEdges=True, edgeType=includeEdges.ALL, title="", clearAxis=False)

	fig.savefig(f'{path}/{i}.png', format='png', dpi=100)
	plt.close()

