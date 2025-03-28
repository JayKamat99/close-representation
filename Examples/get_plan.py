'''
We are given the start state and the end state / end CloSE representation.
We are tasked to plan macro states from the start to the goal
'''

import argparse
import ast
import copy
from dgli import planning
import dgli.loadData as loadData
from dgli.dGLI_class import dGLI_class
from dgli.dGLI_class import includeEdges
from dgli.plot import plot, FigTypes
import matplotlib.pyplot as plt
import numpy as np

###################################### Inputs ############################################
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
parser.add_argument("--goal", type=parse_array, default=np.array([5.70, 1.70]), help="Input goal fold array in the format --goal=[x, y]")

# Parse arguments
input_args = parser.parse_args()
includeEdges_ = input_args.includeEdges

borders_,_ = loadData.loadAllFrames(input_args.filePath)  
borders = []
borders.append(borders_[0])
dGLI_ = dGLI_class(includeEdges_, borders)
Eb = loadData.get_border_mesh_mapping(input_args.filePath)		# Mapping of border to mesh
  
dGLI_.add_border(borders_[len(borders_)-1])
folded_border = copy.deepcopy(borders[-1])                      # This is the copy of the stage we are starting from
 
base = 0
dGLI_.set_start_frame(base)
stage_1 = 1

# Get feature vector of the start
CloSE_rep = []
CloSE_rep.append(dGLI_.get_feature_vector(stage_1, base)) # init 
CloSE_rep.append([dGLI_.get_corners(), input_args.goal]) # goal

# Check if it is in the same semantic class. If not, move to base and then the goal
# To check this. check if there is any point when you move from one to another in either direction
# In moving from point 0 to point 1 on the edge, what indices of corners do you capture?
# if number is same ands the indices are the same, it is in the same category.

# Get corners to be affected by the fold
corners = dGLI_.get_corners()

# Get corners to be affected by the fold
corners = dGLI_.get_corners()
cornersFolded = []
cornersFolded.append(planning.get_affected_corners(corners, CloSE_rep[0][1]))
cornersFolded.append(planning.get_affected_corners(corners, CloSE_rep[1][1]))

same_category=False
if (cornersFolded[0].tolist() == cornersFolded[1].tolist()):
    same_category = True
    
def get_plan(same_category):
    if same_category:
        return f'''
        Go Directly
        move corner(s): {cornersFolded[0]} directly to it's final position
        '''
    else:
        return f'''
        unfold corner(s): {cornersFolded[0]} across line: {CloSE_rep[0][1]%(2*np.pi)}
        fold corner(s): {cornersFolded[1]} across line: {CloSE_rep[1][1]%(2*np.pi)}
        '''
        
print(get_plan(same_category))

# Low-level planning
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
                a = dGLI_.get_border(base)[dGLI_.get_vertex_index(toFoldCorners[i])]
                a = a[:-1].tolist()
                b = dGLI_.get_border(base)[dGLI_.get_vertex_index(toFoldCorners[j])]
                b = b[:-1].tolist()
                area = planning.trapezoid_area([a, b, p1, p2])
                if max_area < area:
                    max_area = area
                    cornersIdx = [i,j]
        return [toFoldCorners[cornersIdx[0]], toFoldCorners[cornersIdx[1]]]
    
# This depends on how many folds we have (for now it's just 1 or 2)
# This same category is incomplete and I will come back to this later
if same_category:
    p1, p2, _, _ = planning.get_foldPoints(dGLI_.get_border(base), CloSE_rep[0][1])
    man_corners = get_corners_to_manipulate(cornersFolded[0], np.array([p1, p2]))
    highlightIndices = []
    for i in range(len(man_corners)):
        highlightIndices.append((dGLI_.get_vertex_index(man_corners[i])+1)%dGLI_.get_edges())
    # Plotting all graphs side by side
    plot_ =  plot()
    fig = plt.figure(figsize=(12, 6))
    ax0 = fig.add_subplot(1, 2, 1, projection='3d')
    plot_.static_plot(ax0, folded_border, len(folded_border), FigTypes.BORDER, highlightEdges=True, edgeType=includeEdges.ALL, title="Before", clearAxis=False, highlightIndices=highlightIndices)
    
    for step in range(2):
        planning.fold(folded_border, CloSE_rep[step][1])
    
    goals = []
    controlPoints = []
    for i in range(len(man_corners)):
        controlPoints.append(Eb[(dGLI_.get_vertex_index(man_corners[i])+1)%dGLI_.get_edges()])
        goals.append(folded_border[(dGLI_.get_vertex_index(man_corners[i])+1)%dGLI_.get_edges()])
    for point in range(len(goals)):
        print(f"    move mesh vertex {controlPoints[point]} to goal location {goals[point]}")
        

    ax1 = fig.add_subplot(1, 2, 2, projection='3d')
    plot_.static_plot(ax1, folded_border, len(folded_border), FigTypes.BORDER, highlightEdges=True, edgeType=includeEdges.ALL, title="After", clearAxis=False, highlightIndices=highlightIndices)
    plt.show()
else:
    # When we need to carry out 2 steps
    steps=2
    for step in range(steps):
        p1, p2, _, _ = planning.get_foldPoints(dGLI_.get_border(base), CloSE_rep[step][1])
        man_corners = get_corners_to_manipulate(cornersFolded[step], np.array([p1, p2]))
        highlightIndices = []
        for i in range(len(man_corners)):
            highlightIndices.append((dGLI_.get_vertex_index(man_corners[i])+1)%dGLI_.get_edges())

        # Plotting all graphs side by side
        plot_ =  plot()
        fig = plt.figure(figsize=(12, 6))
        ax0 = fig.add_subplot(1, 2, 1, projection='3d')
        plot_.static_plot(ax0, folded_border, len(folded_border), FigTypes.BORDER, highlightEdges=True, edgeType=includeEdges.ALL, title="State Before fold", clearAxis=False, highlightIndices=highlightIndices)

        planning.fold(folded_border, CloSE_rep[step][1])

        goals = []
        controlPoints = []
        for i in range(len(man_corners)):
            controlPoints.append(Eb[(dGLI_.get_vertex_index(man_corners[i])+1)%dGLI_.get_edges()])
            goals.append(folded_border[(dGLI_.get_vertex_index(man_corners[i])+1)%dGLI_.get_edges()])

        print(f"step {step}:")
        for point in range(len(goals)):
            print(f"    move mesh vertex {controlPoints[point]} to goal location {goals[point]}")
            

        ax1 = fig.add_subplot(1, 2, 2, projection='3d')
        plot_.static_plot(ax1, folded_border, len(folded_border), FigTypes.BORDER, highlightEdges=True, edgeType=includeEdges.ALL, title="Predicted State after fold", clearAxis=False, highlightIndices=highlightIndices)
        plt.show()