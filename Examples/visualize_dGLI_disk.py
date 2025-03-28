############################# Generate movie of evolution of dGLI with folds ###########################################

import argparse
import os
from matplotlib import pyplot as plt
from dgli import loadData
from dgli.dGLI_class import dGLI_class, includeEdges
from dgli.plot import FigTypes, plot

############################# Priliminaries (Edit this code to change thing like save location) ########################
# Define argument with default value as described
parser = argparse.ArgumentParser()
parser.add_argument("--filePath", type=str, default="Dataset/dataset_MATLAB/50diagonalHalf_napkin_01.mat", help="Path to the file")
parser.add_argument("--includeEdges", type=lambda v: includeEdges[v.upper()], default=includeEdges.ALL, help="whether to use 'alternate', 'all' or 'corners'")

# Parse arguments
input_args = parser.parse_args()
filePath = input_args.filePath
includeEdges_ = input_args.includeEdges

start = 1
path="Results/Animations" 								# The animation will be saved here
os.makedirs(path, exist_ok=True)
offset_ = 0												# Update this if you want to rotate the generated dGLI dish image (only for visualization purposes)

def get_filename():
  return f"{os.path.splitext(os.path.basename(filePath))[0]}"

path_images = f"{path}/Images/{get_filename()}"			# The images for the animation will be saved here
os.makedirs(path_images, exist_ok=True)

output_file = f"{path}/{get_filename()}.mp4"			# Output location of the video

def save_frame(timestep):								# Save timestep as png
	fig = plt.figure(figsize=(19.2, 10.8), dpi=100)		# 1920x1080 resolution

	# Define 3D and 2D subplots
	ax0 = fig.add_subplot(1, 3, 1, projection='3d')  	# 3D plot
	ax1 = fig.add_subplot(1, 3, 2)  					# 2D plot
	ax2 = fig.add_subplot(1, 3, 3)  					# 2D plot

	plot_.static_plot(ax0, dGLI_.get_border(timestep), dGLI_.get_edges(), FigTypes.BORDER, 
										highlightEdges=True, edgeType=includeEdges.ALL, title="")
	plot_.static_plot(ax1, dGLI_.get_dGLICoords(timestep), dGLI_.get_edges(), FigTypes.DGLI_DISK, title="", offset=offset_)
	plot_.static_plot(ax2, dGLI_.get_dGLI_diff(timestep, start), dGLI_.get_edges(), FigTypes.DGLI_DISK, title="", offset=offset_)

	frame_path = f"{path_images}/frame_{timestep:04d}.png"
	plt.savefig(frame_path, dpi=100)  					# High-quality images. Change dpi as required
	plt.close(fig)


############################# The visualization code to generate the dGLI sequence video ###################################
# Extract the borders and the mesh from the data
borders,meshes = loadData.loadAllFrames(filePath)

# Create the dGLI class
dGLI_ = dGLI_class(includeEdges_, borders)
dGLI_.set_start_frame(start)
end = dGLI_.get_frames() - 1

plot_ =  plot(updateNorm=False, showLines=True)

# Run the function save_frame in parallel
import multiprocessing as mp
import subprocess
with mp.Pool(int(mp.cpu_count()/2)) as pool:
	pool.map(save_frame, range(start, end))

	# Convert images to 1080p MP4 using ffmpeg
	fps = 30  # Smooth animation
	ffmpeg_cmd = [
			"ffmpeg", "-framerate", str(fps), "-i", f"{path}/Images/{get_filename()}/frame_%04d.png",
			"-c:v", "libx264", "-preset", "slow", "-crf", "18", "-b:v", "8M",
			"-pix_fmt", "yuv420p", "-vf", "scale=1920:1080", output_file
	]
	
	subprocess.run(ffmpeg_cmd, check=True)

	print(f"Video saved as {output_file} in 1080p")