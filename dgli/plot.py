from matplotlib import patches
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Wedge
from enum import Enum
from dgli.dGLI_class import includeEdges

# Custom normalization class for a sigmoid-like scale
class SigmoidNorm(mpl.colors.Normalize):
  def __init__(self, vmin=None, vmax=None, midpoint=0, scale=10, clip=False):
    super().__init__(vmin, vmax, clip)
    self.midpoint = midpoint
    self.scale = scale  # Control the steepness of the non-linear transition

  def __call__(self, value, clip=None):
    # Normalize value to be between 0 and 1 using a sigmoid-like function
    norm_value = (value - self.midpoint) / (self.vmax - self.vmin)
    scaled_value = 1 / (1 + np.exp(-self.scale * norm_value))  # Sigmoid-like transformation
    return scaled_value

class FigTypes(Enum):
  BORDER = 'border'
  DGLI_DISK = 'dgli_disk'
  DGLI_MATRIX = 'dgli_matrix'

class plot:  
  def __init__(self, showLines=False, updateNorm=True):
    colors = [(0, 0, 1), (1, 1, 1), (1, 0.5, 0)]  # Blue -> White -> Orange
    self.__cmap = LinearSegmentedColormap.from_list('blue_white_orange', colors, N=100)
    self.__norm = SigmoidNorm(vmin=-4, vmax=4, midpoint=0, scale=10)  # Adjust scale for steepness, also need to adjust max, min
    self.__showLines = showLines
    self.__updateNorm = updateNorm
    
  def get_cmap(self):
    return self.__cmap
  
  def get_norm(self):
    return self.__norm
  
  def static_plot(self, ax, array, edges, type=FigTypes.BORDER, title="", highlightEdges=False, edgeType=includeEdges.ALTERNATE, offset=0.0, clearAxis=True, highlightIndices=[]):
    if (type == FigTypes.BORDER):
      self.__update_border(ax, array, title, highlightEdges, edgeType, clearAxis, highlightIndices)
    elif (type == FigTypes.DGLI_DISK):
      self.__update_circular_dGLI(ax, array, edges, title, offset, clearAxis)
    else:
      self.__update_dgli_semimatrix(ax, array, edges, title, clearAxis)
      
  def scatter_plot(self, ax, array, title="", offset=0.0):
    # Unpacking radius and angle
    r, theta = zip(*array)
    offset_tuple = tuple([offset] * len(theta))
    theta = tuple(a + b for a, b in zip(theta, offset_tuple))
    ax.scatter(theta, r, c='b')
    ax.set_title(title)
    ax.set_ylim(0,1)
    # Hide grid and labels
    ax.set_xticks([])
    ax.set_yticks([])
    
  def polar_plot(self, ax, array, title="", offset=0.0, color = "blue", with_orientation=False, ylim=1):
    # Unpacking radius and angle
    r = np.ones(array.size)
    theta = array
    # offset_tuple = tuple([offset] * len(theta))
    # print(theta)
    # theta = tuple(a + b for a, b in zip(theta, offset_tuple))
    ax.scatter(theta, r, c=color, s=500)
    # Annotate each point with its (theta, r) value
    for t, r_val in zip(theta, r):
      ax.annotate(f"{t%(2*np.pi):.2f}", (t, r_val),
                  textcoords="offset points", xytext=(5, 5), ha='center', fontsize=20)
    if with_orientation:
      if (theta[0]>theta[1]):
        theta[0] -= 2*np.pi
      # Generate points along the circular arc
      theta_arc = np.linspace(theta[0], theta[1], 100)  # 100 points from theta[0] to theta[1]
      r_arc = np.ones_like(theta_arc)  # Constant radius

      # Plot the arc
      ax.plot(theta_arc, r_arc, color=color, linewidth=5)
    ax.set_title(title)
    ax.set_ylim(0,ylim)
    # Hide grid and labels
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines['polar'].set_visible(True)  # Show the outer boundary
    
  ## Private Methods ##
  
  def __update_norm(self, array):
    mean = np.mean(np.abs(array))
    std_dev = np.std(np.abs(array))
    vmax = np.minimum(mean + 2*std_dev, np.max(np.abs(array)))
    vmin = -vmax
    self.__norm = SigmoidNorm(vmin, vmax, midpoint=0, scale=10)
  
  def __offset(self, offset, theta):
    for idx in range(len(theta)):
      theta[idx] += offset
    
  def __getAllEdges(self, curve):
    outCurve = []
    # close curve
    # if (curve[0] != curve[-1]).all():
    #   curve.append(curve[0])
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
  
  def __update_border(self, ax, border, title, highlightEdges, edgeType, clearAxis=True, highlightIndices=[]):
    if clearAxis:
      ax.cla()  # Clear the current axes
    ## Highlight indices
    for i in highlightIndices:
      highlight_index = i  # Replace i with the desired index or parameter
      highlight_color = 'green'  # Distinct color for the highlight
      highlight_size = 100  # Marker size for highlighting

      # Ensure the index is within bounds
      if 0 <= highlight_index < len(border):
        vertex_to_highlight = border[highlight_index]
        ax.scatter3D(
          vertex_to_highlight[0], vertex_to_highlight[1], vertex_to_highlight[2], 
          color=highlight_color, s=highlight_size, label='Highlighted Vertex'
        )

    ## Visualizing Border
    ax.scatter3D(*zip(*border), color='blue', label='Border Points')
    if highlightEdges:
      if edgeType is includeEdges.ALTERNATE:
        edges = self.__getAlternateEdges(border)
      elif edgeType is includeEdges.ALL:
        edges = self.__getAllEdges(border)
      else:
        raise TypeError("Depriciated includeEdges type")
      # Scatter plot of all border points
      
      # Highlight alternate edges with a distinct color and annotate them
      for i, edge in enumerate(edges):
        # Unpack start and end points of the edge
        start, end = edge
        
        # Draw the edge in red
        ax.plot3D(
          [start[0], end[0]], 
          [start[1], end[1]], 
          [start[2], end[2]], 
          color='red', linewidth=2, label='Alternate Edge' if i == 0 else ""
        )
        
      # For the last edge
      start, _ = edges[0]
      _, end = edges[-1]
        
      # Draw the edge in red
      ax.plot3D(
        [start[0], end[0]], 
        [start[1], end[1]], 
        [start[2], end[2]], 
        color='red', linewidth=2, label='Alternate Edge' if i == 0 else ""
      ) 
      
        # Label the edge with its index
        # mid_point = [(start[0] + end[0]) / 2, (start[1] + end[1]) / 2, (start[2] + end[2]) / 2]
        # ax.text(mid_point[0], mid_point[1], mid_point[2], f'{i}', color='black', fontsize=5)
    ax.axis('equal')
    ax.set_title(title)
    ax.set_axis_off() # Remove axes, labels, and ticks
    ax.view_init(elev=90, azim=-90) 
    
  def __update_dgli_semimatrix(self, ax, dgli_vals, edges, title, clearAxis=True):
    if clearAxis:
      ax.cla()  # Clear the current axes
    if self.__updateNorm:
      self.__update_norm(dgli_vals)
    data = np.zeros([edges, edges])
    # Arrange data
    count = 0
    # reassign for the new timestep
    for i in range(edges-1):
      for j in range(i+1,edges):
        # print(i,j,count,dgli_vals[count])
        data[i, j] = dgli_vals[count]
        count += 1
    ax.imshow(data, cmap=self.__cmap, norm=self.__norm)

    # Add gridlines
    ax.grid(which='major', axis='both', linestyle='-', color='k', linewidth=0.1)
    ax.set_xticks(np.arange(-.5, edges, 1))
    ax.set_yticks(np.arange(-.5, edges, 1))
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_title(title)
  
  def __update_circular_dGLI(self, ax, dgli_vals, edges, title, offset, clearAxis=True):
    if clearAxis:
      ax.cla()  # Clear the current axes
    if self.__updateNorm:
      self.__update_norm(dgli_vals)
    num_levels = int(edges/2)  # The floor of edges / 2
    
    # Make regions
    radius = 1
    radii = [radius]
    for i in range(num_levels):
      radius -= 1 / num_levels
      radii.append(radius)

    # Function to draw radial lines for each region
    def draw_radial_lines(level_radius_start, level_radius_end, edges, offset=0):
      angles = np.linspace(0, 2 * np.pi, edges, endpoint=False) + offset
      for angle in angles:
        ax.plot([level_radius_start * np.cos(angle), level_radius_end * np.cos(angle)],
                [level_radius_start * np.sin(angle), level_radius_end * np.sin(angle)],
                color='black', lw=0.1)

    # Function to color each sector
    def color_sectors(offset):
      for level in range(num_levels):
        for cell in range(edges):
          # Random color for each sector
          color = get_color(cell,cell+level+1)
          # color = [random.random(),random.random(),random.random()]
          # Define the start and end angle of the sector
          # offset = 0 #This is done to match the image
          theta_start = cell * 2 * np.pi / edges + level * np.pi / edges + offset
          theta_end = 2 * np.pi / edges + theta_start
          # Create filled wedges (annular sectors) between the radii
          ax.add_patch(Wedge((0, 0), radii[level], theta_start * 180 / np.pi, theta_end * 180 / np.pi,
                            width=radii[level] - radii[level+1], color=color, edgecolor='black'))

    def get_color(i,j):
      value = dgli_vals[get_location(i, j, edges)]
      if value == 10:
        return 'red'
      elif value == -10:
        return 'black'
      else:
        return self.__cmap(self.__norm(value))
    
    def get_location(i,j,edges):
      # Ensure i and j are in the required range
      i = i%edges
      j = j%edges
      if i==j:
        raise ValueError("Invalid Query, values of i and j are the same")  
      # Ensure i<j
      if j<i:
        i,j = j,i 
      return (i*edges - i*(i+1)//2 + j -i - 1)

    # Draw the concentric circles
    for radius in radii:
      circle = plt.Circle((0, 0), radius, color='black', fill=False, linewidth=1)
      ax.add_artist(circle)
    
    if self.__showLines:    
      # Draw radial lines
      for i in range(num_levels):
        draw_radial_lines(radii[i+1], radii[i], edges, i*np.pi/edges)
        
    # Draw the partitions and color them
    color_sectors(offset)

    # Set axis limits
    ax.set_xlim(-radii[0] , radii[0] )
    ax.set_ylim(-radii[0] , radii[0] )

    # Maintain aspect ratio
    ax.set_aspect('equal')

    # Hide axes
    ax.axis('off')
    ax.set_title(title)