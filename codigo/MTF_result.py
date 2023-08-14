import glob, os
import numpy as np
import h5py
#import vaex
import gc
import os
import pandas as pd
from matplotlib import pyplot as plt
import matplotlib.colors as colors
import skimage.color
import skimage.filters
import random
import math
gc.enable()
from scipy import ndimage
from scipy import stats
from tqdm import tqdm
from skimage.measure import find_contours

from matplotlib.animation import FuncAnimation
from IPython.display import HTML, display
from mpl_toolkits.mplot3d import Axes3D
import plotly.graph_objects as go
from matplotlib import pyplot as plt, animation
import tools_MTF as MTF
import chunk_readers as ch_read



path_file = glob.glob("*.h5")
a=0;
print(path_file)
h5_1st = h5py.File(path_file[0], 'r') # Load the file but not the data yet.

# list(h5_1st.keys()) # This helps you to know what index you want [15]
#e_field=h5_1st[list(h5_1st.keys())[16]] ;
h_field=h5_1st[list(h5_1st.keys())[17]] ;
j_field=h5_1st[list(h5_1st.keys())[18]] ;

hx=h_field['hx'] ; hy=h_field['hy'] ;  hz=h_field['hz'] ;
jx=j_field['jx'] ; jy=j_field['jy'] ;  jz=j_field['jz'] ;


# Define the outputs that will be stored
k=1
H_mag_mean=[];
n = 8 # 8 chuncks

jx_i, jy_i, jz_i = ch_read.chunk_reader_h(jx,jy,jz,n,2) # Solo leo el un chunk
hx_i, hy_i, hz_i = ch_read.chunk_reader_h(hx,hy,hz,n,2)

j_mag_i = np.sqrt(jx_i*jx_i + jy_i*jy_i + jz_i*jz_i)
h_mag_i = np.sqrt(hx_i*hx_i + hy_i*hy_i + hz_i*hz_i)

## Definir la zona

x_11 = 90; x_12 = 145; y_11 = 5; y_12 = 80; z_11 = 0; z_12 = 100
x_21 = 70; x_22 = 130; y_21 = 85; y_22 = 140; z_21 = 400; z_22 = 700
x_31 = 40; x_32 = 100; y_31 = 125; y_32 = 190; z_31 = 680; z_32 = 1000

zona = [slice(z_11, z_12, None), slice(y_11, y_12, None), slice(x_11, x_12, None)]
binary_zona1 = j_mag_i[tuple(zona)] > 0.2

#----------------------------------------------------------------

path_11,path_12,path_13,path_14 = MTF.four_path(zona)

B_11,E_11 = MTF.extract_data(hx_i, hy_i, hz_i,jx_i, jy_i, jz_i,path_11)
B_12,E_12 = MTF.extract_data(hx_i, hy_i, hz_i,jx_i, jy_i, jz_i,path_12)
B_13,E_13 = MTF.extract_data(hx_i, hy_i, hz_i,jx_i, jy_i, jz_i,path_13)
B_14,E_14 = MTF.extract_data(hx_i, hy_i, hz_i,jx_i, jy_i, jz_i,path_14)

M_11, val_11, vec_11 = MTF.Magnetic_variance(B_11)
M_12, val_12, vec_12 = MTF.Magnetic_variance(B_12)
M_13, val_13, vec_13 = MTF.Magnetic_variance(B_13)
M_14, val_14, vec_14 = MTF.Magnetic_variance(B_14)

U_psi_L, U_psi_N = MTF.U_psi(zona,vec_12,j_mag_i,jx_i,jy_i,jz_i,hx_i,hy_i,hz_i)
U_div1 = MTF.divergence(U_psi_L,U_psi_N,j_mag_i)

log_U1_psi_N = ((U_psi_N/np.abs(U_psi_N))*np.log(1 + (np.abs(U_psi_N)/np.max(np.sqrt(U_psi_L*U_psi_L+U_psi_N*U_psi_N)))))
log_U1_psi_L = ((U_psi_L/np.abs(U_psi_L))*np.log(1 + (np.abs(U_psi_L)/np.max(np.sqrt(U_psi_L*U_psi_L+U_psi_N*U_psi_N)))))

Hl, Hm, Hn, Hmag = MTF.change_of_basis_zone(zona,vec_12,h_mag_i,hx_i,hy_i,hz_i)

#--------------------------------------------------------------------

rows = 2
cols = 2
fig=plt.figure(figsize=(10,8))

ax1 = fig.add_subplot(rows, cols, 1)
ax1.set_title(r"$path 1$")
plt.plot(range(path_11.shape[0]),B_11[:,0],color="red",label=r"$H_{x}$")
plt.plot(range(path_11.shape[0]),B_11[:,1],color="blue",label=r"$H_{y}$")
plt.plot(range(path_11.shape[0]),B_11[:,2],color="k",label=r"$H_{z}$")
plt.legend()
plt.xlabel("Point")
plt.ylabel("H")

# Add the custom legend as text

txt_L = "L = ({x_1:.2f},{y_1:.2f},{z_1:.2f})"
plt.text(0, 0.10, txt_L.format(x_1 = vec_11[0][0],y_1 = vec_11[0][1],z_1 = vec_11[0][2]),
         fontsize=8, bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.5'))
txt_M = "M = ({x_2:.2f},{y_2:.2f},{z_2:.2f})"
plt.text(0, 0.085, txt_M.format(x_2 = vec_11[1][0],y_2= vec_11[1][1],z_2 = vec_11[1][2]),
         fontsize=8, bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.5'))
txt_N = "N = ({x_3:.2f},{y_3:.2f},{z_3:.2f})"
plt.text(0, 0.07, txt_N.format(x_3 = vec_11[2][0],y_3= vec_11[2][1],z_3 = vec_11[2][2]),
         fontsize=8, bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.5'))

fig.tight_layout()

ax2 = fig.add_subplot(rows, cols, 2)
ax2.set_title(r"$path 2$")
plt.plot(range(path_12.shape[0]),B_12[:,0],color="red",label=r"$H_{x}$")
plt.plot(range(path_12.shape[0]),B_12[:,1],color="blue",label=r"$H_{y}$")
plt.plot(range(path_12.shape[0]),B_12[:,2],color="k",label=r"$H_{z}$")
plt.legend()
plt.xlabel("Point")
plt.ylabel("H")

txt_L = "L = ({x_1:.2f},{y_1:.2f},{z_1:.2f})"
plt.text(0, 0.085, txt_L.format(x_1 = vec_12[0][0],y_1 = vec_12[0][1],z_1 = vec_12[0][2]),
         fontsize=8, bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.5'))
txt_M = "M = ({x_2:.2f},{y_2:.2f},{z_2:.2f})"
plt.text(0, 0.07, txt_M.format(x_2 = vec_12[1][0],y_2= vec_12[1][1],z_2 = vec_12[1][2]),
         fontsize=8, bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.5'))
txt_N = "N = ({x_3:.2f},{y_3:.2f},{z_3:.2f})"
plt.text(0, 0.055, txt_N.format(x_3 = vec_12[2][0],y_3= vec_12[2][1],z_3 = vec_12[2][2]),
         fontsize=8, bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.5'))

fig.tight_layout()

ax3 = fig.add_subplot(rows, cols, 3)
ax3.set_title(r"$path 3$")
plt.plot(range(path_13.shape[0]),B_13[:,0],color="red",label=r"$H_{x}$")
plt.plot(range(path_13.shape[0]),B_13[:,1],color="blue",label=r"$H_{y}$")
plt.plot(range(path_13.shape[0]),B_13[:,2],color="k",label=r"$H_{z}$")
plt.legend()
plt.xlabel("Point")
plt.ylabel("H")

txt_L = "L = ({x_1:.2f},{y_1:.2f},{z_1:.2f})"
plt.text(0, 0.085, txt_L.format(x_1 = vec_13[0][0],y_1 = vec_13[0][1],z_1 = vec_13[0][2]),
         fontsize=8, bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.5'))
txt_M = "M = ({x_2:.2f},{y_2:.2f},{z_2:.2f})"
plt.text(0, 0.07, txt_M.format(x_2 = vec_13[1][0],y_2= vec_13[1][1],z_2 = vec_13[1][2]),
         fontsize=8, bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.5'))
txt_N = "N = ({x_3:.2f},{y_3:.2f},{z_3:.2f})"
plt.text(0, 0.055, txt_N.format(x_3 = vec_13[2][0],y_3= vec_13[2][1],z_3 = vec_13[2][2]),
         fontsize=8, bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.5'))


fig.tight_layout()

ax4 = fig.add_subplot(rows, cols, 4)
ax4.set_title(r"$path 3$")
plt.plot(range(path_14.shape[0]),B_14[:,0],color="red",label=r"$H_{x}$")
plt.plot(range(path_14.shape[0]),B_14[:,1],color="blue",label=r"$H_{y}$")
plt.plot(range(path_14.shape[0]),B_14[:,2],color="k",label=r"$H_{z}$")
plt.legend()
plt.xlabel("Point")
plt.ylabel("H")

txt_L = "L = ({x_1:.2f},{y_1:.2f},{z_1:.2f})"
plt.text(15, 0.065, txt_L.format(x_1 = vec_14[0][0],y_1 = vec_14[0][1],z_1 = vec_14[0][2]),
         fontsize=8, bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.5'))
txt_M = "M = ({x_2:.2f},{y_2:.2f},{z_2:.2f})"
plt.text(15, 0.05, txt_M.format(x_2 = vec_14[1][0],y_2= vec_14[1][1],z_2 = vec_14[1][2]),
         fontsize=8, bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.5'))
txt_N = "N = ({x_3:.2f},{y_3:.2f},{z_3:.2f})"
plt.text(15, 0.035, txt_N.format(x_3 = vec_14[2][0],y_3= vec_14[2][1],z_3 = vec_14[2][2]),
         fontsize=8, bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.5'))

fig.tight_layout()
plt.savefig("paths_vs_H.png")

#------------------------------------------------------------------------

# Set up the figure with two subplots
fig, axs = plt.subplots(2, 2, figsize=(9, 9))

# Set number of slices to animate
num_slices = U_psi_N[tuple(zona)].shape[0]

vmin = np.min(U_div1[tuple(zona)])
vmax = np.max(U_div1[tuple(zona)])
norm = colors.TwoSlopeNorm( vcenter=0)

# Define update function for animation
def update(frame):
    hx = Hl[tuple(zona)][frame, :, :]
    hy = Hn[tuple(zona)][frame, :, :]

    x, y = np.meshgrid(np.arange(0, hx.shape[1]), np.arange(0, hy.shape[0]))

    contours = find_contours(binary_zona1[frame, :, :], 0.5)

    axs[0][0].clear()
    axs[0][1].clear()
    axs[1][0].clear()
    axs[1][1].clear()

    # Display the first image in the first subplot
    im1 = axs[0][0].imshow(jz_i[tuple(zona)][frame, :, :], cmap="jet",vmax=0.5,vmin=-0.5)
    axs[0][0].set_title(r"$J_{z}$")

    # Display the second image in the second subplot
    im2 = axs[0][1].imshow(U_div1[tuple(zona)][frame, :, :], cmap='seismic',vmax=50,vmin=-50 )  # Change the colormap if needed
    axs[0][1].set_title(r'$\nabla \cdot U_{\psi}$')

    # Display the second image in the second subplot
    im3 = axs[1][0].imshow(log_U1_psi_N[tuple(zona)][frame, :, :], cmap='seismic',vmax=0.005,vmin=-0.005)  # Change the colormap if needed
    axs[1][0].set_title(r'$U_{\psi, N}$')

    # Display the second image in the second subplot
    im4 = axs[1][1].imshow(log_U1_psi_L[tuple(zona)][frame, :, :], cmap='seismic',vmax=0.005,vmin=-0.005)  # Change the colormap if needed
    axs[1][1].set_title(r'$U_{\psi, L}$')

    # Plot streamlines
    for i in range(2):
      for j in range(2):
        stream = axs[i][j].streamplot(x, y, hx, hy, color="k",
                       density=0.4, linewidth=0.7, arrowstyle='->', arrowsize=1.5)

        for contour in contours:
          axs[i][j].plot(contour[:, 1], contour[:, 0], 'k',alpha=0.8)

    #axs[1][0].quiver(U1_psi_L[tuple(zona_1)][frame, :, :],U1_psi_N[zona_1][frame, :, :])
    #axs[1][1].quiver(U1_psi_L[tuple(zona_1)][frame, :, :],U1_psi_N[zona_1][frame, :, :])
    axs[0][0].set_ylabel(r"y")
    axs[1][0].set_ylabel(r"y")
    axs[1][0].set_xlabel(r"x")
    axs[1][1].set_xlabel(r"x")

    for i in range(2):
      for j in range(2):
        axs[i][j].set_xlim((0,zona[2].stop-zona[2].start))
        axs[i][j].set_ylim((0,zona[1].stop-zona[1].start))

# Create animation
U_psi_animation = FuncAnimation(fig, update, frames=num_slices, interval=50)

# Display animation
HTML(U_psi_animation.to_jshtml())

writervideo = animation.FFMpegWriter(fps=5)

U_psi_animation.save('result_path2.mp4', writer=writervideo)
plt.close()
