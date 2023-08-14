#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  5 12:47:18 2023

@author: afguerrerogu
"""
import numpy as np
from scipy import ndimage
from scipy import stats
from tqdm import tqdm
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import matplotlib.colors as colors
from matplotlib.animation import FuncAnimation
from IPython.display import HTML, display


# basis X, Y, Z
e1_x = np.array([1, 0, 0])
e2_y = np.array([0, 1, 0])
e3_z = np.array([0, 0, 1])
basis_old = np.array([e1_x, e2_y, e3_z])

def extract_data(Hx, Hy, Hz, Jx, Jy, Jz, path):
    """
    Extrae datos de la simulacion que se encuentran en los arreglos
    Hx, Hy, Hz, Jx, Jy, Jz basado en una ruta dada.
    
    Input:
        - Hx (array): Arreglo de valores para la coordenada x del campo magnético.
        - Hy (array): Arreglo de valores para la coordenada y del campo magnético.
        - Hz (array): Arreglo de valores para la coordenada z del campo magnético.
        - Jx (array): Arreglo de valores para la coordenada x de la densidad de corriente.
        - Jy (array): Arreglo de valores para la coordenada y de la densidad de corriente.
        - Jz (array): Arreglo de valores para la coordenada z de la densidad de corriente.
        - path (array): Ruta que especifica los índices de los puntos de interés en los arreglos.
        
    output:
        - E (array): Matriz NumPy que contiene los valores de la densidad de corriente
          (Jx, Jy, Jz) para los puntos de la ruta especificada.
        - B (array): Matriz NumPy que contiene los valores del campo magnético 
          (Hx, Hy, Hz) para los puntos de la ruta especificada.
    """
    
    B = []
    E = []

    for point in path:
        point_z = point[0]
        point_y = point[1]
        point_x = point[2]
        B.append([Hx[point_z,point_y,point_x], Hy[point_z,point_y,point_x], Hz[point_z,point_y,point_x]])
        E.append([Jx[point_z,point_y,point_x], Jy[point_z,point_y,point_x], Jz[point_z,point_y,point_x]])
        
    B = np.array(B)
    E = np.array(E)

    return B, E


def Magnetic_variance(B):
    """
    Calcula la matriz de varianza magnética, los valores y vectores propios correspondientes.

    Descripción:
        Esta función calcula la matriz de varianza magnética a partir del arreglo B de valores magnéticos.
        Luego, calcula los valores propios y los vectores propios asociados a partir de la matriz de varianza magnética.

    Entrada:
        - B (array): Matriz NumPy que contiene los valores del campo magnetico.

    Salida:
        - M (array): Matriz de varianza magnética de tamaño 3x3.
        - eig_val_sorted (list): Lista de los valores propios de la matriz M, ordenados de mayor a menor.
        - vec (array): Array de los vectores propios normalizados correspondientes a los valores propios de la matriz M.
          Estos vectores corresponden las bases de las nuevas coordenadas LMN 
    """

    M = np.zeros((3, 3))

    for i in range(3):
        for j in range(3):
            M[i, j] = np.mean(B[:, i] * B[:, j]) - np.mean(B[:, i]) * np.mean(B[:, j])

    eig_val, eig_vec = np.linalg.eig(M)

    combined = list(zip(eig_val, eig_vec))
    combined.sort(key=lambda x: x[0], reverse=True)

    eig_val_sorted = [item[0] for item in combined]
    other_list_sorted = [item[1] for item in combined]

    vec = np.array([other_list_sorted[0] / np.linalg.norm(other_list_sorted[0]),
                    other_list_sorted[1] / np.linalg.norm(other_list_sorted[1]),
                    other_list_sorted[2] / np.linalg.norm(other_list_sorted[2])])

    return M, eig_val_sorted, vec


def change_of_basis(V, basis_old, eng_vec):
    """
    Realiza un cambio de base para un vector dado.

    Descripción:
        Esta función realiza un cambio de base para un vector dado utilizando una matriz de transformación.
        La matriz de transformación se construye a partir de la base canonica y los vectores de la nueva base.
        Luego, se aplica la transformación al vector V para obtener el vector resultante en la nueva base.

    Entrada:
        - V (array): Vector de entrada que se desea transformar.
        - basis_old (array): Matriz NumPy que representa la base original.
        - eng_vec (array): Matriz NumPy que contiene los vectores propios asociados a la nueva base.

    Salida:
        - vec_new (array): Vector resultante después de aplicar el cambio de base.
    """

    # Matriz de transformación
    transformation_matrix = np.linalg.inv(basis_old).dot(eng_vec)

    # Transformar el vector a la nueva base
    vec_new = transformation_matrix.dot(V)

    return vec_new

# basis X, Y, Z
e1_x = np.array([1, 0, 0])
e2_y = np.array([0, 1, 0])
e3_z = np.array([0, 0, 1])
basis_old = np.array([e1_x, e2_y, e3_z])

def velocity(E, B, eng_vec):
    """
    Calcula la velocidad del flujo magnetico en un sistema de coordenadas dado.
    
    U_{psi} = cE_z/B_p (zxb_p) 

    Descripción:
        Esta función calcula la velocidad en un sistema de coordenadas basado en los campos eléctrico y magnético,
        así como en los vectores propios asociados a una nueva base.

    Entrada:
        - E (array): Vector que representa el campo eléctrico en el sistema de coordenadas original.
        - B (array): Vector que representa el campo magnético en el sistema de coordenadas original.
        - eng_vec (array): Matriz NumPy que contiene los vectores propios asociados a la nueva base.

    Salida:
        - velocity (array): Vector que representa la velocidad  del flujo calculada
        en el sistema de coordenadas LMN.
    """

    E_LMN = change_of_basis(E, basis_old, eng_vec)
    B_LMN = change_of_basis(B, basis_old, eng_vec)

    E_z = E_LMN[1]  # Componente M

    B_p_vec = [B_LMN[0], 0, B_LMN[2]]

    B_p = np.linalg.norm(B_p_vec)

    b_p = B_p_vec / B_p

    velocity = (E_z / B_p) * np.cross([0, 1, 0], b_p)

    return velocity


# Define a 3D structure using a numpy array called str_3D, which represents a structure
# with a 3x3x3 connectivity, meaning that points can be connected either by the sides or the corners of the pixels.
str_3D = np.ones((3, 3, 3), dtype=int)


str_3D = np.ones((3, 3, 3), dtype=int)

def seleccion_region(arr_j, arr_b, j, sigma):
    """
    Realiza la selección de regiones en una simulación basada en criterios de valor y sigma.

    Descripción:
        Esta función selecciona regiones en una simulación binaria basándose en criterios de valor y sigma.
        Se eliminan las regiones que no cumplen con los criterios y se devuelve información sobre las regiones seleccionadas.

    Parámetros de entrada:
        - arr_j (array): Simulación de referencia para la selección de regiones.
        - arr_b (array): Simulación binaria utilizada como máscara para la selección de regiones.
        - j (float): Valor central para el criterio de selección.
        - sigma (float): Ancho de la ventana de selección.

    Resultado:
        - bounding_boxes (list): Lista de las cajas delimitadoras de las regiones seleccionadas.
        - binary_simulacion (array): Simulación binaria modificada, con las regiones no seleccionadas eliminadas.
    """

    # Crear simulación binaria
    binary_simulacion = np.logical_and(arr_j > j - sigma, arr_j < j + sigma)

    labeled_structures, num_structures = ndimage.label(binary_simulacion)
    # Calcular volúmenes de las regiones conectadas
    structure_volumes = ndimage.sum(binary_simulacion, labeled_structures, range(1, num_structures + 1))
    # Calcular índices de las estructuras a ser eliminadas
    del_structure = np.argwhere(structure_volumes < (4 / 3) * np.pi * pow(2, 3))
    # Eliminar estructuras estableciendo los valores correspondientes en Falso
    binary_simulacion[np.isin(labeled_structures, del_structure + 1)] = False

    labeled_structures, num_structures = ndimage.label(binary_simulacion)
    structure_centers = ndimage.center_of_mass(binary_simulacion, labeled_structures, range(1, num_structures + 1))
    structure_areas = ndimage.sum(binary_simulacion, labeled_structures, range(1, num_structures + 1))
    bounding_boxes = ndimage.find_objects(labeled_structures)

    return bounding_boxes, binary_simulacion 


def generar_trayectoria(coord_inicial, coord_final,V):

    # Calcular la distancia y el vector de dirección entre los puntos
    distancia = np.linalg.norm(coord_final - coord_inicial)
    direccion = (coord_final - coord_inicial) / distancia

    trayectoria = []
    # Generar la trayectoria recta
    for i in range(0,int(distancia), V):
        punto = coord_inicial + i * direccion
        punto = punto.astype(int)
        trayectoria.append(punto)
    return np.array(trayectoria)



def four_path(zona,V=2):
  # zona path 1

  start_11 = np.array([zona[0].start,zona[1].start,zona[2].start])
  end_11 = np.array([zona[0].stop,zona[1].stop,zona[2].stop])

  path_11 = generar_trayectoria(start_11, end_11, V)

  # zona path 2

  start_12 = np.array([(zona[0].start + zona[0].stop) / 2, zona[1].start, (zona[2].start + zona[2].stop) / 2])
  end_12 = np.array([(zona[0].start + zona[0].stop) / 2, zona[1].stop, (zona[2].start + zona[2].stop) / 2])

  path_12 = generar_trayectoria(start_12, end_12, V)

  # zona path 3

  start_13 = np.array([zona[0].start, (zona[1].start + zona[1].stop) / 2, (zona[2].start + zona[2].stop) / 2])
  end_13 = np.array([zona[0].stop, (zona[1].start + zona[1].stop) / 2, (zona[2].start + zona[2].stop) / 2])

  path_13 = generar_trayectoria(start_13, end_13, V)

  # zona path 4

  start_14 = np.array([(zona[0].start + zona[0].stop) / 2, (zona[1].start + zona[1].stop) / 2, zona[2].start])
  end_14 = np.array([(zona[0].start + zona[0].stop) / 2, (zona[1].start + zona[1].stop) / 2, zona[2].stop])

  path_14 = generar_trayectoria(start_14, end_14, V)

  return path_11,path_12,path_13,path_14


def plot_paths(path_11, path_12, path_13, path_14):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    paths = [path_11, path_12, path_13, path_14]
    colors = ['k', 'b', 'r', 'g']

    for path, color in zip(paths, colors):
        x_coords = [point[0] for point in path]
        y_coords = [point[1] for point in path]
        z_coords = [point[2] for point in path]
        ax.scatter(x_coords, y_coords, z_coords, c=color)

    ax.set_xlabel('Z')
    ax.set_ylabel('Y')
    ax.set_zlabel('X')

    plt.show()
    return None

def plot_vectores(zona_1,vec_11,vec_12,vec_13,vec_14,binary_zona1):

  fig = plt.figure()
  ax = fig.add_subplot(111, projection='3d')

  center_x = np.abs(zona_1[0].start-zona_1[0].stop)/2
  center_y = np.abs(zona_1[1].start-zona_1[1].stop)/2
  center_z = np.abs(zona_1[2].start-zona_1[2].stop)/2

  for ve in vec_11:
      ax.quiver(center_x, center_y,center_z, ve[0]*20, ve[1]*20, ve[2]*20, color="k")
  for ve in vec_12:
      ax.quiver(center_x, center_y,center_z, ve[0]*20, ve[1]*20, ve[2]*20, color="b")
  for ve in vec_13:
      ax.quiver(center_x, center_y,center_z, ve[0]*20, ve[1]*20, ve[2]*20, color="r")
  for ve in vec_14:
      ax.quiver(center_x, center_y,center_z, ve[0]*20, ve[1]*20, ve[2]*20, color="g")

  ax.text(center_x+vec_11[0][0]*30, center_y+vec_11[0][1]*30, center_z+vec_11[0][2]*30, "L", color="k",fontsize=20)
  ax.text(center_x+vec_11[1][0]*30, center_y+vec_11[1][1]*30, center_z+vec_11[1][2]*30, "M", color="k",fontsize=20)
  ax.text(center_x+vec_11[2][0]*30, center_y+vec_11[2][1]*30, center_z+vec_11[2][2]*30, "N", color="k",fontsize=20)

  ax.voxels(binary_zona1,alpha=0.1)

  # Personalizar la gráfica
  ax.set_xlabel('Z',fontsize=15)
  ax.set_ylabel('y',fontsize=15)
  ax.set_zlabel('X',fontsize=15)

  ax.view_init(azim=45)

  plt.show()

  return None

def U_psi(zona,vec,array,jx_i,jy_i,jz_i,hx_i,hy_i,hz_i):

  U_psi_L = np.zeros(array.shape)
  U_psi_N = np.zeros(array.shape)

  # Recorrer todos los elementos del arreglo
  for i in range(zona[0].start, zona[0].stop):
    for j in range(zona[1].start, zona[1].stop):
      for k in range(zona[2].start, zona[2].stop):
          E = [jx_i[i, j, k],jy_i[i, j, k],jz_i[i, j, k]]
          B = [hx_i[i, j, k],hy_i[i, j, k],hz_i[i, j, k]]
          U_psi_L[i,j,k] = velocity(E,B,vec)[0]
          U_psi_N[i,j,k] = velocity(E,B,vec)[2]

  return U_psi_L, U_psi_N

def plot_velocity(array, zona):

    # Obtener las dimensiones del arreglo
    x_dim, y_dim, z_dim = array[tuple(zona)].shape

    # Crear una figura en 3D
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Crear una rejilla tridimensional de puntos
    x, y, z = np.meshgrid(range(x_dim), range(y_dim), range(z_dim))

    # Obtener los valores del arreglo en cada punto de la rejilla
    values = array[tuple(zona)][x, y, z]

    vmin = np.min(array)
    vmax = np.max(array)
    norm = colors.TwoSlopeNorm( vcenter=0)
    # Graficar la figura en 3D
    scatter = ax.scatter(x, y, z, c=values,cmap='seismic',norm=norm)
    ax.set_xlabel('N')
    ax.set_ylabel('M')
    ax.set_zlabel('L')

    # Agregar una barra de color
    cbar = plt.colorbar(scatter)

    # Mostrar la figura
    plt.show()
    return None

def plot_vectores_3d(zona, vec11, vec12, vec13, vec14, binary_zona, path_11, path_12, path_13, path_14):

    fig = go.Figure()

    center_x = (zona[0].start + zona[0].stop) / 2
    center_y = (zona[1].start + zona[1].stop) / 2
    center_z = (zona[2].start + zona[2].stop) / 2

    colors = ['black', 'blue', 'red', 'green']
    mark = ["solid","longdashdot","dashdot"]
    vectors = [vec11, vec12, vec13, vec14]
    colors_l = ['silver', 'cornflowerblue', 'lightcoral', 'lightgreen']
    for i, vec_group in enumerate(vectors):
        for j,ve in enumerate(vec_group):
            fig.add_trace(go.Scatter3d(
                x=[center_x, center_x + ve[2] * 20],
                y=[center_y, center_y + ve[1] * 20],
                z=[center_z, center_z + ve[0] * 20],
                mode='lines',
                line=dict(color=colors_l[i], width=5),
                #line=dict(color=colors[i],width=5),
                showlegend=False
            ))

    for i, vec_group in enumerate(vectors):
        for j,ve in enumerate(vec_group):
            fig.add_trace(go.Scatter3d(
                x=[center_x, center_x + ve[2] * 20],
                y=[center_y, center_y + ve[1] * 20],
                z=[center_z, center_z + ve[0] * 20],
                mode='lines',
                line=dict(color=colors[i], width=7, dash=mark[j]),
                #line=dict(color=colors[i],width=5),
                showlegend=False
            ))

    labels = ['L', 'M', 'N']
    for i, label in enumerate(labels):
        fig.add_trace(go.Scatter3d(
            x=[center_x, center_x + vec11[i][2] * 30],
            y=[center_y, center_y + vec11[i][1] * 30],
            z=[center_z, center_z + vec11[i][0] * 30],
            mode='text',
            text=label,
            textposition='top center',
            textfont=dict(size=12, color='black'),
            showlegend=False
        ))

    binary_zona_coords = np.array(np.where(binary_zona)).T.astype(float)
    binary_zona_coords[:, :3] += np.array([zona[0].start, zona[1].start, zona[2].start])
    fig.add_trace(go.Scatter3d(
        x=binary_zona_coords[:, 0],
        y=binary_zona_coords[:, 1],
        z=binary_zona_coords[:, 2],
        mode='markers',
        marker=dict(size=1, color='blue', opacity=0.4),
        showlegend=False
    ))

    # Agregar los scatter plots de los paths
    fig.add_trace(go.Scatter3d(
        x=[point[0] for point in path_11],
        y=[point[1] for point in path_11],
        z=[point[2] for point in path_11],
        mode='markers',
        marker=dict(size=3, color='black',opacity=0.2),
        showlegend=False
    ))

    fig.add_trace(go.Scatter3d(
        x=[point[0] for point in path_12],
        y=[point[1] for point in path_12],
        z=[point[2] for point in path_12],
        mode='markers',
        marker=dict(size=3, color='blue',opacity=0.2),
        showlegend=False
    ))

    fig.add_trace(go.Scatter3d(
        x=[point[0] for point in path_13],
        y=[point[1] for point in path_13],
        z=[point[2] for point in path_13],
        mode='markers',
        marker=dict(size=3, color='red',opacity=0.2),
        showlegend=False
    ))

    fig.add_trace(go.Scatter3d(
        x=[point[0] for point in path_14],
        y=[point[1] for point in path_14],
        z=[point[2] for point in path_14],
        mode='markers',
        marker=dict(size=3, color='green',opacity=0.15),
        showlegend=False
    ))

    fig.update_layout(
        scene=dict(
            xaxis=dict(title='Z'),
            yaxis=dict(title='Y'),
            zaxis=dict(title='X'),
            aspectmode='data',
        ),
        showlegend=False
    )

    fig.show()

    return None


def change_of_basis_zone(zona,vec,array,array_x,array_y,array_z):

  j_psi_L = np.zeros(array.shape)
  j_psi_M = np.zeros(array.shape)
  j_psi_N = np.zeros(array.shape)

  # Recorrer todos los elementos del arreglo
  for i in range(zona[0].start, zona[0].stop):
    for j in range(zona[1].start, zona[1].stop):
      for k in range(zona[2].start, zona[2].stop):
          E = [array_x[i, j, k],array_y[i, j, k],array_z[i, j, k]]
          E_LMN = change_of_basis(E,basis_old,vec)
          j_psi_L[i,j,k] = E_LMN[0]
          j_psi_M[i,j,k] = E_LMN[1]
          j_psi_N[i,j,k] = E_LMN[2]

  j_mag_lmn = np.sqrt(j_psi_L*j_psi_L + j_psi_M*j_psi_M + j_psi_N*j_psi_N)
  return j_psi_L, j_psi_M, j_psi_N, j_mag_lmn

def divergence(U_psi_L,U_psi_N,array):
  U_psi_M = np.zeros((array.shape))

  u = U_psi_L
  v = U_psi_M
  w = U_psi_N

  # Calculate the partial derivatives
  du_dx = np.gradient(u,0.06,axis=0)
  dv_dy = np.gradient(v,0.06,axis=1)
  dw_dz = np.gradient(w,0.06,axis=2)

  # Calculate the divergence
  div = np.add.reduce([du_dx, dv_dy, dw_dz])
  return div


def animation_zones(array, zona_i):

  # Initialize figure and axis
  fig, ax = plt.subplots()

  # Set number of slices to animate
  num_slices = array[zona_i].shape[0]

  # Define update function for animation
  def update(frame):
      ax.clear()
      ax.imshow(array[zona_i][frame, :, :], cmap='seismic')
      ax.set_title("Slice {}".format(frame))

  # Create animation
  anim = FuncAnimation(fig, update, frames=num_slices, interval=50)

  return anim

def figure_path_3d(zona_i,vec,path,binary_zona,name,color="red"):
  fig = go.Figure()

  center_x = (zona_i[0].start + zona_i[0].stop) / 2
  center_y = (zona_i[1].start + zona_i[1].stop) / 2
  center_z = (zona_i[2].start + zona_i[2].stop) / 2

  labels = ['L', 'M', 'N']
  for i, label in enumerate(labels):
        fig.add_trace(go.Scatter3d(
            x=[center_x, center_x + vec[i][2] * 30],
            y=[center_y, center_y + vec[i][1] * 30],
            z=[center_z, center_z + vec[i][0] * 30],
            mode='text',
            text=label,
            textposition='top center',
            textfont=dict(size=12, color='black'),
            showlegend=False
        ))

  binary_zona_coords = np.array(np.where(binary_zona)).T.astype(float)
  binary_zona_coords[:, :3] += np.array([zona_i[0].start, zona_i[1].start, zona_i[2].start])

  fig.add_trace(go.Scatter3d(
        x=binary_zona_coords[:, 0],
        y=binary_zona_coords[:, 1],
        z=binary_zona_coords[:, 2],
        mode='markers',
        marker=dict(size=3, color='blue', opacity=0.05),
        showlegend=False
    ))

    # Agregar los scatter plots de los paths
  fig.add_trace(go.Scatter3d(
        x=[point[0] for point in path],
        y=[point[1] for point in path],
        z=[point[2] for point in path],
        mode='markers',
        marker=dict(size=2, color=color,opacity=0.5),
        showlegend=False
    ))

  for j,ve in enumerate(vec):
            fig.add_trace(go.Scatter3d(
                x=[center_x, center_x + ve[2] * 20],
                y=[center_y, center_y + ve[1] * 20],
                z=[center_z, center_z + ve[0] * 20],
                mode='lines',
                line=dict(color="red", width=5),
                #line=dict(color=colors[i],width=5),
                showlegend=False
            ))

  fig.update_layout(
        scene=dict(
            xaxis=dict(title='Z'),
            yaxis=dict(title='Y'),
            zaxis=dict(title='X'),
            aspectmode='data',
        ),
        showlegend=False
    )
  fig.show()


  # Create an animation scene
  frames = [go.Frame(data=[go.Scatter3d(x=[point[0] for point in path[:i]],
                                     y=[point[1] for point in path[:i]],
                                     z=[point[2] for point in path[:i]],
                                     mode='markers',
                                     marker=dict(size=2, color=color, opacity=0.5),
                                     showlegend=False
                                     )],
                  name=str(i)
                  )
          for i in range(1, len(path) + 1)]

  # Add frames to the layout
  fig.update(frames=frames)

  # Create an animation configuration
  animation_settings = dict(frame=dict(duration=100, redraw=True), fromcurrent=True)

  # Update layout to enable animation and specify animation settings
  fig.update_layout(updatemenus=[dict(type='buttons', showactive=False, buttons=[dict(label='Play',
                                                                                method='animate',
                                                                                args=[None,
                                                                                      animation_settings])])])

  # Save the animation as an HTML file
  fig.write_html(name+'.html')
  return None