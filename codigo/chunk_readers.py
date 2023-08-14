#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep  1 18:15:51 2020
Inputs: h_field,n_chuncks,n_quarter 
This is the function to read the file by chunks.
The input should be a group, 2, 4 or 8 
@author: jeffersson_agudelo
"""

# Function to get a field with 3 components as the magnetic field
# Magnetic field components
# hx as a group; hx as a group; hx as a dataset; hx.size returns the whole number of cells, hx.shape returns the numbers by axis
# Read just a segment   
############################################################################################################################
def chunk_reader_h(hx,hy,hz,n_chunks,n_quarter):
    """
    This is the function to read field variables hx,hy,hz by chunks from a large file.
    Inputs: hx,hy,hz,n_chuncks,n_quarter 
    The input should be a group. The options for chunks currently are 2, 4 or 8. 
    The quarter shoulb be between 0 and n_chunks-1.
    """
    hx=hx['p0'] ; hx=hx['3d'] ;
    hy=hy['p0'] ; hy=hy['3d'] ;
    hz=hz['p0'] ; hz=hz['3d'] ;

    # All the arrays will have the same dimensions. 
    # Create chunks by quarters
    
    file_shape = hx.shape;
    #n_quarter=0;    n_chunks=4;
    nth_chunk=range(0,n_chunks+1);
    
    #--------------------------------------------------------------------------
    if(n_chunks==1):
    # Size of chunks by coordinates
        chunk_0 = int(file_shape[0]);
        chunk_1 = int(file_shape[1]);
        chunk_2 = int(file_shape[2]);   
    elif (n_chunks==2):
    # Size of chunks by coordinates
        chunk_0 = int(file_shape[0]/2);
        chunk_1 = int(file_shape[1]);
        chunk_2 = int(file_shape[2]);    
    elif (n_chunks==4):
    # Size of chunks by coordinates
        chunk_0 = int(file_shape[0]/2);
        chunk_1 = int(file_shape[1]/2);
        chunk_2 = int(file_shape[2]);
    elif (n_chunks==8):
    # Size of chunks by coordinates
        chunk_0 = int(file_shape[0]/2);
        chunk_1 = int(file_shape[1]/2);
        chunk_2 = int(file_shape[2]/2);
    
    #--------------------------------------------------------------------------    
    if (n_chunks==2 and n_quarter==0):
        hx_0=hx[nth_chunk[0]*chunk_0:nth_chunk[1]*chunk_0,:,:];
        hy_0=hy[nth_chunk[0]*chunk_0:nth_chunk[1]*chunk_0,:,:];
        hz_0=hz[nth_chunk[0]*chunk_0:nth_chunk[1]*chunk_0,:,:];
    elif (n_chunks==2 and n_quarter==1):
        hx_0=hx[nth_chunk[1]*chunk_0:nth_chunk[2]*chunk_0,:,:];
        hy_0=hy[nth_chunk[1]*chunk_0:nth_chunk[2]*chunk_0,:,:];
        hz_0=hz[nth_chunk[1]*chunk_0:nth_chunk[2]*chunk_0,:,:];
    #--------------------------------------------------------------------------        
    elif (n_quarter==0):
        hx_0=hx[nth_chunk[0]*chunk_0:nth_chunk[1]*chunk_0, nth_chunk[0]*chunk_1:nth_chunk[1]*chunk_1, nth_chunk[0]*chunk_2:nth_chunk[1]*chunk_2];
        hy_0=hy[nth_chunk[0]*chunk_0:nth_chunk[1]*chunk_0, nth_chunk[0]*chunk_1:nth_chunk[1]*chunk_1, nth_chunk[0]*chunk_2:nth_chunk[1]*chunk_2];
        hz_0=hz[nth_chunk[0]*chunk_0:nth_chunk[1]*chunk_0, nth_chunk[0]*chunk_1:nth_chunk[1]*chunk_1, nth_chunk[0]*chunk_2:nth_chunk[1]*chunk_2];
    elif (n_quarter==1):
        hx_0=hx[nth_chunk[0]*chunk_0:nth_chunk[1]*chunk_0, nth_chunk[1]*chunk_1:nth_chunk[2]*chunk_1, nth_chunk[0]*chunk_2:nth_chunk[1]*chunk_2];        
        hy_0=hy[nth_chunk[0]*chunk_0:nth_chunk[1]*chunk_0, nth_chunk[1]*chunk_1:nth_chunk[2]*chunk_1, nth_chunk[0]*chunk_2:nth_chunk[1]*chunk_2];
        hz_0=hz[nth_chunk[0]*chunk_0:nth_chunk[1]*chunk_0, nth_chunk[1]*chunk_1:nth_chunk[2]*chunk_1, nth_chunk[0]*chunk_2:nth_chunk[1]*chunk_2];
    elif (n_quarter==2):
        hx_0=hx[nth_chunk[1]*chunk_0:nth_chunk[2]*chunk_0, nth_chunk[0]*chunk_1:nth_chunk[1]*chunk_1, nth_chunk[0]*chunk_2:nth_chunk[1]*chunk_2];        
        hy_0=hy[nth_chunk[1]*chunk_0:nth_chunk[2]*chunk_0, nth_chunk[0]*chunk_1:nth_chunk[1]*chunk_1, nth_chunk[0]*chunk_2:nth_chunk[1]*chunk_2];
        hz_0=hz[nth_chunk[1]*chunk_0:nth_chunk[2]*chunk_0, nth_chunk[0]*chunk_1:nth_chunk[1]*chunk_1, nth_chunk[0]*chunk_2:nth_chunk[1]*chunk_2];
    elif (n_quarter==3):
        hx_0=hx[nth_chunk[1]*chunk_0:nth_chunk[2]*chunk_0, nth_chunk[1]*chunk_1:nth_chunk[2]*chunk_1, nth_chunk[0]*chunk_2:nth_chunk[1]*chunk_2];        
        hy_0=hy[nth_chunk[1]*chunk_0:nth_chunk[2]*chunk_0, nth_chunk[1]*chunk_1:nth_chunk[2]*chunk_1, nth_chunk[0]*chunk_2:nth_chunk[1]*chunk_2];
        hz_0=hz[nth_chunk[1]*chunk_0:nth_chunk[2]*chunk_0, nth_chunk[1]*chunk_1:nth_chunk[2]*chunk_1, nth_chunk[0]*chunk_2:nth_chunk[1]*chunk_2];
    elif (n_quarter==4):
        hx_0=hx[nth_chunk[0]*chunk_0:nth_chunk[1]*chunk_0, nth_chunk[0]*chunk_1:nth_chunk[1]*chunk_1, nth_chunk[1]*chunk_2:nth_chunk[2]*chunk_2];
        hy_0=hy[nth_chunk[0]*chunk_0:nth_chunk[1]*chunk_0, nth_chunk[0]*chunk_1:nth_chunk[1]*chunk_1, nth_chunk[1]*chunk_2:nth_chunk[2]*chunk_2];
        hz_0=hz[nth_chunk[0]*chunk_0:nth_chunk[1]*chunk_0, nth_chunk[0]*chunk_1:nth_chunk[1]*chunk_1, nth_chunk[1]*chunk_2:nth_chunk[2]*chunk_2];
    elif (n_quarter==5):
        hx_0=hx[nth_chunk[0]*chunk_0:nth_chunk[1]*chunk_0, nth_chunk[1]*chunk_1:nth_chunk[2]*chunk_1, nth_chunk[1]*chunk_2:nth_chunk[2]*chunk_2];        
        hy_0=hy[nth_chunk[0]*chunk_0:nth_chunk[1]*chunk_0, nth_chunk[1]*chunk_1:nth_chunk[2]*chunk_1, nth_chunk[1]*chunk_2:nth_chunk[2]*chunk_2];
        hz_0=hz[nth_chunk[0]*chunk_0:nth_chunk[1]*chunk_0, nth_chunk[1]*chunk_1:nth_chunk[2]*chunk_1, nth_chunk[1]*chunk_2:nth_chunk[2]*chunk_2];
    elif (n_quarter==6):
        hx_0=hx[nth_chunk[1]*chunk_0:nth_chunk[2]*chunk_0, nth_chunk[0]*chunk_1:nth_chunk[1]*chunk_1, nth_chunk[1]*chunk_2:nth_chunk[2]*chunk_2];        
        hy_0=hy[nth_chunk[1]*chunk_0:nth_chunk[2]*chunk_0, nth_chunk[0]*chunk_1:nth_chunk[1]*chunk_1, nth_chunk[1]*chunk_2:nth_chunk[2]*chunk_2];
        hz_0=hz[nth_chunk[1]*chunk_0:nth_chunk[2]*chunk_0, nth_chunk[0]*chunk_1:nth_chunk[1]*chunk_1, nth_chunk[1]*chunk_2:nth_chunk[2]*chunk_2];
    elif (n_quarter==7):
        hx_0=hx[nth_chunk[1]*chunk_0:nth_chunk[2]*chunk_0, nth_chunk[1]*chunk_1:nth_chunk[2]*chunk_1, nth_chunk[1]*chunk_2:nth_chunk[2]*chunk_2];        
        hy_0=hy[nth_chunk[1]*chunk_0:nth_chunk[2]*chunk_0, nth_chunk[1]*chunk_1:nth_chunk[2]*chunk_1, nth_chunk[1]*chunk_2:nth_chunk[2]*chunk_2];
        hz_0=hz[nth_chunk[1]*chunk_0:nth_chunk[2]*chunk_0, nth_chunk[1]*chunk_1:nth_chunk[2]*chunk_1, nth_chunk[1]*chunk_2:nth_chunk[2]*chunk_2];
    
    return hx_0, hy_0, hz_0 
############################################################################################################################  



# Function to get scalars as the density
############################################################################################################################
def chunk_reader_n(hx,n_chunks,n_quarter):
    """
    This is the function to read a variable hx by chunks from a large file.
    Inputs: hx,n_chuncks,n_quarter 
    The input should be a group. The options for chunks currently are 2, 4 or 8. 
    The quarter shoulb be between 0 and n_chunks-1.
    """    
    hx=hx['p0'] ; hx=hx['3d'] ;
    # Create chunks by quarters
    
    file_shape = hx.shape;
    #n_quarter=0;    n_chunks=4;
    nth_chunk=range(0,n_chunks+1);    
    if (n_chunks==1):
    # Size of chunks by coordinates
        chunk_0 = int(file_shape[0]);
        chunk_1 = int(file_shape[1]);
        chunk_2 = int(file_shape[2]); 
    elif (n_chunks==2):
    # Size of chunks by coordinates
        chunk_0 = int(file_shape[0]/2);
        chunk_1 = int(file_shape[1]);
        chunk_2 = int(file_shape[2]);    
    elif (n_chunks==4):
    # Size of chunks by coordinates
        chunk_0 = int(file_shape[0]/2);
        chunk_1 = int(file_shape[1]/2);
        chunk_2 = int(file_shape[2]);
    elif (n_chunks==8):
    # Size of chunks by coordinates
        chunk_0 = int(file_shape[0]/2);
        chunk_1 = int(file_shape[1]/2);
        chunk_2 = int(file_shape[2]/2);
    
    # load quater   
    #--------------------------------------------------------------------------    
    if (n_chunks==2 and n_quarter==0):
        hx_0=hx[nth_chunk[0]*chunk_0:nth_chunk[1]*chunk_0,:,:];
    elif (n_chunks==2 and n_quarter==1):
        hx_0=hx[nth_chunk[1]*chunk_0:nth_chunk[2]*chunk_0,:,:];
    #--------------------------------------------------------------------------        
    elif (n_quarter==0):
        hx_0=hx[nth_chunk[0]*chunk_0:nth_chunk[1]*chunk_0, nth_chunk[0]*chunk_1:nth_chunk[1]*chunk_1, nth_chunk[0]*chunk_2:nth_chunk[1]*chunk_2];
    elif (n_quarter==1):
        hx_0=hx[nth_chunk[0]*chunk_0:nth_chunk[1]*chunk_0, nth_chunk[1]*chunk_1:nth_chunk[2]*chunk_1, nth_chunk[0]*chunk_2:nth_chunk[1]*chunk_2];        
    elif (n_quarter==2):
        hx_0=hx[nth_chunk[1]*chunk_0:nth_chunk[2]*chunk_0, nth_chunk[0]*chunk_1:nth_chunk[1]*chunk_1, nth_chunk[0]*chunk_2:nth_chunk[1]*chunk_2];        
    elif (n_quarter==3):
        hx_0=hx[nth_chunk[1]*chunk_0:nth_chunk[2]*chunk_0, nth_chunk[1]*chunk_1:nth_chunk[2]*chunk_1, nth_chunk[0]*chunk_2:nth_chunk[1]*chunk_2];        
    elif (n_quarter==4):
        hx_0=hx[nth_chunk[0]*chunk_0:nth_chunk[1]*chunk_0, nth_chunk[0]*chunk_1:nth_chunk[1]*chunk_1, nth_chunk[1]*chunk_2:nth_chunk[2]*chunk_2];
    elif (n_quarter==5):
        hx_0=hx[nth_chunk[0]*chunk_0:nth_chunk[1]*chunk_0, nth_chunk[1]*chunk_1:nth_chunk[2]*chunk_1, nth_chunk[1]*chunk_2:nth_chunk[2]*chunk_2];        
    elif (n_quarter==6):
        hx_0=hx[nth_chunk[1]*chunk_0:nth_chunk[2]*chunk_0, nth_chunk[0]*chunk_1:nth_chunk[1]*chunk_1, nth_chunk[1]*chunk_2:nth_chunk[2]*chunk_2];        
    elif (n_quarter==7):
        hx_0=hx[nth_chunk[1]*chunk_0:nth_chunk[2]*chunk_0, nth_chunk[1]*chunk_1:nth_chunk[2]*chunk_1, nth_chunk[1]*chunk_2:nth_chunk[2]*chunk_2];        
        
    return hx_0 
############################################################################################################################




# Function to get the coordinates
############################################################################################################################
def chunk_reader_crd(hx,hy,hz,n_chunks,n_quarter):
    """
    This is the function to read a field variables hx,hy,hz by chunks from a large file.
    Inputs: hx,hy,hz,n_chuncks,n_quarter 
    The input should be a group. The options for chunks currently are 2, 4 or 8. 
    The quarter shoulb be between 0 and n_chunks-1.
    """
    hx=hx['crd[2]'] ; hx=hx['p0'] ; hx=hx['1d'] ; # [z,y,x]
    hy=hy['crd[1]'] ; hy=hy['p0'] ; hy=hy['1d'] ;
    hz=hz['crd[0]'] ; hz=hz['p0'] ; hz=hz['1d'] ;

    # All the arrays will have the same dimensions. 
    # Create chunks by quarters
    
    file_shape0 = hx.shape;
    file_shape1 = hy.shape;
    file_shape2 = hz.shape;
    #n_quarter=0;    n_chunks=4;
    nth_chunk=range(0,n_chunks+1);
    
    
    if(n_chunks==1):
    # Size of chunks by coordinates
        chunk_0 = int(file_shape0[0]);
        chunk_1 = int(file_shape1[0]);
        chunk_2 = int(file_shape2[0]);   
    elif (n_chunks==2):
    # Size of chunks by coordinates
        chunk_0 = int(file_shape0[0]/2);
        chunk_1 = int(file_shape1[0]);
        chunk_2 = int(file_shape2[0]);    
    elif (n_chunks==4):
    # Size of chunks by coordinates
        chunk_0 = int(file_shape0[0]/2);
        chunk_1 = int(file_shape1[0]/2);
        chunk_2 = int(file_shape2[0]);
    elif (n_chunks==8):
    # Size of chunks by coordinates
        chunk_0 = int(file_shape0[0]/2);
        chunk_1 = int(file_shape1[0]/2);
        chunk_2 = int(file_shape2[0]/2);

    # load quater   
    #--------------------------------------------------------------------------    
    if (n_chunks==2 and n_quarter==0):
        hx_0=hx[nth_chunk[0]*chunk_0:nth_chunk[1]*chunk_0];
        hy_0=hy[:];
        hz_0=hz[:];
    elif (n_chunks==2 and n_quarter==1):
        hx_0=hx[nth_chunk[1]*chunk_0:nth_chunk[2]*chunk_0];
        hy_0=hy[:];
        hz_0=hz[:];
    #-------------------------------------------------------------------------- 
    elif (n_quarter==0):
        hx_0=hx[nth_chunk[0]*chunk_0:nth_chunk[1]*chunk_0];
        hy_0=hy[nth_chunk[0]*chunk_1:nth_chunk[1]*chunk_1];
        hz_0=hz[nth_chunk[0]*chunk_2:nth_chunk[1]*chunk_2];
    elif (n_quarter==1):
        hx_0=hx[nth_chunk[0]*chunk_0:nth_chunk[1]*chunk_0];        
        hy_0=hy[nth_chunk[1]*chunk_1:nth_chunk[2]*chunk_1];
        hz_0=hz[nth_chunk[0]*chunk_2:nth_chunk[1]*chunk_2];
    elif (n_quarter==2):
        hx_0=hx[nth_chunk[1]*chunk_0:nth_chunk[2]*chunk_0];        
        hy_0=hy[nth_chunk[0]*chunk_1:nth_chunk[1]*chunk_1];
        hz_0=hz[nth_chunk[0]*chunk_2:nth_chunk[1]*chunk_2];
    elif (n_quarter==3):
        hx_0=hx[nth_chunk[1]*chunk_0:nth_chunk[2]*chunk_0];        
        hy_0=hy[nth_chunk[1]*chunk_1:nth_chunk[2]*chunk_1];
        hz_0=hz[nth_chunk[0]*chunk_2:nth_chunk[1]*chunk_2];
    elif (n_quarter==4):
        hx_0=hx[nth_chunk[0]*chunk_0:nth_chunk[1]*chunk_0];
        hy_0=hy[nth_chunk[0]*chunk_1:nth_chunk[1]*chunk_1];
        hz_0=hz[nth_chunk[1]*chunk_2:nth_chunk[2]*chunk_2];
    elif (n_quarter==5):
        hx_0=hx[nth_chunk[0]*chunk_0:nth_chunk[1]*chunk_0];        
        hy_0=hy[nth_chunk[1]*chunk_1:nth_chunk[2]*chunk_1];
        hz_0=hz[nth_chunk[1]*chunk_2:nth_chunk[2]*chunk_2];
    elif (n_quarter==6):
        hx_0=hx[nth_chunk[1]*chunk_0:nth_chunk[2]*chunk_0];        
        hy_0=hy[nth_chunk[0]*chunk_1:nth_chunk[1]*chunk_1];
        hz_0=hz[nth_chunk[1]*chunk_2:nth_chunk[2]*chunk_2];
    elif (n_quarter==7):
        hx_0=hx[nth_chunk[1]*chunk_0:nth_chunk[2]*chunk_0];        
        hy_0=hy[nth_chunk[1]*chunk_1:nth_chunk[2]*chunk_1];
        hz_0=hz[nth_chunk[1]*chunk_2:nth_chunk[2]*chunk_2];
    
    return hx_0, hy_0, hz_0 # returns z, y, x
############################################################################################################################  




# Function to get particle information
############################################################################################################################
def chunk_reader_p(h5_1stp,n_div,n_quarter):
    """
    This is the function to read a particle output by chunks from a large file.
    This is done using a single running index and not using several indexes as in the fields case
    Inputs: h5_1stp,n_div,n_quarter. For large ppc use large numbers for n_div. i.e., n_div =10000 
    The quarter shoulb be between 0 and n_chunks-1.
    """
    #n_div = 10000;    n_quarter = 0;    h5_1stp = h5py.File(filenamep, 'r') 
    import numpy as np
    
    h5_1stp=h5_1stp['particles'];
    h5_1stp=h5_1stp['p0']; # ['1d', 'idx_begin', 'idx_end']
    data=h5_1stp['1d'];  # Thesefiles are saved in x,y,z order
    
    #idx_begin=h5_1stp['idx_begin']; #This one gives the size of the box in terms of the indexes 
    #idx_end=h5_1stp['idx_end'];
    #file_shapep = idx_begin.shape; # Thesefiles are saved in x,y,z order

    # Size of chunks by coordinates
    file_length = data.size;
    chunk_0 = int(file_length/n_div); 
    nth_chunk=range(0,n_div);  

    a=nth_chunk[n_quarter]*chunk_0
    b=nth_chunk[n_quarter+1]*chunk_0

    r = range(a, b)
    r = [*r]
    data2=data[r]
    
    #Separate by charge
    data_e_j=np.where(data2['q'] == -1) 
    data_e=data2[data_e_j]
    data_i_j=np.where(data2['q'] == 1)
    data_i=data2[data_i_j]

    #x_e=data_e['x']; y_e=data_e['y']; z_e=data_e['z'];
    #x_i=data_i['x']; y_i=data_i['y']; z_i=data_i['z'];
    
    vx_e=data_e['px']; vy_e=data_e['py'] ; vz_e=data_e['pz']; 
    vx_i=data_i['px']; vy_i=data_i['py'] ; vz_i=data_i['pz']; 
    
    #return x_e, y_e, z_e, vx_e, vy_e, vz_e, x_i, y_i, z_i, vx_i, vy_i, vz_i       
    return vx_e, vy_e, vz_e, vx_i, vy_i, vz_i   
############################################################################################################################



## Create a `Summation` class
#class Summation(object):
#  def sum(self, a, b):
#    self.contents = a + b
#    return self.contents 
#
## Instantiate `Summation` class to call `sum()`
#sumInstance = Summation()
#sumInstance.sum(1,2)
