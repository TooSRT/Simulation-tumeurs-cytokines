"""
Density EDP Module

This module provides a class Density_EDP for tumor density simulation.

Authors: William LANGUE, Théo DUFRESNE
Date: 17/03/2024

Usage:
    from Simulation.density_edp import Density_EDP

    # Example usage:
    density_edp = Density_EDP(nb_cells=100, Nx=grid_side_length, cells0=initial_cells, n0=initial_density, w_max=max_density_value, delta_x=spatial_step_size, delta_t=time_step_size, Dn=diffusion_coefficient, rn=division_rate)
    cellsmouv, cellspro, choice = density_edp.proliferation()
"""

import time
import numpy as np

class Density_EDP:
    """
    Density_EDP class

    This class represents a model for tumor growth using the HDC-EDP method.

    Attributes:
        nb_cells (int): Number of cells in the grid.
        Nx (int): Size of the grid.
        delta_t (int): Time step size (in hours).
        delta_x (float): Spatial step size.
        w_max (int): Maximum density value.
        Dn (float): Diffusion coefficient.
        rn (float): Division rate (in hours).
        cells (numpy.ndarray): Vector of cancer cells.
        n (numpy.ndarray): Vector of densities.
        cells_size (numpy.ndarray): List of the number of cells per iteration.
    """

    def __init__(self, Nx, cells0, w0, T0, n0, w_max, delta_x, delta_t, Dn, rn):
        """
        Initialize a Density_EDP object.

        Args:
            nb_cells (int): Number of cells in the grid.
            Nx (int): Size of the grid.
            cells0 (numpy.ndarray): Initial cancer cell vector.
            w0 (numpy.ndarray): Initial density vector (T-cells + tumor).
            n0 (numpy.ndarray): Initial density vector (tumor).
            T0 (numpy.ndarray): Initial density vector (T-cells).
            w_max (int): Maximum density value.
            delta_x (float): Spatial step size.
            delta_t (int): Time step size (in hours).
            Dn (float): Diffusion coefficient.
            rn (float): Division rate (in hours).
        """
        self.Nx = Nx
        self.cells = cells0 
        self.w = w0 
        self.T = T0
        self.n = n0
        self.w_max = w_max
        self.delta_x = delta_x
        self.delta_t = delta_t
        self.Dn = Dn
        self.rn = rn 
        self.cells_size = np.array([len(cells0)]) 
    
    def Nx(self):
        """
        Get the size of the grid.

        Returns:
            int: Size of the grid.
        """
        return self.Nx

    def delta_t(self):
        """
        Get the time step size.

        Returns:
            int: Time step size in hours.
        """
        return self.delta_t
    
    def delta_x(self):
        """
        Get the spatial step size.

        Returns:
            float: Spatial step size.
        """
        return self.delta_x

    def w_max(self):
        """
        Get the maximum density value.

        Returns:
            int: Maximum density value.
        """
        return self.w_max
    
    def Dn(self):
        """
        Get the diffusion coefficient.

        Returns:
            float: Diffusion coefficient.
        """
        return self.Dn
    
    def rn(self):
        """
        Get the division rate.

        Returns:
            float: Division rate in hours.
        """
        return self.rn
    
    def cells(self):
        """
        Get the vector of cancer cells.

        Returns:
            numpy.ndarray: Vector of cancer cells.
        """
        return self.cells
    
    def n(self):
        """
        Get the vector of tumor densities.

        Returns:
            numpy.ndarray: Vector of densities.
        """
        return self.n
    
    def T(self):
        """
        Get the vector of T-cells densities.

        Returns:
            numpy.ndarray: Vector of densities.
        """
        return self.T

    def w(self):
        """
        Get the vector of total densities.

        Returns:
            numpy.ndarray: Vector of densities.
        """
        self.w=self.T+self.n
        return self.w
    
    def cells_size(self):
        """
        Get the list of the number of cells per iteration.

        Returns:
            numpy.ndarray: List of the number of cells per iteration.
        """
        return self.cells_size
    
    def proliferation(self):
        """
        Perform cell proliferation.

        Returns:
            tuple: Tuple containing cell movement, cell proliferation, and choice vectors.
        """
        np.random.seed(int(time.time()))

        # Probability of proliferation
        Nx = self.Nx
        cells0 = self.cells
        w0 = self.w
        w_max = self.w_max
        m = len(cells0)
        l = Nx**2
        f = np.ones(l) - (1./w_max)*w0
        Pp = np.zeros(l)
        print()
        # Using f(n) to create probabilities for each cell
        Pp[f>=0] = self.delta_t*self.rn*f[f>=0] 
        # Creating probabilities for each cell
        prolif = Pp[cells0] 
        #print(prolif)
        # Random
        V = np.random.uniform(0,1,m)
        choice = V > prolif

        # Extracting cells in proliferation and those in movement
        cellspro = cells0[V <= prolif] 
        cellsmouv = cells0[V > prolif]
        cellspro = cellspro.astype(int)

        return cellsmouv, cellspro, choice

    def movement(self, cellsmouv, cellspro, m0, choice):
        """
        Perform cell movement.

        Args:
            cellsmouv (numpy.ndarray): Cells to move.
            cellspro (numpy.ndarray): Cells to proliferate.
            m0 (int): Number of cells to move.
            choice (numpy.ndarray): Movement choice vector.
        """
        np.random.seed(int(time.time()))
        cells0 = self.cells
        Nx = self.Nx
        if (m0 > 0):
            Dn = self.Dn
            delta_t = self.delta_t
            l = Nx**2
            
            # Creating a probability vector
            P = np.zeros((5,l))
            R = np.zeros((5,l))
            for k in range(1,5):
                P[k] = delta_t*Dn/(self.delta_x**2) * np.ones(l)
            P[0] = np.ones(l) - P[1] - P[2] - P[3] - P[4]
    
            # Neumann conditions
            Neumann = np.arange(Nx)
            # Top
            P[0,Neumann] += P[3,Neumann]
            P[3,Neumann] = np.zeros(len(Neumann))
            # Bottom
            P[0,Neumann+Nx*(Nx-1)] += P[4,Neumann+Nx*(Nx-1)]
            P[4,Neumann+Nx*(Nx-1)] = np.zeros(len(Neumann))
            # Left
            P[0,Nx*Neumann] += P[2,Nx*Neumann]
            P[2,Nx*Neumann] = np.zeros(len(Neumann))
            # Right
            P[0,Nx*Neumann+Nx-1] += P[1,Nx*Neumann+Nx-1]
            P[1,Nx*Neumann+Nx-1] = np.zeros(len(Neumann))
            
            # Creating intervals
            R[0] = P[0]
            for k in range(1,5):
                R[k] = R[k-1] + P[k]
        
            C = R[:,cellsmouv]

            # Random part:
            U = np.random.uniform(0,1,m0)
            direction = np.zeros(m0)
            
            # Handling directions
            for k in range(1,len(R)):
                direction[(U>C[k-1])&(U<= C[k])] = k
            direction = direction.astype(int)
            
            # Associating each direction with its coefficient
            coeff = np.zeros(m0)
            coeff[(direction == 1)] = 1 # right
            coeff[(direction == 2)] = -1 # left
            coeff[(direction == 3)] = -Nx # top
            coeff[(direction == 4)] = Nx # bottom

            # Update the cells vector at time t+1 
            cellsmouv = (cellsmouv + coeff).astype(int)

        # Assembly
        cells0[choice] = cellsmouv
        # Add daughter cells to the end
        self.cells = np.append(cells0,cellspro)
        # Update the base density vector based on the movements made     
        self.n = np.bincount(self.cells, minlength=int(Nx**2))
        self.w = self.n + self.T

    def update_density_tcells(self, new_density):
        """
        Update tumors density in tcells_mvt.

        Args:
            new_density_tcells (numpy.ndarray): List of the new density for Tcells.
        """
        self.T = np.array(new_density) 
