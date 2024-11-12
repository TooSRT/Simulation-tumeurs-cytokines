"""
Grid Module

This module provides a Grid class for printing O2 concentration on a grid.

Authors: LANGUE William, DUFRESNE Théo
Date: 30/03/2024

Usage:
    from O2_EDP.grid_O2 import Grid

    # Example usage:
    grid = Grid(c_size=100, unit="mm")
    grid.print(c=c_array, cblood=8.9, Nx=10, t=0)
"""

from matplotlib.colors import LinearSegmentedColormap
import numpy as np
from matplotlib import pyplot as plt

class O2_Grid :
    """
    Grid class

    This class represents a grid for o2 concentration simulation.

    Attributes:
        color_scale (LinearSegmentedColormap): Color scale for the grid.
        square_size (numpy.ndarray): Size of each grid square.
        unit (str): Unit of measurement for the grid.
    """

    def __init__(self, unit):
        """
        Initialize a Grid object.

        Args:
            c_size (int): Size of the grid.
            unit (str): Measurement unit of the grid area ("mm" or "cm").
        """
        assert unit in ['mm', 'cm']
        self.unit = unit
        
    
    def unit(self):
        """
        Get the unit of measurement for the grid.

        Returns:
            str: The unit of measurement.
        """
        return self.unit
    
    def print(self, c, X, Y, Nx, t):
        """
        Print the grid at the given time step.

        Args:
            c (numpy.ndarray): Array representing the grid.
            cblood (float): O2 concentration in blood vessel.
            Nx (int): Size of the grid.
            t (int): Time step.
        """
        c_mat = np.zeros((Nx,Nx))
        for ii in range(0,Nx):
            for jj in range(0,Nx):
                c_mat[ii, jj] = c[Nx*ii+jj]
        surf = plt.contourf(X,Y,c_mat)
        plt.title("O2 concentration at t = " + str(t) + "h")
        plt.xlabel('1' + self.unit)
        plt.ylabel('1' + self.unit)
        plt.colorbar(surf, shrink=0.5, aspect=10)
        plt.savefig('./plot/o2_pictures/O2_concentration at t =' + str(t) + 'h.png')
        plt.clf()
        