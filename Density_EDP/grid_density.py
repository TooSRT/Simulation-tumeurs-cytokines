"""
Grid Module

This module provides a Grid class for printing tumor growth on a grid.

Authors: LANGUE William, DUFRESNE Théo
Date: 17/03/2024

Usage:
    from Density_EDP.grid_Density import Grid

    # Example usage:
    grid = Grid(n_size=100, unit="mm")
    grid.print(n=n_array, n_max=100, Nx=10, t=0)
"""

from matplotlib.colors import LinearSegmentedColormap
import numpy as np
from matplotlib import pyplot as plt

class Density_Grid:
    """
    Grid class

    This class represents a grid for tumor simulation.

    Attributes:
        color_scale (LinearSegmentedColormap): Color scale for the grid.
        square_size (numpy.ndarray): Size of each grid square.
        unit (str): Unit of measurement for the grid.
    """

    def __init__(self, n_size, unit):
        """
        Initialize a Grid object.

        Args:
            n_size (int): Size of the grid.
            unit (str): Measurement unit of the grid area ("mm" or "cm").
        """
        assert unit in ['mm', 'cm']
        c = [(0, 0, 0), (0.5, 0.5, 1), (0, 0, 1), (1, 0.5, 0.5), (1, 0, 0)]
        p = [0, 0.25, 0.5, 0.75, 1]
        self.color_scale = LinearSegmentedColormap.from_list('custom', list(zip(p, c)))
        square_size = np.full(n_size, 800)
        if unit == 'cm':
            square_size = np.full(n_size, 3.65)
        self.square_size = square_size
        self.unit = unit

    def color_scale(self):
        """
        Get the color scale used in the grid.

        Returns:
            LinearSegmentedColormap: The color scale.
        """
        return self.color_scale
    
    def square_size(self):
        """
        Get the size of each grid square.

        Returns:
            numpy.ndarray: Size of each grid square.
        """
        return self.square_size
    
    def unit(self):
        """
        Get the unit of measurement for the grid.

        Returns:
            str: The unit of measurement.
        """
        return self.unit

    def print(self, n, n_max, Nx, t, size):
        """
        Print the grid at the given time step.

        Args:
            n (numpy.ndarray): Array representing the grid.
            n_max (int): Maximum value in the grid.
            Nx (int): Size of the grid.
            t (int): Time step.
        """
        x = np.empty(0)
        y = np.empty(0)
        for j in range(Nx):
            x = np.append(x, np.zeros(Nx) + j)
            y = np.append(y, np.arange(0, Nx))
        colors = n
        plt.title("Tumor at t = " + str(t) + "h")
        plt.xticks([x[0],x[-1]], ['0','1'])
        plt.yticks([y[0],y[-1]], ['0','1'])
        plt.xlabel('1' + self.unit)
        plt.ylabel('1' + self.unit)
        plt.scatter(x, y, c=colors, s=self.square_size, cmap=self.color_scale, vmin=0, vmax=n_max, marker='s')
        plt.colorbar()
        plt.figtext(0.6, 0.02, 'number of tumors = ' + str(size), fontsize=10, color='black')
        plt.savefig('./plot/density_pictures/cells_t =' + str(t) + 'h.png')
        plt.clf()

    def growth(self, cells_size):
        """
        Plot the tumor growth.

        Args:
            cells_size (numpy.ndarray): Array representing the tumor size over time.
        """
        x = np.arange(len(cells_size)) * 15
        plt.title("Tumor Growth")
        plt.xlabel("Time in hours")
        plt.ylabel("Number of cells")
        plt.plot(x, cells_size, 'ro--')
        plt.savefig('./plot/Tumor_growth.png')
        plt.clf()

