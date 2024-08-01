"""
O2 EDP Module

This module provides a class O2_EDP for simulation of O2 concentration.

Authors: LANGUE William, DUFRESNE Théo
Date: 22/03/2024

Usage:
    from Simulation.o2_edp import O2_EDP

    # Example usage:
    o2_edp = O2_EDP(Nx=100, c0=np.zeros(100**2), pos=np.array([500, 501]), tol=1e-8, delta_x=0.0001, delta_t=15, Dc=0.1, c=1.0, kappa=0.05)
    o2_edp.O2_diffusion()
"""

import numpy as np
from scipy.sparse import diags
from scipy.sparse.linalg import cg

class O2_EDP:
    """
    O2_EDP class

    This class represents a model for O2 volatility using the HDC-EDP method.

    Attributes:
        kappa (float): Coefficient of oxygen permeability through a blood vessel.
        Nx (int): Size of the grid.
        delta_t (int): Time step size (in hours).
        delta_x (float): Spatial step size.
        cblood (float): Oxygen concentration in a blood vessel (in mmol/mL).
        Dc (float): Diffusion coefficient.
        c (numpy.ndarray): Vector of oxygen concentration.
        pos (numpy.ndarray): Position's vector of blood vessels.
        tol (float): Tolerance of the conjugate gradient algorithm.
    """

    def __init__(self, Nx, c0, pos0, tol, delta_x, delta_t, Dc, c, kappa):
        """
        Initialize an O2_EDP object.

        Args:
            Nx (int): Size of the grid.
            c0 (numpy.ndarray): Initial oxygen concentration vector.
            pos (numpy.ndarray): Position's vector of blood vessels.
            tol (float): Tolerance of the conjugate gradient algorithm.
            delta_x (float): Spatial step size.
            delta_t (int): Time step size (in hours).
            Dc (float): Diffusion coefficient.
            c (float): Oxygen concentration in a blood vessel (in mmol/mL).
            kappa (float): Coefficient of oxygen permeability through a blood vessel.
        """
        self.Nx = Nx
        self.c = c0
        self.pos = pos0
        self.tol = tol 
        self.delta_x = delta_x
        self.delta_t = delta_t
        self.Dc = Dc
        self.cblood = c
        self.kappa = kappa
        self.B, self.supply = self.init_mat_vect_b(pos0)
        self.A = self.init_A()
        # Attributs utiles pour la grille 
        Lx = 0.1
        x,dx = np.linspace(delta_x,Lx-delta_x,Nx,retstep=True) #grid in x and step in x
        y,dx = np.linspace(delta_x,Lx-delta_x,Nx,retstep=True) #grid in y and step in x
        # in data grid form
        self.X, self.Y = np.meshgrid(x, y)

    def init_mat_vect_b(self, pos): 
        """
        Initialize the matrix B and the vector supply.

        Args:
            pos (numpy.ndarray): Position's vector of blood vessels.
        
        Returns:
            tuple: A tuple containing the initialized matrix B and vector supply.
        """
        Nx = self.Nx
        kappa = self.kappa
        cblood = self.cblood
        delta_t = self.delta_t
        supply = np.zeros(Nx**2)
        assert (len(pos) < Nx**2 + 1)
        for p in pos:
            supply[p] = 1
        B = diags([np.ones(Nx**2)], [0], shape=(Nx**2, Nx**2), format='csc') - delta_t * kappa * diags([supply], [0], shape=(Nx**2, Nx**2), format='csc')
        supply = (delta_t * kappa * cblood) * supply
        return B, supply
    
    def init_A(self):
        Nx = self.Nx
        Dc = self.Dc
        delta_t = self.delta_t
        delta_x = self.delta_x
        
        center_diag = 4*np.ones(Nx**2)
        left_diag = -np.ones(Nx**2 - 1)
        right_diag = -np.ones(Nx**2)
        
        # Neumann BC
        for ii in range(0,Nx):
            center_diag[ii*Nx] = center_diag[ii*Nx]- 1. # left most cell
            center_diag[(ii+1)*Nx-1] = center_diag[(ii+1)*Nx-1]- 1. # rightmost cell
            if ii != 0:
                left_diag[ii*Nx-1] = 0

            right_diag[ii*Nx] = 0
            
        # bottom line
        center_diag[:Nx] = center_diag[:Nx] - 1.
        # top line
        center_diag[-Nx:] = center_diag[-Nx:] - 1 
        
        right_diag = np.roll(right_diag,-1)
        
        diagonals = [-np.ones(Nx**2-Nx), left_diag, center_diag,right_diag,-np.ones(Nx**2 - Nx)]
        offsets = [-Nx,-1,0,1,Nx] 
        A = diags([np.ones(Nx**2)],[0], shape=(Nx**2,Nx**2), format='csc') + (Dc*delta_t/delta_x**2)*diags(diagonals,offsets, shape=(Nx**2,Nx**2), format='csc')

        return A
    
    def iter_b(self):
        """
        Perform iteration for the vector b.

        Returns:
            numpy.ndarray: Iterated vector b.
        """
        B = self.B
        supply = self.supply
        c = self.c
        b = B * c + supply
        return b
    
    def O2_diffusion(self):
        """
        Perform O2 diffusion calculation.
        """
        A = self.A
        b = self.iter_b()
        # Solve linear system using Conjugate Gradient method
        self.c = cg(A, b, self.c, self.tol)[0]

