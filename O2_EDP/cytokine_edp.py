"""
Cytokines EDP Module

This module provides a class cytokines_EDP for simulation of O2 concentration.

Authors: LANGUE William, DUFRESNE Théo
Date: 22/03/2024

Usage:
    from Simulation.o2_edp import O2_EDP

    # Example usage:
    o2_edp = O2_EDP(Nx=100, c0=np.zeros(100**2), pos=np.array([500, 501]), tol=1e-8, delta_x=0.0001, delta_t=15, D_cytokine=0.1, c=1.0, kappa=0.05)
    o2_edp.O2_diffusion()
"""

import numpy as np
import random
from scipy.sparse import diags
from scipy.sparse.linalg import cg

class cytokine_EDP:
    """
    cytokine_EDP class

    This class represents a model for cytokine evolution using the HDC-EDP method.

    Attributes:
        Nx (int): Size of the grid.
        delta_t (int): Time step size (in hours).
        delta_x (float): Spatial step size.
        D_cytokine (float): Diffusion coefficient.
        cyto (numpy.ndarray): Vector of cytokine concentration.
        pos (numpy.ndarray): Position's vector of blood vessels.
        tol (float): Tolerance of the conjugate gradient algorithm.
    """

    def __init__(self, Nx, c0, pos0, tol, delta_x, delta_t, D_cytokine, Rp_vect, Rc_vect):
        """
        Initialize an O2_EDP object.

        Args:
            Nx (int): Size of the grid.
            c0 (numpy.ndarray): Initial cytokine concentration vector.
            delta_t (int): Time step size (in hours).
            delta_x (float): Spatial step size.
            D_cytokine (float): Diffusion coefficient for cytokine.
            cyto (numpy.ndarray): Vector of cytokine concentration.
            R_p (float): Cytokine production.
            R_c (float): Cytokine consumption.
            pos (numpy.ndarray): Position's vector of blood vessels.
            tol (float): Tolerance of the conjugate gradient algorithm.
        """
        self.Nx = Nx
        self.cyto = c0 #à revoir ajouter une valeur initiale de concentration en cytokine
        self.pos = pos0
        self.tol = tol 
        self.delta_x = delta_x
        self.delta_t = delta_t
        self.D_cytokine = D_cytokine
        self.Rp_vect= Rp_vect
        self.Rc_vect= Rc_vect
        self.B, self.supply = self.init_b(pos0)
        #self.b= self.init_b()
        self.A = self.init_A()
        # Attributs utiles pour la grille 
        Lx = 0.1
        x,dx = np.linspace(delta_x,Lx-delta_x,Nx,retstep=True) #grid in x and step in x
        y,dx = np.linspace(delta_x,Lx-delta_x,Nx,retstep=True) #grid in y and step in x
        # in data grid form
        self.X, self.Y = np.meshgrid(x, y)
    
    def init_b(self,pos): 
        """
        Initialize the matrix B and the vector supply.

        Args:
            pos (numpy.ndarray): Position's vector of blood vessels.
        
        Returns:
            tuple: A tuple containing the initialized matrix B and vector supply.
        """
        Nx = self.Nx
        delta_t = self.delta_t
        Rp_vect = self.Rp_vect #contient les vecteurs producteurs correspondant
        Rc_vect = self.Rc_vect
        supply = np.zeros(Nx**2)
        assert (len(pos) < Nx**2 + 1)
        for i, p in enumerate(pos):
            print(f"Position: {p}, Production: {Rp_vect[i]:.4f}, Consommation: {Rc_vect[i]*self.cyto[self.pos][i]:.4f}")
            supply[p] = delta_t*(Rp_vect[i] - Rc_vect[i]*self.cyto[self.pos][i]) #self.cyto[self.pos][i]=concentration en cyto à une pos donnée
        B = delta_t*diags([np.ones(Nx**2)], [0], shape=(Nx**2, Nx**2), format='csc') 
        return B, supply 
        
    
    #Matrice A similaire à l'oxygène avec les même conditions aux bords
    def init_A(self):
        Nx = self.Nx
        D_cytokine = self.D_cytokine
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
        A = diags([np.ones(Nx**2)],[0], shape=(Nx**2,Nx**2), format='csc') + (D_cytokine*delta_t/delta_x**2)*diags(diagonals,offsets, shape=(Nx**2,Nx**2), format='csc')

        return A
    
    
    def iter_b(self):
        """
        Perform iteration for the vector b.

        Returns:
            numpy.ndarray: Iterated vector b.
        """
        B = self.B
        #b = self.init_b(self.pos) #mettre à jour le vecteur b à chaque itération (inutile pour le moment)
        supply = self.supply
        cyto = self.cyto
        b = B @ cyto + supply
        return b
    
    def cytokine_diffusion(self):
        A = self.A
        b = self.iter_b()
        self.cyto, info = cg(A, b, self.cyto, tol=self.tol)
        if info != 0:
            print("Conjugate gradient did not converge")
        self.cyto = np.maximum(self.cyto, 0)  #empêche les valeurs négatives pour la concentration
        

