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
from O2_EDP.tcells_mvt import Tcells_mvt
from scipy.sparse import diags
from scipy.sparse.linalg import cg
import sys
np.set_printoptions(threshold=sys.maxsize)

class cytokine_EDP:
    """
    cytokine_EDP class

    This class represents a model for cytokine evolution.

    Attributes:
        Nx (int): Size of the grid.
        delta_t (int): Time step size (in hours).
        delta_x (float): Spatial step size.
        D_cytokine (float): Diffusion coefficient.
        cyto (numpy.ndarray): Vector of cytokine concentration.
        pos (numpy.ndarray): Position's vector of blood vessels.
        tol (float): Tolerance of the conjugate gradient algorithm.
    """

    def __init__(self, Nx, c0, pos0, tol, delta_x, delta_t, D_cytokine, Tau_p, Tau_c, P_prod, P_cons, alpha_c, tcells_mvt):
        """
        Initialize an O2_EDP object.

        Args:
            Nx (int): Size of the grid.
            c0 (numpy.ndarray): Initial cytokine concentration vector.
            delta_t (int): Time step size (in hours).
            delta_x (float): Spatial step size.
            D_cytokine (float): Diffusion coefficient for cytokine.
            cyto (numpy.ndarray): Vector of cytokine concentration.
            Tau_p (float): Cytokine production.
            Tau_c (float): Cytokine consumption.
            P_prod (float): Probability of Tcells to be a cytokine producer
            P_cons (float): Probability of Tcells to be a cytokine consumer
            alpha_c (float): decay rate for cytokine
            pos (numpy.ndarray): Position's vector of blood vessels.
            tol (float): Tolerance of the conjugate gradient algorithm.
        """
        self.Nx = Nx
        self.cyto = c0 #à revoir 
        self.pos = pos0
        self.tol = tol 
        self.delta_x = delta_x
        self.delta_t = delta_t
        self.D_cytokine = D_cytokine
        self.Tau_p = Tau_p
        self.Tau_c = Tau_c
        self.alpha_c = alpha_c

        #Initialisation of cells and their phenotype
        Vect_unif = np.random.uniform(low=0.0, high=1.0, size=np.size(pos0)) #Vecteur suivant une loi uniforme sur [0,1]
        
        #Density of producer and consumer 
        Pheno_actif_prod = np.zeros(len(pos0))  #Liste phenotype actif produisant n_prod
        Pheno_actif_cons = np.zeros(len(pos0))  #Liste phenotype actif consommant n_cons
        
        #(revoir les probas utilisés)
        for j in range(len(pos0)):
            if Vect_unif[j] <= P_prod:  #Déterminer aléatoirement les producteurs
                Pheno_actif_prod[j] = 1
            if Vect_unif[j] >= 1 - P_cons:  #Déterminer aléatoirement les consommateurs
                Pheno_actif_cons[j] = 1

        #Si la cytokine est productrice ou consomatrice (donc Pheno_actif_prod[i]=1) on la multiplie par un facteur de production ou consommation
        self.Rp_vect = Pheno_actif_prod * Tau_p 
        self.Rc_vect = Pheno_actif_cons * Tau_c

        self.A = self.init_A()
        self.B, self.supply, A_new = self.init_b(pos0)
        self.tcells_mvt = tcells_mvt
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
        Rc_vect = self.Rc_vect #contient les vecteurs consommateurs correspondant
        supply = np.zeros(Nx**2)
        identify_consum_immune_cells = np.zeros((Nx**2,))
        assert (len(pos) < Nx**2 + 1)
        for i, p in enumerate(pos):
            #print(f"Position: {p}, Production: {Rp_vect[i]:.4f}, Consommation: {Rc_vect[i]*self.cyto[self.pos][i]:.4f}")
            supply[p] = delta_t*(Rp_vect[i])
            if Rc_vect[i]*self.cyto[self.pos][i] > 0: #cyto[self.pos][i] concentration en cytokine à la position i
                identify_consum_immune_cells[p] = delta_t*Rc_vect[i]

        #Màj de la matrice A en fonction des cellules consommatrices
        A_new = self.A + diags([identify_consum_immune_cells],[0], shape=(Nx**2,Nx**2),format='csc')
        B = diags([np.ones(Nx**2)], [0], shape=(Nx**2, Nx**2), format='csc')
        return B, supply, A_new 
    
    #Matrice A presque similaire à l'oxygène
    #Ajout d'un terme (delta_t*alpha_c)*diags([np.ones(Nx**2)],[0], shape=(Nx**2,Nx**2), format='csc')
    def init_A(self):
        Nx = self.Nx
        D_cytokine = self.D_cytokine
        delta_t = self.delta_t
        delta_x = self.delta_x
        alpha_c = self.alpha_c
        
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
        
        A = diags([np.ones(Nx**2)],[0], shape=(Nx**2,Nx**2), format='csc') + (delta_t*alpha_c)*diags([np.ones(Nx**2)],[0], shape=(Nx**2,Nx**2), format='csc') + (D_cytokine*delta_t/delta_x**2)*diags(diagonals,offsets, shape=(Nx**2,Nx**2) , format='csc') 

        return A
    
    def iter_b(self):
        """
        Perform iteration for the vector b.

        Returns:
            numpy.ndarray: Iterated vector b.
        """
        B, supply, A_new = self.init_b(self.pos) #mettre à jour le vecteur b à chaque itération 
        cyto = self.cyto
        b = cyto + supply
        return b, A_new
    
    def cytokine_diffusion(self):
        b, A_new = self.iter_b()
        self.cyto, info = cg(A_new, b, self.cyto, tol=self.tol)
        if info != 0:
            print("Conjugate gradient did not converge")
        #print("Min cyto = " +str(min(self.cyto)))
        
        self.tcells_mvt.movement(self.Rc_vect) #màj des positions à chaque itération
        self.pos = self.tcells_mvt.pos

