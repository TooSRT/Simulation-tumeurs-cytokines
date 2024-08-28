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

    def __init__(self, Nx, c0, pos0, tol, delta_x, delta_t, D_cytokine, Tau_p_CD4, Tau_c_CD4, Tau_c_CD8, alpha_c, Pheno_CD4, Pheno_CD8, Active_CD4, Active_CD8, Inactive_CD4, Inactive_CD8, n0):
        """
        Initialize an O2_EDP object.

        Args:
            Nx (int): Size of the grid.
            c0 (numpy.ndarray): Initial cytokine concentration vector.
            delta_t (int): Time step size (in hours).
            delta_x (float): Spatial step size.
            D_cytokine (float): Diffusion coefficient for cytokine.
            cyto (numpy.ndarray): Vector of cytokine concentration.
            Tau_p_CD4 (float): Cytokine production by CD4.
            Tau_c_CD4 (float): Cytokine consumption by CD4.
            Tau_c_CD8 (float): Cytokine consumption by CD8.
            P_prod (float): Probability of Tcells to be a cytokine producer
            P_cons (float): Probability of Tcells to be a cytokine consumer
            alpha_c (float): decay rate for cytokine
            Pheno_CD4 (list): List of CD4 Tcells
            Pheno_CD8 (list): List of CD8 Tcells
            Active_CD4 (list): List of active CD4 Tcells
            Active_CD8 (list): List of active CD8 Tcells
            Inactive_CD4 (list): List of inactive CD4 Tcells
            Inactive_CD8 (list): List of inactive CD8 Tcells
            n0 (numpy.ndarray): Initial density vector (tumor).
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
        self.Tau_p_CD4 = Tau_p_CD4
        self.Tau_c_CD4 = Tau_c_CD4
        self.Tau_c_CD8 = Tau_c_CD8
        self.alpha_c = alpha_c
        self.Pheno_CD4 = Pheno_CD4
        self.Pheno_CD8 = Pheno_CD8
        self.n = n0
        self.Active_CD4 = Active_CD4
        self.Active_CD8 = Active_CD8
        self.Inactive_CD4 = Inactive_CD4
        self.Inactive_CD8 = Inactive_CD8

        self.A = self.init_A()
        self.B, self.supply, A_new = self.init_b(pos0)

        # Attributs utiles pour la grille 
        Lx = 0.1
        x,dx = np.linspace(delta_x,Lx-delta_x,Nx,retstep=True) #grid in x and step in x
        y,dx = np.linspace(delta_x,Lx-delta_x,Nx,retstep=True) #grid in y and step in x
        # in data grid form
        self.X, self.Y = np.meshgrid(x, y)

    def init_b(self, pos): 
        """
        Initialize the matrix B and the vector supply.

        Args:
            pos (numpy.ndarray): Position's vector of blood vessels.
        
        Returns:
            tuple: A tuple containing the initialized matrix B and vector supply.
        """
        Nx = self.Nx
        delta_t = self.delta_t
        #Update Rp_vect and rc_vect
        self.Rp_vect = self.Active_CD4 * self.Tau_p_CD4 #Only CD4 produce
        self.Rc_vect = self.Pheno_CD4 * self.Tau_c_CD4 + self.Pheno_CD8 * self.Tau_c_CD8 #Both CD4 and CD8 consume and it does not depend from their activities
        supply = np.zeros(Nx**2)
        identify_consum_immune_cells = np.zeros((Nx**2,))

        assert (len(pos) < Nx**2 + 1)
        for i, p in enumerate(pos):
            #print(f"Position: {p}, Production: {Rp_vect[i]:.4f}, Consommation: {Rc_vect[i]*self.cyto[self.pos][i]:.4f}")
            supply[p] = delta_t*(self.Rp_vect[i])
            if self.Rc_vect[i]*self.cyto[self.pos][i] > 0: #cyto[self.pos][i] concentration en cytokine à la position i
                identify_consum_immune_cells[p] = delta_t*self.Rc_vect[i]

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
        
        A = (1 + delta_t*alpha_c)*diags([np.ones(Nx**2)],[0], shape=(Nx**2,Nx**2), format='csc') + (D_cytokine*delta_t/delta_x**2)*diags(diagonals,offsets, shape=(Nx**2,Nx**2) , format='csc') 

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
        self.update_pheno()
    
    def update_positions(self, new_positions):
        """
        Update Tcells positions in Cytokine_EDP.

        Args:
            new_positions (numpy.ndarray): List of the new positions for Tcells.
        """
        self.pos = np.array(new_positions)

    def update_pheno(self):
        """
        Updates the active phenotypes of T-cells based on their properties.
        """
        Active_CD4 = self.Active_CD4 
        Active_CD8 = self.Active_CD8 
        Inactive_CD4 = self.Inactive_CD4
        Inactive_CD8 = self.Inactive_CD8
        #List of activity of each tcells
        """
        Inactive_CD4 = [1,1,1,1]
        Active_CD4   = [0,0,0,0,]

        Example:
        After one step, one CD4 become active:

        Inactive_CD4 = [1,0,1,1]
        Active_CD4   = [0,1,0,0,]

        Use different coefficient of consumption to keep the activity of and active Tcell and an inactive one
        """
        self.Tcells_memorize = np.zeros(len(self.pos), dtype=bool) #Memorize Tcells that are influenced by cytokine or interacted with tumor

        n_pos = [i for i, qte in enumerate(self.n) if qte != 0] #get positions of tumors cells
        for idx, i in enumerate(self.pos):
            diff = np.array([i] * len(n_pos)) #create list of size n_pos with the position of one T cell

#-----------T-cells that have interacted with tumor cells-----------
            #(n_pos - diff) space between each Tcells and tumor cells
            if np.any(np.abs(n_pos - diff)) <= 1: #Check if they are on the side of the tumor cell
                #print('test réussi side')
                self.Tcells_memorize[idx]=True #save T-cells that have interacted with tumors cells
                Inactive_CD4[idx] = 0 #Tcell is now active
                Inactive_CD8[idx] = 0
                Active_CD4[idx] = 1 #Tcell is now active
                Active_CD8[idx] = 1
            elif self.Nx - 1 <= np.any(n_pos - diff) <= self.Nx +1: #Check if they are above the tumor cell
                #print('test réussi above')
                self.Tcells_memorize[idx]=True 
                Inactive_CD4[idx] = 0 
                Inactive_CD8[idx] = 0
                Active_CD4[idx] = 1 
                Active_CD8[idx] = 1
            elif -1 - self.Nx <= np.any(n_pos - diff) <= 1 - self.Nx: #Check if they are under the tumor cell
                #print('test réussi under')
                self.Tcells_memorize[idx]=True
                Inactive_CD4[idx] = 0 
                Inactive_CD8[idx] = 0
                Active_CD4[idx] = 1 
                Active_CD8[idx] = 1

#-----------T-cells under cytokine influence-----------
            #Définir le taux seuil à partir du quel une cellule T a assez consommé de cytokines
            #à cause de la diffusion nous avons toujours une infime concentration en cytokine sur la grille et donc >0 (peu importe l'endroit)
            if self.Rc_vect[idx]*self.cyto[self.pos][idx] > 2: #Check concentration consumption of consumer
                self.Tcells_memorize[idx]=True #save T-cells that are under cytokine influence
                Inactive_CD4[idx] = 0 
                Inactive_CD8[idx] = 0 
                Active_CD4[idx] = 1 
                Active_CD8[idx] = 1

#-----------T-cells loose cytokine influence or don't have enough cytokines to become active-----------
            #Définir le taux seuil à partir du quel une cellule T doit se maintenir pour rester sous l'influence des cytokines
            elif self.Rc_vect[idx]*self.cyto[self.pos][idx] <= 1: #Check concentration consumption of consumer
                self.Tcells_memorize[idx]=False #T-cells is not under cytokine influence anymore
                Inactive_CD4[idx] = 1
                Inactive_CD8[idx] = 1
                Active_CD4[idx] = 0
                Active_CD8[idx] = 0

    def update_density_tumors(self, new_density):
        """
        Update tumors density in tcells_mvt and cytokine_edp.

        Args:
            new_density (numpy.ndarray): List of the new density for tumors.
        """
        self.n = np.array(new_density)
        
        
