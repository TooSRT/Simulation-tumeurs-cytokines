"""
Simulate the movement of tcells
"""

import time
import numpy as np
from matplotlib import pyplot as plt

class Tcells_mvt:
    """
    Density_EDP class

    This class represents a model for random T-cells movements.

    Attributes:
        nb_cells (int): Number of cells in the grid.
        Nx (int): Size of the grid.
        delta_t (int): Time step size (in hours).
        delta_x (float): Spatial step size.
        w_max (int): Maximum density value.
        D_tcells (float): Diffusion coefficient.
        cells (numpy.ndarray): Vector of cancer cells.
        w (numpy.ndarray): Vector of total densities.
        n (numpy.ndarray): Vector of tumors densities.
        T (numpy.ndarray): Vector of T_cells densities.
    """
    def __init__(self, Nx, pos0, w0, T0, n0, w_max, delta_x, delta_t, D_tcells, Pheno_CD4, Pheno_CD8, T_CD4, T_CD8, cytokine_edp_instance):
        """
        Initialize a Density_EDP object.

        Args:
            nb_cells (int): Number of cells in the grid.
            Nx (int): Size of the grid.
            cells0 (numpy.ndarray): Initial cancer cell vector.
            w0 (numpy.ndarray): Initial density vector.
            n0 (numpy.ndarray): Initial density vector (tumor).
            T0 (numpy.ndarray): Initial density vector (T-cells).
            w_max (int): Maximum density value.
            delta_x (float): Spatial step size.
            delta_t (int): Time step size (in hours).
            Pheno_actif_prod (numpy.ndarray): List of cytokines producer (CD8)
            Pheno_actif_cons (numpy.ndarray): List of cytokines consummer (CD4)
            T_CD4 (numpy.ndarray): List of initial density for CD4.
            T_CD8 (numpy.ndarray): List of initial density for CD8.
        """
        self.pos = pos0
        self.Nx = Nx
        self.w = w0
        self.n = n0
        self.T = T0
        self.w_max = w_max
        self.delta_x = delta_x
        self.delta_t = delta_t
        self.D_tcells = D_tcells
        self.Pheno_CD4 = Pheno_CD4
        self.Pheno_CD8 = Pheno_CD8
        self.T_CD4 = T_CD4
        self.T_CD8 = T_CD8
        self.cytokine_edp = cytokine_edp_instance

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
    
    def D_tcells(self):
        """
        Get the diffusion coefficient.

        Returns:
            float: Diffusion coefficient.
        """
        return self.D_tcells
    
    def pos(self):
        """
        Get the vector of cancer cells.

        Returns:
            numpy.ndarray: Vector of cancer cells.
        """
        return self.pos
    
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
    
    def psi(self,w):
        """
        Fonction pour moduler les probabilités de mouvement en fonction de la densité cellulaire.

        Args:
            w (float): total density at each position
        Returns:
            float: Value of psi.
        """
        w_max = self.w_max

        if 0<=w<=w_max:
            return (1-w/w_max)
        else:
            return 0
    
    def Pheno_CD4(self):
        """
        Get the list of Tcells CD4

        Returns:
            numpy.ndarray: Position vector of CD4.
        """
        return self.Pheno_actif_prod
    
    def Pheno_CD8(self):
        """
        Get the list of Tcells CD8

        Returns:
            numpy.ndarray: Position vector of CD8.
        """
        return self.Pheno_actif_cons
    
    def T_CD4(self):
        """
        Get the density of CD4 in the grid

        Returns:
            numpy.darray: Vector of CD4
        """
        return self.T_CD4
    
    def T_CD8(self):
        """
        Get the density of CD8 in the grid

        Returns:
            numpy.darray: Vector of CD8
        """
        return self.T_CD8

    def movement(self):
        """
        Perform cell movement for Tcells.
        """
        l=len(self.pos)
        movement_vector = np.zeros(len(self.pos),dtype=int) #initialize our movement vector to update position
        prob_moove = np.zeros((l,5)) #initialize our probability vector for movement of each cell at every time step

        for idx, i in enumerate(self.pos):
#-----------T-cells under cytokine influence or have interacted with tumors-----------
            if self.cytokine_edp.Tcells_memorize[idx]: #If our T cells has already been influenced by cytokines or interacted with tumors
                #Moove to left
                if i % self.Nx != 0: #ne doit pas se trouver sur la colonne gauche
                    T_left = 0.25
                else:
                    T_left = 0
                #Moove to right
                if i % self.Nx != self.Nx - 1 : #ne doit pas se trouver sur la colonne droite
                    T_right = 0.25
                else:
                    T_right = 0

                #Moove below
                if 0 < i < self.Nx: #Ne doit pas se trouver sur la bordure du bas
                    T_below = 0.25
                else:
                    T_below = 0

                #Moove upper
                if  self.Nx*self.Nx - self.Nx < i < self.Nx*self.Nx: #Ne doit pas se trouver sur la bordure du haut
                    T_upper = 0.25
                else:
                    T_upper = 0

                #Stay
                T_stay = 1 - T_left - T_right - T_upper - T_below

                prob_moove[idx, 0] = T_left
                prob_moove[idx, 1] = T_right
                prob_moove[idx, 2] = T_upper
                prob_moove[idx, 3] = T_below
                prob_moove[idx, 4] = T_stay

                #Choose a direction based on probability 
                #print(prob_moove)
                moove = np.random.choice(['left', 'right', 'upper','below','stay'], p=[T_left, T_right, T_upper, T_below, T_stay])
            
                if moove == 'left':
                    movement_vector[idx] = -1 
                    #print(f"Cellule {idx} se déplace vers la gauche.")
                elif moove == 'right':
                    movement_vector[idx] = 1 
                    #print(f"Cellule {idx} se déplace vers la droite.")
                elif moove == 'upper':
                    movement_vector[idx] = self.Nx
                    #print(f"Cellule {idx} se déplace en haut.")
                elif moove == 'below':
                    movement_vector[idx] = - self.Nx
                    #print(f"Cellule {idx} se déplace en bas.")                    
                
                #If stay, movement is 0 (by default)

                #print(movement_vector)

#-----------T-cells inactive or loose cytokine influence-----------
            else:
                #Moove to left
                if i % self.Nx != 0: #ne doit pas se trouver sur la colonne gauche
                    T_left = 0
                else:
                    T_left = 0
                #Moove to right
                if i % self.Nx != self.Nx - 1 : #ne doit pas se trouver sur la colonne droite
                    T_right = 1
                else:
                    T_right = 0
                
                #Moove below
                if 0 < i < self.Nx: #Ne doit pas se trouver sur la bordure du bas
                    T_below = 0
                else:
                    T_below = 0

                #Moove upper
                if  self.Nx*self.Nx -  self.Nx < i < self.Nx*self.Nx: #Ne doit pas se trouver sur la bordure du haut
                    T_upper = 0
                else:
                    T_upper = 0
                
                #Stay
                T_stay = 1 - T_left - T_right - T_upper - T_below
                
                prob_moove[idx, 0] = T_left
                prob_moove[idx, 1] = T_right
                prob_moove[idx, 2] = T_upper
                prob_moove[idx, 3] = T_below
                prob_moove[idx, 4] = T_stay

                #Choose a direction based on probability 
                #print(prob_moove)
                moove = np.random.choice(['left', 'right', 'upper','below','stay'], p=[T_left, T_right, T_upper, T_below, T_stay])
            
                if moove == 'left':
                    movement_vector[idx] = -1 
                    #print(f"Cellule {idx} se déplace vers la gauche.")
                elif moove == 'right':
                    movement_vector[idx] = 1 
                    #print(f"Cellule {idx} se déplace vers la droite.")
                elif moove == 'upper':
                    movement_vector[idx] = self.Nx
                    #print(f"Cellule {idx} se déplace en haut.")
                elif moove == 'below':
                    movement_vector[idx] = - self.Nx
                    #print(f"Cellule {idx} se déplace en bas.")                    
                
                #If stay, movement is 0 (by default)

                #print(movement_vector)

        self.pos = self.pos + movement_vector #Update positions

        #Update pos in the grid of CD4 and CD8
        self.T_CD4 = np.zeros(self.Nx**2)
        for idx, i in enumerate(self.Pheno_CD4):
            if i != 0:  #Check if we have CD4 in the cell
                self.T_CD4[self.pos[idx]] += 1 #Put the quantity of CD4 in the grid Nx*Nx

        self.T_CD8 = np.zeros(self.Nx**2)
        for idx, i in enumerate(self.Pheno_CD8):
            if i !=0 :  #Check if we have CD8 in the cell
                self.T_CD8[self.pos[idx]] += 1  #Put the quantity of CD8 in the grid Nx*Nx
                
        self.T = self.T_CD8 + self.T_CD4 #Initial T-cells density in each cell of the gris (CD4 +CD8)

        #multiplier T par un coefficient pour la densité ?
        self.w = self.n + self.T #mise à jour de nos densités

    def update_density_tumors(self, new_density):
        """
        Update tumors density in tcells_mvt and cytokine_edp.

        Args:
            new_density (numpy.ndarray): List of the new density for tumors.
        """
        self.n = np.array(new_density)


