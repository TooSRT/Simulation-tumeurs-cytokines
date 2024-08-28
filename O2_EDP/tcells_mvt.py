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
    def __init__(self, Nx, pos0, w0, T0, n0, w_max, delta_x, delta_t, D_tcells, Pheno_CD4, Pheno_CD8, cytokine_edp_instance):
        """
        Initialize a Density_EDP object.

        Args:
            nb_cells (int): Number of cells in the grid.
            Nx (int): Size of the grid.
            cells0 (numpy.ndarray): Initial cancer cell vector.
            w0 (numpy.ndarray): Initial density vector.
            n0 (numpy.ndarray): Initial density vector (tumor).
            T0 (numpy.ndarray): Initial density vector (T-cells).
            n_max (int): Maximum density value.
            delta_x (float): Spatial step size.
            delta_t (int): Time step size (in hours).
            Pheno_actif_prod (numpy.ndarray): List of cytokines producer (CD8)
            Pheno_actif_cons (numpy.ndarray): List of cytokines consummer (CD4)
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


        self.cytokine_edp = cytokine_edp_instance
        self.Rc_vect = self.cytokine_edp.Rc_vect
        self.cyto = self.cytokine_edp.cyto

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
            numpy.ndarray: Vector of producers.
        """
        return self.Pheno_actif_prod
    
    def Pheno_CD8(self):
        """
        Get the list of Tcells CD8

        Returns:
            numpy.ndarray: Vector of consummers.
        """
        return self.Pheno_actif_cons
    
    def update_pheno(self):
        """
        Updates the active phenotypes of T-cells based on their properties.
        """
        Pheno_CD4 = self.Pheno_CD4
        Pheno_CD8 = self.Pheno_CD8
        #List of activity of each tcells
        Active_CD4 = np.zeros(len(Pheno_CD4)) 
        Active_CD8 = np.zeros(len(Pheno_CD8))
        Inactive_CD4 = np.ones(len(Pheno_CD4)) 
        Inactive_CD8 = np.ones(len(Pheno_CD8))
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
            if self.cytokine_edp.Rc_vect[idx]*self.cytokine_edp.cyto[self.pos][idx] > 2: #Check concentration consumption of consumer
                self.Tcells_memorize[idx]=True #save T-cells that are under cytokine influence
                Inactive_CD4[idx] = 0 
                Inactive_CD8[idx] = 0 
                Active_CD4[idx] = 1 
                Active_CD8[idx] = 1

#-----------T-cells loose cytokine influence or don't have enough cytokines to become active-----------
            #Définir le taux seuil à partir du quel une cellule T doit se maintenir pour rester sous l'influence des cytokines
            elif self.cytokine_edp.Rc_vect[idx]*self.cytokine_edp.cyto[self.pos][idx] <= 1: #Check concentration consumption of consumer
                self.Tcells_memorize[idx]=False #T-cells is not under cytokine influence anymore
                Inactive_CD4[idx] = 1
                Inactive_CD8[idx] = 1
                Active_CD4[idx] = 0
                Active_CD8[idx] = 0

    def movement(self):
        """
        Perform cell movement.

        Args:
            Rc_vect (list): concentration in cytokine 
        """
        l=len(self.pos)
        movement_vector = np.zeros(len(self.pos),dtype=int) #initialize our movement vector to update position
        prob_moove = np.zeros((l,3)) #initialize our probability vector for movement
        for idx, i in enumerate(self.pos):

#-----------T-cells under cytokine influence or have interacted with tumors-----------

            if self.Tcells_memorize[idx]: #If our T cells has already been influenced by cytokines or interacted with tumors
                #Moove to left
                if i % self.Nx != 0: #ne doit pas se trouver sur la colonne gauche
                    T_left = 0
                else:
                    T_left = 0
                #Moove to right
                if i % self.Nx != self.Nx - 1 : #ne doit pas se trouver sur la colonne droite
                    T_right = 0
                else:
                    T_right = 0
                #Stay
                T_stay = 1 - T_left - T_right
                '''
                #Moove below
                if 0 < i < len(self.Nx): 
                    T_below=....
                else:
                    T_below=0

                #Moove upper
                if  len(self.Nx*self.Nx) -  len(self.Nx) < i < len(self.Nx*self.Nx) 
                    T_upper=...
                else:
                    T_upper=0
                '''       
                prob_moove[idx, 0] = T_left
                prob_moove[idx, 1] = T_right
                prob_moove[idx, 2] = T_stay
                #print(prob_moove)

                #Choose a direction based on probability 
                moove = np.random.choice(['left', 'right', 'stay'], p=[T_left, T_right, T_stay])
            
                if moove == 'left':
                    movement_vector[idx] = -1 
                    #print(f"Cellule {idx} se déplace vers la gauche.")
                elif moove == 'right':
                    movement_vector[idx] = 1 
                    #print(f"Cellule {idx} se déplace vers la droite.")
                #If stay, movement is 0 (by default)
                else:
                    pass
                    #movement_vector[idx] = 0
                    #print(f"Cellule {idx} reste sur place.")
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
                #Stay
                T_stay = 1 - T_left - T_right
                '''
                #Moove below
                if 0 < i < len(self.Nx): 
                    T_below=....
                else:
                    T_below=0

                #Moove upper
                if  len(self.Nx*self.Nx) -  len(self.Nx) < i < len(self.Nx*self.Nx) 
                    T_upper=...
                else:
                    T_upper=0
                '''
            
                prob_moove[idx, 0] = T_left
                prob_moove[idx, 1] = T_right
                prob_moove[idx, 2] = T_stay

                #Choose a direction based on probability 
                #print(prob_moove)
                moove = np.random.choice(['left', 'right', 'stay'], p=[T_left, T_right, T_stay])
            
                if moove == 'left':
                    movement_vector[idx] = -1 
                    #print(f"Cellule {idx} se déplace vers la gauche.")
                elif moove == 'right':
                    movement_vector[idx] = 1 
                    #print(f"Cellule {idx} se déplace vers la droite.")
                #If stay, movement is 0 (by default)
                else:
                    pass
                    #movement_vector[idx] = 0
                    #print(f"Cellule {idx} reste sur place.")
                #print(movement_vector)

        self.pos = self.pos + movement_vector #Update positions

        self.T = np.zeros(self.Nx**2) #Initial T-cells density in each case
        for i in self.pos:
                self.T[i] += 1

        #multiplier T par un coefficient pour la densité ?

        self.w = self.n + self.T #mise à jour de nos densités

    def update_density_tumors(self, new_density):
        """
        Update tumors density in tcells_mvt.

        Args:
            new_density (numpy.ndarray): List of the new density for tumors.
        """
        self.n = np.array(new_density)


