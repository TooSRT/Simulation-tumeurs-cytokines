import shutil
import time
import numpy as np
import os
import re
from Density_EDP.density_edp import Density_EDP
from Density_EDP.grid_density import Density_Grid
from O2_EDP.cytokine_edp import cytokine_EDP 
from O2_EDP.grid_cytokine import cytokine_Grid
from O2_EDP.tcells_mvt import Tcells_mvt

class Simulation:
    def __init__(self, nb_tumor, unit, distrib, tol, Nb_cells_cyt, Nx, delta_x, delta_t, Dn, D_cytokine, w_max, rn, Tau_p_CD4, Tau_c_CD4, Tau_c_CD8, P_prod, P_cons, D_tcells, alpha_c):
        """
        Initializes an instance of the Simulation class.

        Args:
            nb_tumor (int): Number of initial tumor cells.
            tol (float): Tolerance for numerical computations.
            Nx (int): Size of the grid.
            unit (str): Measurement unit of the grid area (default: "cm").
            distrib (str): Distribution of initial tumor cells (default: "uniform").
            delta_x (float): Spatial step size.
            delta_t (float): Time step size.
            Dn (float): Diffusion coefficient for tumor cells.
            rn (float): Proliferation rate of tumor cells.
            w_max (float): Max density on each case.
            D_cytokine (float): Diffusion coefficient for cytokines.
            D_tcells (float): Diffusion coefficient for tcells.
            alpha_c (float): decay rate of Tcells cytokines.
            Nb_cells_cyt (int): Number of cells producing cytokines.
            P_prod (float): probability production of cytokines for a immune cell.
            P_cons (float): probability production of cytokines for a immune cell.
            Tau_p_CD4 (float): Cytokinew production by CD4.
            Tau_c_CD4 (float): Cytokines consumption by CD4.
            Tau_c_CD8 (float): Cytokines consumption by CD8.
        """     
        cells0 = self.init_cells0(Nx**2, nb_tumor, distrib)
        c0 = np.zeros(Nx**2)  #Initial cytokine concentration
        n0 = np.bincount(cells0, minlength=Nx**2) #Initial tumor density

        pos0 = [5000] #np.random.randint(0, int(Nx*Nx), Nb_cells_cyt) #modify positions and number of Tcells

        #Initialisation of cells and their phenotype
        Vect_unif = np.random.uniform(low=0.0, high=1.0, size=np.size(pos0)) #Vecteur suivant une loi uniforme sur [0,1]
        
        #Density of producer and consumer 
        Pheno_CD4 = np.zeros(len(pos0))  #Liste phenotype actif produisant n_prod
        Pheno_CD8 = np.zeros(len(pos0))  #Liste phenotype actif consommant n_cons
        
        #(revoir les probas utilisés)
        for j in range(len(pos0)):
            if Vect_unif[j] <= P_prod:  #Déterminer aléatoirement les CD4
                Pheno_CD4[j] = 1
            if Vect_unif[j] >= 1 - P_cons:  #Déterminer aléatoirement les CD8
                Pheno_CD8[j] = 1

        #Create a grid of size Nx*Nx and put the density of each Tcells type in each cell
        T_CD4 = np.zeros(Nx**2)
        for idx, i in enumerate(Pheno_CD4):
            if i != 0:  #Check if we have CD4 in the cell
                T_CD4[pos0[idx]] += 1 #Put the quantity of CD4 in the grid Nx*Nx

        T_CD8 = np.zeros(Nx**2)
        for idx, i in enumerate(Pheno_CD8):
            if i !=0 :  #Check if we have CD8 in the cell
                T_CD8[pos0[idx]] += 1  #Put the quantity of CD8 in the grid Nx*Nx
                
        T0 = T_CD8 + T_CD4 #Initial T-cells density in each cell of the grid (CD4 +CD8)

        w0 = T0 + n0 #Initial total density (Tcells + tumors cells)

        #Initial active and inactive CD4 and CD8 
        Active_CD4 = np.zeros(len(Pheno_CD4)) 
        Active_CD8 = np.zeros(len(Pheno_CD8))
        Inactive_CD4 = np.ones(len(Pheno_CD4)) 
        Inactive_CD8 = np.ones(len(Pheno_CD8))

        self.cytokine_edp = cytokine_EDP(Nx, c0, pos0, tol, delta_x, delta_t, D_cytokine, Tau_p_CD4, Tau_c_CD4, Tau_c_CD8, alpha_c, Pheno_CD4, Pheno_CD8, Active_CD4, Active_CD8, Inactive_CD4, Inactive_CD8, n0, T_CD4, T_CD8)
        self.tcells_mvt = Tcells_mvt(Nx, pos0, w0, T0, n0, w_max, delta_x, delta_t, D_tcells, Pheno_CD4, Pheno_CD8, T_CD4, T_CD8, self.cytokine_edp)
        self.density_grid = Density_Grid(len(n0), unit)
        self.o2_grid = cytokine_Grid(unit)
        self.density_edp = Density_EDP(Nx, cells0, w0, T0, n0, w_max, delta_x, delta_t, Dn, rn)
        self.prepare_plot()


    def init_cells0(self, nb_cells, nb_tumor, choice="uniform"):
        """
        Initializes the initial tumor cells according to the distribution choice.

        Args:
            nb_cells (int): Total number of cells in the grid.
            nb_tumor (int): Number of initial tumor cells.
            choice (str): Choice of distribution method ("uniform" or "gaussian").

        Returns:
            np.array: Array containing indices of the initial tumor cells.
        """
        np.random.seed(int(time.time()))
        cells0 = np.empty(0)
        if choice == "uniform":
            cells0 = np.random.randint(0, nb_cells, size=nb_tumor)
        elif choice == "gaussian":
            Nx = int(nb_cells ** 0.5)
            var = np.zeros((2, 2))
            var[0, 0] = Nx / 4
            var[1, 1] = Nx / 4
            multi = np.transpose(np.random.multivariate_normal(np.array([(Nx - 1) / 2, (Nx - 1) / 2]), var, size=nb_tumor) % Nx).astype(int)
            cells0 = (Nx * multi[0] + multi[1]).astype(int)
        else:
            raise ValueError("Invalid value for 'choice' parameter")
        return cells0

    def prepare_plot(self):
        """
        Prepares the directory for density and cytokines concentration plots by creating necessary directories.
        """
        # Full path of parent folder
        parent_folder = "./plot"
        
        # Remove 'plot' folder if exists
        if os.path.exists(parent_folder):
            try:
                shutil.rmtree(parent_folder)  # Remove folder and its contents recursively
                print(f"'{parent_folder}' folder successfully removed.")
            except Exception as e:
                print(f"Error while removing '{parent_folder}' folder: {e}")
        
        # Create 'plot' folder and empty 'density_pictures' and 'cytokine_pictures sub-folders
        try:
            os.makedirs(os.path.join(parent_folder, "density_pictures"))
            os.makedirs(os.path.join(parent_folder, "cytokine_pictures"))
            print(f"'{parent_folder}' folder created successfully.")
            print(f"'density_pictures' sub-folder created successfully.")
            print(f"'cytokine_pictures' sub-folder created successfully.")
        except Exception as e:
            print(f"Error while creating '{parent_folder}' folder or 'density_pictures' sub-folder or 'cytokine_pictures': {e}")
    
    def print_density(self, i):
        """
        Prints information about the density grid at the given time step.

        Args:
            i (int): Time step.
        """
        self.density_grid.print(self.density_edp.n, self.density_edp.w_max, self.density_edp.Nx, i * self.density_edp.delta_t, self.density_edp.cells_size[i])

    def print_cytokine(self, i):
        """
        Prints information about the cytokine concentration grid at the given time step.

        Args:
            i (int): Time step.
        """
        #self.o2_edp.plot_fig() # plot provisoire
        self.o2_grid.print(self.cytokine_edp.cyto, self.cytokine_edp.X, self.cytokine_edp.Y, self.cytokine_edp.Nx, i * self.cytokine_edp.delta_t)

    def growth_print(self):
        """
        Prints the tumor growth over time.
        """
        self.density_grid.growth(self.density_edp.cells_size)

    def one_time_step(self):
        """
        Performs a single time step of simulation.
        """
        #Tcells and cytokines
        self.cytokine_edp.cytokine_diffusion() #Perform cytokine diffusion by CD4 and update their phenotype 
        self.tcells_mvt.movement() #Perform T-cells movement
        self.cytokine_edp.update_positions(self.tcells_mvt.pos) #Update position of Tcells in cytokine_edp
        self.cytokine_edp.update_density_Tcells(self.tcells_mvt.T_CD4, self.tcells_mvt.T_CD8) ##Update density of CD4 and CD8 in cytokine_edp
        self.density_edp.update_density_tcells(self.tcells_mvt.T) #Update tumors density in tcells_mvt based on proliferation of tumor
        
        #Tumor
        cellsmouv, cellspro, choice = self.density_edp.proliferation()
        m0 = 0 #len(cellsmouv) si on veut que les cellules bougent
        self.density_edp.movement(cellsmouv, cellspro, m0, choice)
        self.density_edp.cells_size = np.append(self.density_edp.cells_size, len(self.density_edp.cells))
        self.tcells_mvt.update_density_tumors(self.density_edp.n) #Update tumors density in tcells_mvt based on proliferation of tumor
        self.cytokine_edp.update_density_tumors(self.density_edp.n)  #Update tumors density in cytokine_edp based on proliferation of tumor

    def load_simulation(self, iter_max, iter_print):
        """
        Loads and executes the simulation with specified parameters.

        Args:
            iter_max (int): Maximum number of simulation iterations.
            iter_print (int): Frequency of displaying simulation steps.
        """
        self.print_density(0)
        self.print_cytokine(0)
        for i in range(1, iter_max + 1):
            self.one_time_step()
            if i % iter_print == 0:
                self.print_density(i)
                self.print_cytokine(i)
                self.growth_print()
        #self.density_plot_clip()
