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

class Simulation2:
    def __init__(self, nb_tumor, unit, distrib, tol, Nb_cells_cyt, Nx, delta_x, delta_t, Dn, D_cytokine, w_max, rn, Tau_p, Tau_c, P_prod, P_cons, D_tcells, alpha_c):
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
            Rp (float): Cytokine production.
            Rc (float): cytokine consumption.
            Nb_cells_cyt (int): Number of cells producing cytokines.
            P_prod (float): probability production of cytokines for a immune cell
            P_cons (float): probability production of cytokines for a immune cell
        """     
        Nx = int(Nx)
        cells0 = self.init_cells0(Nx**2, nb_tumor, distrib)
        c0 = np.zeros(Nx**2)  #Initial cytokine concentration
        n0 = np.bincount(cells0, minlength=Nx**2) #Initial tumor density

        pos0 = [5000] #np.random.randint(0, int(Nx*Nx), Nb_cells_cyt) #self.init_pos0() permet de modifier la position et d'ajouter des sources
        T0 = np.zeros(Nx**2) #Initial T-cells density in each case
        for i in pos0:
                T0[i] += 1

        w0 = T0 + n0 #Initial density 
        
        self.tcells_mvt_instance = Tcells_mvt(Nx, pos0, w0, T0, n0, w_max, delta_x, delta_t, D_tcells)
        self.cytokine_edp = cytokine_EDP(Nx, c0, pos0, tol, delta_x, delta_t, D_cytokine, Tau_p, Tau_c, P_prod, P_cons, alpha_c, tcells_mvt=self.tcells_mvt_instance)
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
    
    def init_pos0(self) :
        """
        Initializes the initial positions.

        Returns:
            np.array: Array containing initial positions.
        """
        pos0 = np.zeros(1)
        return pos0
    
    def init_c0(self) :
        """
        Initializes the initial cytokine concentration.

        Returns:
            np.array: Array containing initial cytokine concentration.
        """
        c0 = np.zeros(100)
        return c0

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

    '''def density_plot_clip(self):
        """
        Creates a video clip from generated plot images.
        """
        # Directory containing plot images
        images_folder = './plot/density_pictures'

        # Get list of image file names in folder
        image_files = [img for img in os.listdir(images_folder) if img.endswith(".png")]

        # Sort file names
        image_files = sorted(image_files, key=lambda img: int(re.findall(r'\d+', img)[0]))

        # Create list containing full paths to images
        image_paths = [os.path.join(images_folder, img) for img in image_files]

        # Create video clip from images
        video_clip = ImageSequenceClip(image_paths, fps=8)

        # Export video
        video_clip.write_videofile("./plot/density_clip.mp4", codec="libx264")
    '''
    def one_time_step(self):
        """
        Performs a single time step of simulation.
        """
        self.cytokine_edp.cytokine_diffusion()
        cellsmouv, cellspro, choice = self.density_edp.proliferation()
        m0 = 0 #len(cellsmouv) si on veut que les cellules bougent
        self.density_edp.movement(cellsmouv, cellspro, m0, choice)
        self.density_edp.cells_size = np.append(self.density_edp.cells_size, len(self.density_edp.cells))

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
