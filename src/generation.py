from __future__ import annotations
import numpy as np
import random
from sklearn.gaussian_process.kernels import Kernel

import numpy as np
import random
import math

class GPPSampleGenerator:
    def __init__(self,num: int, min_dis: float, extent: tuple[float, float, float, float],kernel: Kernel,coefficients: list[float], noise: float,seed=2024):
        """
        Initializes a SampleGenerator object with parameters for generating synthetic data.

        Parameters:
        - num (int): The number of random points to generate.
        - min_dis (float): The minimum distance between any two points.
        - extent (tuple): A tuple containing the extent of the area to generate points.
            Should be in the format (x_min, x_max, y_min, y_max).
        - kernel (Kernel): A kernel function for covariance calculation.
        - coefficients (list): A list of coefficients for generating X values.
        - noise (float): The standard deviation of noise in the generated data.
        - seed (int, optional): Seed for random number generation (default is 2024).
        """
        self.num=num
        self.min_dis=min_dis
        self.extent=extent
        self.kernel=kernel
        self.noise=noise
        self.coefficients=coefficients
        self.seed=seed
        

    def random_points(self) -> list[tuple[float, float]]:
        """
        Generates a list of random points within the specified extent, 
        with a minimum distance between points along both x and y axes.

        Parameters:
        - num (int): The number of random points to generate.
        - min_dis (float): The minimum distance between any two points.
        - extent (tuple): A tuple containing the extent of the area to generate points.
            Should be in the format (x_min, x_max, y_min, y_max).

        Returns:
        - list: A list of tuples, each representing a random point (x, y) within the extent.
        """
        x_min, x_max, y_min, y_max = self.extent  # Unpack extent tuple

        # Generate grid of points with minimum distance min_dis
        xs = np.arange(x_min, x_max, self.min_dis)
        if xs[-1] + self.min_dis <= x_max:
            xs = np.append(xs, x_max)

        ys = np.arange(y_min, y_max, self.min_dis)
        if ys[-1] + self.min_dis <= y_max:
            ys = np.append(ys, y_max)

        random.seed(self.seed)
        np.random.seed(self.seed)
        # Create list of all possible points with added noise
        noise_scale = 0.4 * self.min_dis  # Adjust this value to control noise level
        points = [(x + np.random.uniform(-noise_scale, noise_scale), y + np.random.uniform(-noise_scale, noise_scale)) for x in xs for y in ys]

        # Randomly sample num points from the list
        
        ran_points = random.sample(points, self.num)

        return ran_points
    
    def get_knots_random(self, locations, m):
        """
        Randomly selects m knot points from given locations.

        Parameters:
        - locations (list): List of tuples representing locations.
        - m (int): Number of knot points to select.

        Returns:
        - list: List of tuples representing randomly selected knot points.
        """
        random.seed(self.seed)
        knots = random.sample(locations, m)
        return knots
    
    def get_knots_grid(self, m):
        """
        Generates knot points on a grid within the extent.

        Parameters:
        - m (int): Number of knot points to generate.

        Returns:
        - list: List of tuples representing grid-generated knot points.
        """
        l = int(math.sqrt(m))
        x_min, x_max, y_min, y_max = self.extent  # Unpack extent tuple

        # Generate grid of points with minimum distance min_dis
        xs = np.linspace(x_min, x_max, l)
        ys = np.linspace(y_min, y_max, l)
        knots = [(x, y) for x in xs for y in ys]

        return knots
        
    def generate_x_epsilon(self):
        """
        Generates linear term with added noise.

        Returns:
        - numpy.ndarray: Array of X values adjusted with noise.
        """

        p = len(self.coefficients)  # Number of coefficients

        # Generate design matrix X with normal distribution
        np.random.seed(self.seed)
        X = np.random.normal(size=(self.num, p))

        # Generate noise term epsilon with normal distribution
        epsilon = np.random.normal(scale=self.noise, size=self.num)
        epsilon=epsilon.reshape(-1,1)
        # Convert coefficients to NumPy array and reshape for matrix multiplication
        coefficients_array = np.array(self.coefficients).reshape(p, 1)

        value= X @ coefficients_array + epsilon

        return value,X
    
    def generate_obs_gp(self):
        """
        Generates observations following Gaussian Process.

        Returns:
        - numpy.ndarray: Array of generated observations.
        """

        locations=self.random_points()

        # Compute the covariance matrix using the kernel function
        cov = self.kernel(np.array(locations))

        # Initialize the mean vector as zeros
        mean = np.zeros(self.num)

        # Generate random samples (y) from a multivariate normal distribution
        np.random.seed(self.seed)
        y = np.random.multivariate_normal(mean, cov)
        y=y.reshape(-1,1)
        # Generate observations based on linear model: z = X @ coefficients + y + epsilon
        value,X=self.generate_x_epsilon()
        z = y+value
        data=np.hstack((locations,z,X))
        return data
    
    
    def generate_obs_gpp(self,m,method):

        """
        Generates observations following Gaussian Predictive Process(GPP).

        Parameters:
        - m (int): Number of knot points.
        - method (str): Method for selecting knot points ("random" or "grid").

        Returns:
        - numpy.ndarray: Array of generated observations.
        """

        locations=self.random_points()    

        # Compute the number of knots (m)
        if method=="random":
            knots=self.get_knots_random(locations,m)
        elif method=="grid":
            knots=self.get_knots_grid(m)
        else:  
            raise("Invalid choice. Please select from 'random' or 'grid'.")  
        m = len(knots)

        # Generate random eta values from a multivariate normal distribution
        np.random.seed(self.seed)
        mean_eta = np.zeros(m)
        cov_eta = self.kernel(np.array(knots))
        eta = np.random.multivariate_normal(mean_eta, cov_eta)
        eta=eta.reshape(-1,1)
        # Compute the product B = K(locations, knots) @ inv(cov_eta)
        B = self.kernel(np.array(locations), np.array(knots)) @ np.linalg.inv(cov_eta)
        
        

        # Generate y using the GPP model: y = B @ eta
        y = B @ eta

        # res1=[]
        # for i in range(100):  
        #     eta = np.random.multivariate_normal(mean_eta, cov_eta)
        #     eta=eta.reshape(-1,1)
        #     y1 = B @ eta
        #     res1.append((y1.T@y1).item())
        
        
        
        # mean=np.zeros(y.shape[0])
        # cov=B@cov_eta@B.T
        # avg=0
        # res=[]
        # for i in range(100):
        #     y2=np.random.multivariate_normal(mean=mean,cov=cov).reshape(-1,1)
        #     res.append(y2.T@y2)
        # avg=avg/100
        # print(avg)

        value,X=self.generate_x_epsilon()
        z = y+value
        data=np.hstack((locations,z,X))
        return data,knots
    def generate_obs_gpp_est_pre(self,N_pre,m,method):

        """
        Generates observations following Gaussian Predictive Process(GPP) and divide into two data set for estimation and prediction,respetively.

        Parameters:
        - m (int): Number of knot points.
        - method (str): Method for selecting knot points ("random" or "grid").

        Returns:
        - numpy.ndarray: Array of generated observations.
        """

        locations=self.random_points()    
        N=len(locations)
        N_est=N-N_pre
        locations_est=locations[0:N_est]
        locations_pre=locations[N_est:]
        # Compute the number of knots (m)
        if method=="random":
            knots=self.get_knots_random(locations_est,m)
        elif method=="grid":
            knots=self.get_knots_grid(m)
        else:  
            raise("Invalid choice. Please select from 'random' or 'grid'.")  
        m = len(knots)

        # Generate random eta values from a multivariate normal distribution
        np.random.seed(self.seed)
        mean_eta = np.zeros(m)
        cov_eta = self.kernel(np.array(knots))
        eta = np.random.multivariate_normal(mean_eta, cov_eta)
        eta=eta.reshape(-1,1)
        # Compute the product B = K(locations, knots) @ inv(cov_eta)
        B = self.kernel(np.array(locations), np.array(knots)) @ np.linalg.inv(cov_eta)
        
        

        # Generate y using the GPP model: y = B @ eta
        y = B @ eta


        value,X=self.generate_x_epsilon()
        z = y+value
        data=np.hstack((locations,z,X))
        
        return data,knots
    
    
    def data_split(self,data,J,method='random'):
        '''
        method: random, by area, rnearest
        '''
        if method=='random':
            dis_data=np.array_split(data,J,axis=0)
        if method=='1':
            1
        return dis_data
    
    
    

        