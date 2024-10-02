# export PYTHONPATH="${PYTHONPATH}:/home/shij0d/Documents/Dis_Spatial"

import unittest
from src.generation import GPPSampleGenerator
from sklearn.gaussian_process.kernels import Matern
import math
import numpy as np

class TestGPPEstimation(unittest.TestCase):
    
    def setUp(self):
        
        self.alpha=3
        self.length_scale=2
        self.nu=0.5
        self.kernel=self.alpha*Matern(length_scale=self.length_scale,nu=self.nu)
        self.num=4000
        self.min_dis=0.2
        self.l=math.sqrt(2*self.num)
        self.extent=-self.l/2,self.l/2,-self.l/2,self.l/2,
        self.coefficients=[-1,2,3,-2,1]
        self.noise_level=2
        self.gpp_gen=GPPSampleGenerator(num=self.num,min_dis=self.min_dis,extent=self.extent,kernel=self.kernel,coefficients=self.coefficients,noise=self.noise_level)

    def test_random_points(self):
        points = self.gpp_gen.random_points()
        self.assertEqual(len(points), self.num)
        for i in range(self.num):
            for j in range(i+1, self.num):
                dist = np.linalg.norm(np.array(points[i]) - np.array(points[j]))+0.01
                self.assertGreaterEqual(dist, self.min_dis)
        for point in points:
            self.assertGreaterEqual(point[0], self.extent[0])
            self.assertLessEqual(point[0], self.extent[1])
            self.assertGreaterEqual(point[1], self.extent[2])
            self.assertLessEqual(point[1], self.extent[3])

    def test_get_knots_random(self):
        locations = self.gpp_gen.random_points()
        m = 5
        knots = self.gpp_gen.get_knots_random(locations, m)
        self.assertEqual(len(knots), m)
        for knot in knots:
            self.assertIn(knot, locations)

    def test_get_knots_grid(self):
        m = 4
        knots = self.gpp_gen.get_knots_grid(m)
        self.assertEqual(len(knots), m)
        for knot in knots:
            self.assertGreaterEqual(knot[0], self.extent[0])
            self.assertLessEqual(knot[0], self.extent[1])
            self.assertGreaterEqual(knot[1], self.extent[2])
            self.assertLessEqual(knot[1], self.extent[3])

    def test_generate_x_epsilon(self):
        value, X = self.gpp_gen.generate_x_epsilon()
        self.assertEqual(value.shape[0], self.num)
        self.assertEqual(X.shape[0], self.num)
        self.assertEqual(X.shape[1], len(self.coefficients))

    def test_generate_obs_gp(self):
        data = self.gpp_gen.generate_obs_gp()
        self.assertEqual(data.shape[0], self.num)
        self.assertEqual(data.shape[1], len(self.coefficients) + 3)  # x, y, z, coefficients

    def test_generate_obs_gpp(self):
        m = 4
        data, knots = self.gpp_gen.generate_obs_gpp(m, method="grid")
        self.assertEqual(data.shape[0], self.num)
        self.assertEqual(len(knots), m)
        for knot in knots:
            self.assertGreaterEqual(knot[0], self.extent[0])
            self.assertLessEqual(knot[0], self.extent[1])
            self.assertGreaterEqual(knot[1], self.extent[2])
            self.assertLessEqual(knot[1], self.extent[3])
        self.assertEqual(data.shape[1], len(self.coefficients) + 3)  # x, y, z, coefficients

    def test_data_split(self):
        data = self.gpp_gen.generate_obs_gp()
        J = 2
        dis_data = self.gpp_gen.data_split(data, J)
        self.assertEqual(len(dis_data), J)
        self.assertEqual(sum(len(part) for part in dis_data), self.num)
        for part in dis_data:
            self.assertEqual(part.shape[1], data.shape[1])

testgpp=TestGPPEstimation()
testgpp.setUp()
testgpp.test_random_points()
testgpp.test_get_knots_random()
testgpp.test_get_knots_grid()
testgpp.test_generate_x_epsilon()
#testgpp.test_generate_obs_gp()
testgpp.test_generate_obs_gpp()
testgpp.test_data_split()

#
import matplotlib.pyplot as plt


points = testgpp.gpp_gen.random_points()
knots_grid=testgpp.gpp_gen.get_knots_grid(100)
knots_random=testgpp.gpp_gen.get_knots_random(points,100)

# Unpack points and knots into x and y coordinates
x_points, y_points = zip(*points)
x_knots_grid, y_knots_grid = zip(*knots_grid)
x_knots_random, y_knots_random = zip(*knots_random)

# Create the plot
plt.figure(figsize=(8, 6))
plt.scatter(x_points, y_points, color='blue', label='Points')
plt.scatter(x_knots_grid, y_knots_grid, color='red', marker='x', label='Knots_grid')
plt.scatter(x_knots_random, y_knots_random, color='black', marker='o', label='Knots_random')


# Adding titles and labels
plt.title('2D Points and Knots Visualization')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')

# Adding a legend
plt.legend()

# Display the plot
plt.grid(True)
plt.show()
