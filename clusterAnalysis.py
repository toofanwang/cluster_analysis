"""
Cluster analysis and simulation code for super-resolution microscopy.

This file contains a number of functions for simulating and analyzing
clustered point localization data from super-resolution fluorescence
microscopy experiments.
"""

__author__ = 'Kyle M. Douglass'
__email__ = 'kyle.m.douglass@gmail.com'
__version__ = 0.1

import numpy as np

class simExperiment():
    """
    Contains a given number of clusters of points. The cluster centers
    are randomly distributed, as are the locations of the points within
    each cluster.
    """

    def __init__(self, numClusters, ptsPerCluster, clusterWidth, distr):
        """
        Randomly distribute locations of all points.
        """
        self.numClusters = numClusters
        self.ptsPerCluster = ptsPerCluster
        self.clusterWidth = clusterWidth

        self.points = clusterWidth * np.random.randn(ptsPerCluster, 3, numClusters)

    def findRadGyr(self):
        centers = np.mean(self.points, axis = 0)
        deltaR = self.points - centers

        Rg2 = np.sum(np.sum(deltaR**2, axis = 1), axis = 0) / self.ptsPerCluster

        return np.sqrt(Rg2)

if __name__ == '__main__':
    myExp = simExperiment(10, 100, 100, 'gaussian')
    Rg = myExp.findRadGyr()
    print(Rg)










