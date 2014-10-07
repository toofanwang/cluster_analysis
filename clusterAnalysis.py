"""Cluster analysis and simulation code for super-resolution
microscopy.

This file contains a number of functions for simulating and analyzing
clustered point localization data from super-resolution fluorescence
microscopy experiments.

"""

__author__ = 'Kyle M. Douglass'
__email__ = 'kyle.m.douglass@gmail.com'
__version__ = 0.4

import numpy as np

class simExperiment():
    """This is a simulation of a cluster analysis experiment. The
    experiment contains a number of clusters of randomly distributed
    points.

    It is possible to compute the radius of gyration of each cluster
    and to simulate the effects of a non-zero localization uncertainty
    on determining the cluster's size.

    """
    
    def __init__(self, numClusters, ptsPerCluster, clusterWidth, distr):
        """Randomly distribute locations of all points.

        """
        self.numClusters = numClusters
        self.ptsPerCluster = ptsPerCluster
        self.clusterWidth = clusterWidth
        self.distr = distr.lower()

        distrTypes = ['gaussian', 'uniform']
        assert self.distr in distrTypes

        if self.distr == 'gaussian':
            self.points = clusterWidth * np.random.randn(ptsPerCluster, 3, numClusters)
        elif self.distr == 'uniform':
            self.points = 2 * clusterWidth * (np.random.rand(ptsPerCluster, 3, numClusters) - 0.5)

    def findRadGyr(self, points):
        """Finds the radius of gyration of each cluster.

        """
        centers = np.mean(points, axis = 0)
        deltaR = points - centers

        Rg2 = np.sum(np.sum(deltaR**2, axis = 1), axis = 0) / self.ptsPerCluster

        return np.sqrt(Rg2)

    def bumpPoints(self, locPrec):
        """Randomly bumps the position of all points.

        This function simulates the actual measurement of the points
        within a cluster by bumping the positions of each point randomly
        in each direction. The distance bumped in each direction is
        determined from a multi-modal Gaussian distribution whose
        standard deviations are equivalent to the localization precision
        in x, y, and z.

        Parameters
        ---------
        locPrec : tuple of floats
            Contains the localization precision in x, y, and z.

        Returns
        -------
        bumpedPoints : ndarray

        """
        pointShape = self.points.shape
        bumpDistX = locPrec[0] * np.random.randn(ptsPerCluster, numClusters)
        bumpDistY = locPrec[1] * np.random.randn(ptsPerCluster, numClusters)
        bumpDistZ = locPrec[2] * np.random.randn(ptsPerCluster, numClusters)

        bumpedPoints = np.zeros(self.points.shape)
        bumpedPoints[:,0,:] = self.points[:,0,:] + bumpDistX
        bumpedPoints[:,1,:] = self.points[:,1,:] + bumpDistY
        bumpedPoints[:,2,:] = self.points[:,2,:] + bumpDistZ

        return bumpedPoints

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    numClusters = 10
    ptsPerCluster = 1000
    clusterWidth = 200
    distr = 'uniform'
    bumpDist = (10, 10, 10)
    
    myExp = simExperiment(numClusters, ptsPerCluster, clusterWidth, distr)
    Rg = myExp.findRadGyr(myExp.points)
    bp = myExp.bumpPoints(bumpDist)
    
    fig = plt.figure(figsize=(8,8))
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(myExp.points[:,0,0], myExp.points[:,1,0], myExp.points[:,2,0], color = 'blue')
    ax.scatter(bp[:,0,0], bp[:,1,0], bp[:,2,0], color = 'red')
    plt.show()









