import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
from mpl_toolkits.mplot3d import Axes3D
import random
from sklearn import linear_model
 

class SOM(nn.Module):
    """
    2D Self-organizing map with Gaussian kernel similarity function
    and particle swarm optimization for update step
    """
    def __init__(self, particle_inits, x, y, dim, num_iters, learning_radius=0.5, sigma=None, cognitive=.01, social=.1, inertia=1e-3):
        super(SOM, self).__init__()
        self.particle_inits = particle_inits
        self.x = x
        self.y = y
        self.dim = dim
        self.num_iters = num_iters
        self.learning_radius = learning_radius
        self.cognitive = cognitive
        self.social = social
        self.inertia = inertia

        if sigma is None:
            self.sigma = max(self.x, self.y) / 2.0
        self.grid_locations = torch.LongTensor(np.array(list(self.neuron_locations())))
        self.particles = torch.rand(self.x * self.y, dim)
        for d in range(len(self.particles[0])):
            self.particles[:, d] *= particle_inits[d]
        self.velocities = torch.zeros(self.x * self.y, dim)
        self.pdist = nn.PairwiseDistance(p=2)

    def neuron_locations(self):
        # Initialize positions in grid-like arrangement
        for i in range(self.x):
            for j in range(self.y):
                yield np.array([i, j])

    def particle_swarm_update(self, bmu_idx, neighborhood, radius):
        global_best = self.particles[bmu_idx]
        global_nbest = neighborhood[bmu_idx].float()
        for idx in range(len(self.particles)):
            particle = self.particles[idx]
            velocity = self.velocities[idx]

            # Check if particle is in learning radius
            if global_nbest - neighborhood[idx].float() > radius:
                continue

            # Find centroid of particles within radius of current point
            boolArr = neighborhood <= neighborhood[idx].float() + radius
            region = self.particles[boolArr]
            centroid = region.mean(axis=0)

            # Update each dimension of particle
            for dim in range(self.dim):
                r1 = np.random.rand()
                r2 = np.random.rand()

                # Move in individual direction, which is towards centroid
                v_cognitive = self.cognitive * r1 * (centroid[dim] - particle[dim])

                # Move towards global best
                v_social = self.social * r2 * (global_best[dim] - particle[dim])

                # Update position and velocity
                v_update = self.inertia * velocity[dim] + v_cognitive + v_social
                self.velocities[idx][dim] = v_update
                p_update = particle[dim] + v_update
                self.particles[idx][dim] = particle[dim] + v_update

    def kernel_func(self, bmu_loc, sigma):

        # Gaussian kernel function
        bmu_distance_squares = torch.sum(torch.pow(self.grid_locations.float() \
            - torch.stack([bmu_loc for i in range(self.x*self.y)]).float(), 2), 1)
        return torch.exp(torch.neg(torch.div(bmu_distance_squares, sigma**2)))

    def forward(self, input, iter_num):

        # Find distance from input vector to each particle, which is location of each particle
        dists = self.pdist(torch.stack([input for i in range(self.x*self.y)]), self.particles)
        
        # Best matching unit is grid location in x by y grid
        _, bmu_idx = torch.min(dists, 0)
        bmu_loc = self.grid_locations[bmu_idx,:]
        bmu_loc = bmu_loc.squeeze()

        # Update learning rate and sigma for time decay
        decay = 1.0 - iter_num/self.num_iters
        lr_decay = self.learning_radius * decay
        sigma_decay = self.sigma * decay

        # Gaussian kernel function for neighborhood function
        neighborhood = self.kernel_func(bmu_loc, sigma_decay)
        
        # Update step with PSO
        self.particle_swarm_update(bmu_idx, neighborhood, lr_decay)


def train_SOM(data, model, iters, csv_name):
    for it in tqdm(range(iters)):
        for i in tqdm(range(len(data))):
            som(data[i], it)
    particles = som.particles.numpy()
    model = linear_model.LinearRegression()
    model.fit(particles[:,1:], particles[:,0])
    coefs = model.coef_
    intercept = model.intercept_
    diffs = []
    for particle in curr_list:
        diffs.append(particle[0] - (intercept + coefs[0]*particle[1] + coefs[1]*particle[2]))
    pd.DataFrame(np.array(diffs), columns=['EURUSD Fluctuations']).to_csv(csv_name)


if __name__=='__main__':

    # Train currency inputs
    currencies = pd.read_csv('../data/input.csv')
    curr_list = currencies[['EURUSD', 'USDJPY', 'EURJPY']].to_numpy()
    particle_inits = curr_list.mean(axis=0)

    data = list()
    for i in range(curr_list.shape[0]):
        data.append(torch.FloatTensor(curr_list[i,:]))
    
    #Train a 20x30 SOM with 100 iterations
    m = 4
    n = 6
    n_iter = 10
    som = SOM(particle_inits, m, n, 3, n_iter)
    train_SOM(data, som, n_iter, '../data/raw_data/EURUSD_fluctuations.csv')

    # Plot
    # particles = som.particles.numpy()
    # fig = plt.figure()
    # ax = Axes3D(fig)
    # plt.title('Currency SOM')
    # ax.scatter(particles[:,0], particles[:,1], particles[:,2])
    # plt.show()

    # # Get line of best fit
    # ax.set_xlabel("EURUSD")
    # ax.set_ylabel("USDJPY")
    # ax.set_zlabel("EURJPY")
    # ax.set_xlim(np.min(particles[:, 0]), np.max(particles[:, 0]))
    # ax.set_ylim(np.min(particles[:, 1]), np.max(particles[:, 1]))
    # ax.set_zlim(np.min(particles[:, 2]), np.max(particles[:, 2]))

    # model = linear_model.LinearRegression()
    # model.fit(particles[:,1:], particles[:,0])
    # coefs = model.coef_
    # intercept = model.intercept_
    # for particle in curr_list:
    #     print(particle[0], intercept + coefs[0]*particle[1] + coefs[1]*particle[2])

    # centroid_grid = [[] for i in range(m)]
    # particles = som.particles
    # positions = som.grid_locations
    # for i, loc in enumerate(positions):
    #     centroid_grid[loc[0]].append(particles[i].numpy())
    # image_grid = np.array(centroid_grid)
    # for d in range(image_grid.shape[-1]):
    #     mean = np.mean(image_grid[:,:,d])
    #     std = np.std(image_grid[:,:,d])
    #     minval = mean-2*std
    #     maxval = mean+2*std

    #     diff =  maxval - minval
    #     for x in range(image_grid.shape[0]):
    #         for y in range(image_grid.shape[1]):
    #             image_grid[x,y,d] = (image_grid[x,y,d] - minval) / diff

    # plt.imshow(image_grid)
    # plt.show()

