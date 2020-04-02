"""
Created on Wed Mar 18 12:55:05 2020

@author: User
"""

import numpy as np
import matplotlib.pylab as plt
import math
import dump

Avogadro = 6.02214076e23 
Boltzmann = 1.38064852e-23

def wallHitCheck(pos, vels, box):
    """ This function enforces reflective boundary conditions.
    All particles that hit a wall  have their velocity updated
    in tje opposite direction.
    @pos: atomic positions (ndarray)
    @vels: atomic velocity (ndarray, updated if collisions detected)
    @box: simulation box size (tuple)
    """
    ndims = len(box)

    for i in range(ndims):
        vels[((pos[:,i] <= box[i][0]) | (pos[:,i] >= box[i][1])),i] *= -0.4

def integrate(pos, vels, forces, mass, dt):
    """ A simple forward Euler integrator that moves the system in time 
    @pos: atomic positions (ndarray, updated)
    @vels: atomic velocity (ndarray, updated)
    """
    pos += vels * dt
    
    vels += forces * dt / mass[np.newaxis].T
    
def computeForce(pos, mass, vels, sepP, sepF, permFS, radius, temp, relax, dt, step):
    """ Computes the Langevin force for all particles
    @mass: particle mass (ndarray)
    @vels: particle velocities (ndarray)
    @temp: temperature (float)
    @relax: thermostat constant (float)
    @dt: simulation timestep (float)
    returns forces (ndarray)
    """

    natoms, ndims = vels.shape
    
    time = step * dt
    
    #Magnetic Force
    switch = 0.15e-11
    B0_og  = 0.8e-3
 
    if switch <= time < 3*switch:
        B0 = (B0_og/switch)*time - B0_og
        f = -1
    elif 3*switch <= time:
        B0=0
        f =1 
    else:
        B0 = -(B0_og/switch)*time + B0_og
        f = 1
    dis = 100*1e-8/500
    c = (((4/3)*math.pi*radius**3 * (sepP - sepF) * B0**2)/permFS)
    a = 1e-3
    
    magForce = (-1*f) * c[np.newaxis].T * (pos+dis)/((pos+dis)**2 + a**2)**4
    magForce[:, [0,2]] = 0
    
    
    #Brownian motion

    sigma = np.sqrt(2.0 * mass * temp * Boltzmann / (relax * dt))
    
    noise = np.random.randn(natoms, ndims) * sigma[np.newaxis].T
    
    #Viscous force

    visc = - (vels * mass[np.newaxis].T) / relax

    force = magForce + noise + visc
    
    return force

def removeCOM(pos, mass):
    """ Removes center of mass motion. This function is not used. """
    pos -= np.dot(mass, pos) / mass.sum()

def run(**args):
    """ This is the main function that solves Langevin's equations for
    a system of natoms usinga forward Euler scheme, and returns an output
    list that stores the time and the temperture.
    
    @natoms (int): number of particles
    @temp (float): temperature (in Kelvin)
    @mass (float): particle mass (in Kg)
    @relax (float): relaxation constant (in seconds)
    @dt (float): simulation timestep (s)
    @nsteps (int): total number of steps the solver performs
    @box (tuple): simulation box size (in meters) of size dimensions x 2
    e.g. box = ((-1e-9, 1e-9), (-1e-9, 1e-9)) defines a 2D square
    @ofname (string): filename to write output to
    @freq (int): write output every 'freq' steps
    
    @[radius]: particle radius (for visualization)
    
    Returns a list (of size nsteps x 2) containing the time and temperature.
    
    """

    natoms, dt, temp, box = args['natoms'], args['dt'], args['temp'], args['box']
    mass, relax, nsteps, radius = args['mass'], args['relax'], args['steps'], args['radius']
    sepP, sepF, permFS = args['sepP'], args['sepF'], args['permFS']
    ofname, freq = args['ofname'], args['freq']
    
    dim = len(box)
        
    #set to appear in a specific location of the box
    pos = np.random.rand(natoms,dim)
    for i in range(dim):
        if i == 0:
            pos[:,i] = box[0][1]/10
        if i == 1:
            pos[:,i] = box[1][1]/2
        if i == 2:
            pos[:,i] = 0
    
    if 'initial_vels' not in args:
        vels = np.random.rand(natoms,3)
    else:
        vels = np.ones((natoms,3)) * args['initial_vels']

    mass = np.ones(natoms) * mass #/ Avogadro #I define in grams in paras
    radius = np.ones(natoms) * radius
    step = 0

    output = []
    

    while step <= nsteps:
        step += 1

        # Compute all forces
        forces = computeForce(pos, mass, vels, sepP, sepF, permFS, radius, temp, relax, dt, step)
        
        # Move the system in time
        integrate(pos, vels, forces, mass, dt)

        # Check if any particle has collided with the wall
        wallHitCheck(pos,vels,box)

        # Compute output (temperature)
        ins_temp = np.sum(np.dot(mass, (vels - vels.mean(axis=0))**2)) / (Boltzmann * dim * natoms)
        output.append([step * dt, ins_temp])
        
        if not step%freq:
            dump.writeOutput(ofname, natoms, step, box, radius=radius, pos=pos, v=vels)

    return np.array(output)

if __name__ == '__main__':

#    params = {
#        'natoms': 10,
#        'temp': 300,
#        'mass': 0.001,
#        'radius': 120e-12,
#        'relax': 1e-13,
#        'dt': 1e-15,
#        'steps': 10000,
#        'freq': 100,
#        'box': ((0, 1e-8), (0, 1e-8), (0, 240e-12)),
#        'sepP': 7200,
#        'sepF': 0.712e-6,
#        'permFS': math.pi*4e-7,
#        'ofname': 'traj-hydrogen-3D.dump'
#        }

    """These are my updated parameters that cause an overload"""
    params = {
        'natoms': 10,
        'temp': 3e13,
        'mass': 27e-12,
        'radius': 50e-6,
        'initial_vels': 1e-6,
        'relax': 1e-6,
        'dt': 1e-8,
        'steps': 100000,
        'freq': 1000,
        'box': ((0, 1e-3), (0, 1e-3), (0, 1e-3)),
        'sepP': 7200,
        'sepF': 0.712e-6,
        'permFS': math.pi*4e-7,
        'ofname': 'traj-hydrogen-3D.dump'
        }
    output = run(**params)

    plt.plot(output[:,0], output[:,1])
    plt.xlabel('Time (ps)')
    plt.ylabel('Temp (K)')
    plt.show()
