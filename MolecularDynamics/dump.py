'''
Created on September 22, 2018
@author: Andrew Abi-Mansour
'''

# !/usr/bin/python
# -*- coding: utf8 -*-
# -------------------------------------------------------------------------
#
#   A simple molecular dynamics solver that simulates the motion
#   of non-interacting particles in the canonical ensemble using
#   a Langevin thermostat.
#
# --------------------------------------------------------------------------
#
#   This program is free software: you can redistribute it and/or modify
#   it under the terms of the GNU General Public License as published by
#   the Free Software Foundation, either version 2 of the License, or
#   (at your option) any later version.

#   This program is distributed in the hope that it will be useful,
#   but WITHOUT ANY WARRANTY; without even the implied warranty of
#   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#   GNU General Public License for more details.

#   You should have received a copy of the GNU General Public License
#   along with this program.  If not, see <http://www.gnu.org/licenses/>.

# -------------------------------------------------------------------------

import numpy as np

def writeOutput(filename, natoms, timestep, box, **data):
    """ Writes the output (in dump format) """
    
    axis = ('x', 'y', 'z')
    
    with open(filename, 'a') as fp:
        
        fp.write('ITEM: TIMESTEP\n')
        fp.write('{}\n'.format(timestep))
        
        fp.write('ITEM: NUMBER OF ATOMS\n')
        fp.write('{}\n'.format(natoms))
        
        fp.write('ITEM: BOX BOUNDS' + ' f' * len(box) + '\n')
        for box_bounds in box:
            fp.write('{} {}\n'.format(*box_bounds))

        for i in range(len(axis) - len(box)):
            fp.write('0 0\n')
            
        keys = list(data.keys())
        
        for key in keys:
            isMatrix = len(data[key].shape) > 1
            
            if isMatrix:
                _, nCols = data[key].shape
                
                for i in range(nCols):
                    if key == 'pos':
                        data['{}'.format(axis[i])] = data[key][:,i]
                    else:
                        data['{}_{}'.format(key,axis[i])] = data[key][:,i]
                        
                del data[key]
                
        keys = data.keys()
        
        fp.write('ITEM: ATOMS' + (' {}' * len(data)).format(*data) + '\n')
        
        output = []
        for key in keys:
            output = np.hstack((output, data[key]))
            
        if len(output):
            np.savetxt(fp, output.reshape((natoms, len(data)), order='F'))
