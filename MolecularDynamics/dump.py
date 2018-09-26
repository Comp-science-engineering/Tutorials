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

def writeOutput(filename, natoms, timestep, box, **args):
    """ Writes the output (in dump format) """
    
    axis = ('x', 'y', 'z')

    with open(filename, 'a') as fp:
 
         fp.write('ITEM: TIMESTEP\n')
         fp.write('{}\n'.format(timestep))
         fp.write('ITEM: NUMBER OF ATOMS\n')
         fp.write('{}\n'.format(natoms))
         fp.write('ITEM: BOX BOUNDS ff ff ff\n')

         for box_low, box_high in box:
            fp.write( '{} {}\n'.format(box_low,box_high) )
  
         keys = args.keys()

         for key in keys:
            dims = len(args[key].shape)

            if dims > 1:
                for i in range(args[key].shape[1]):
                    if key == 'pos':
                        args[axis[i]] = args[key][:,i]
                    else:
                        args['{}_{}'.format(key, axis[i])] = args[key][:,i]

                del args[key]

         keys = args.keys()
         fp.write('ITEM: ATOMS' + (' {}' * len(keys)).format(*keys) + '\n')
         
         output = []
         
         if args:
             for key in keys:
                 data = args[key]

                 if len(output):
                     output = np.vstack((output, data.T))
                 else:
                     output = data
         
         if len(output):
             np.savetxt(fp, output.T)