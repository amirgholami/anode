#*
# @file odesolver.py 
# This file is part of ANODE library.
#
# ANODE is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# ANODE is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with ANODE.  If not, see <http://www.gnu.org/licenses/>.
#*
from .scheme import Euler, RK2, RK4


def odesolver(func, z0, options = None):
    if options == None:
        Nt = 2
    else:
        Nt = options['Nt']
    if (options['method'] == 'Euler'):
        solver = Euler(func, z0, Nt = Nt)
    elif (options['method'] == 'RK2'):
        solver = RK2(func, z0, Nt = Nt)
    elif (options['method'] == 'RK4'):
        solver = RK4(func, z0, Nt = Nt)
    else:
        print('error unsupported method passed')
        return
    z1 = solver.integrate(z0)

    return z1
