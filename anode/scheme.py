#*
# @file scheme.py 
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
from .time_stepper import Time_Stepper

class Euler(Time_Stepper):
    def step(self, func, t, dt, y):
        out = y + dt * func(t, y)
        return out

class RK2(Time_Stepper):
    def step(self, func, t, dt, y):
        k1 = dt * func(t, y)
        k2 = dt * func(t + dt / 2.0, y + 1.0 / 2.0 * k1)
        out = y + k2
        return out

class RK4(Time_Stepper):
    def step(self, func, t, dt, y):
        k1 = dt * func(t, y)
        k2 = dt * func(t + dt / 2.0, y + 1.0 / 2.0 * k1)
        k3 = dt * func(t + dt / 2.0, y + 1.0 / 2.0 * k2)
        k4 = dt * func(t + dt, y + k3)
        out = y + 1.0 / 6.0 * k1 + 1.0 / 3.0 * k2 + 1.0 / 3.0 * k3 + 1.0 / 6.0 * k4
        return out
