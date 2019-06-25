#*
# @file time_stepper.py 
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
import abc
import torch
import copy
import numpy as np


class Time_Stepper(object):
    __metaclass__ = abc.ABCMeta

    def __init__(self, func, y0, Nt = 2 ):
        self.func = func
        self.Nt = Nt

    @abc.abstractmethod
    def step(self, func, t, dt, y):
        pass

    def integrate(self, y0):
        y1 = y0
        dt = 1. / float(self.Nt)
        for n in range(self.Nt):
            t0 = 0 + n * dt
            y1 = self.step(self.func, t0, dt, y1)

        return y1

