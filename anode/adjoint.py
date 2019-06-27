#*
# @file adjoint.py 
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
import torch
import torch.nn as nn
from . import odesolver
from torch.autograd import Variable

def flatten_params(params):
    flat_params = [p.contiguous().view(-1) for p in params]
    return torch.cat(flat_params) if len(flat_params) > 0 else torch.tensor([])

def flatten_params_grad(params, params_ref):
    _params = [p for p in params]
    _params_ref = [p for p in params_ref]
    flat_params = [p.contiguous().view(-1) if p is not None else torch.zeros_like(q).view(-1)
        for p, q in zip(_params, _params_ref)]

    return torch.cat(flat_params) if len(flat_params) > 0 else torch.tensor([])

class Checkpointing_Adjoint(torch.autograd.Function):

    @staticmethod
    def forward(ctx, *args):
        z0, func, flat_params, options= args[0], args[1], args[2], args[3]
        ctx.func = func

        with torch.no_grad():
            ans = odesolver(func, z0, options) 
        ctx.save_for_backward(z0)
        ctx.in1 = options
        return ans

    @staticmethod
    def backward(ctx, grad_output):

        z0 = ctx.saved_tensors
        options = ctx.in1
        func = ctx.func
        f_params = func.parameters()
        t = 0

        with torch.set_grad_enabled(True):
            z = Variable(z0[0].detach(),requires_grad=True)
            func_eval = odesolver(func, z, options) 
            out1 = torch.autograd.grad(
               func_eval,  z,
               grad_output, allow_unused=True, retain_graph=True)
            out2 = torch.autograd.grad(
               func_eval,  f_params,
               grad_output, allow_unused=True, retain_graph=True)

        return out1[0], None, flatten_params_grad(out2, func.parameters()), None



def odesolver_adjoint(func, z0, options = None):

    flat_params = flatten_params(func.parameters())
    zs = Checkpointing_Adjoint.apply(z0, func, flat_params, options)

    return zs
