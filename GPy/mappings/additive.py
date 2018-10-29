# Copyright (c) 2013, GPy authors (see AUTHORS.txt).
# Licensed under the BSD 3-clause license (see LICENSE.txt)

import numpy as np
from ..core import Mapping
from ..core import Param
from .kernel import Kernel, Kernel_with_A
class Additive(Mapping):
    """
    Mapping based on adding two existing mappings together.

    .. math::

       f(\mathbf{x}*) = f_1(\mathbf{x}*) + f_2(\mathbf(x)*)

    :param mapping1: first mapping to add together.
    :type mapping1: GPy.mappings.Mapping
    :param mapping2: second mapping to add together.
    :type mapping2: GPy.mappings.Mapping

    """

    def __init__(self, mapping1, mapping2):
        assert(mapping1.input_dim==mapping2.input_dim)
        assert(mapping1.output_dim==mapping2.output_dim)
        input_dim, output_dim = mapping1.input_dim, mapping1.output_dim
        super(Additive, self).__init__(input_dim=input_dim, output_dim=output_dim)
        self.mapping1 = mapping1
        self.mapping2 = mapping2
        self.link_parameters(self.mapping1, self.mapping2)

    def f(self, X):
        return self.mapping1.f(X) + self.mapping2.f(X)

    def update_gradients(self, dL_dF, X):
        self.mapping1.update_gradients(dL_dF, X)
        self.mapping2.update_gradients(dL_dF, X)

    def gradients_X(self, dL_dF, X):
        return self.mapping1.gradients_X(dL_dF, X) + self.mapping2.gradients_X(dL_dF, X)

class Additive_with_active_dims(Mapping):
    def __init__(self, mapping1, mapping2, active_dims1, active_dims2):
        assert(mapping1.input_dim==len(active_dims1))
        assert(mapping2.input_dim==len(active_dims2))

        assert(mapping1.output_dim==mapping2.output_dim)
        input_dim, output_dim = mapping1.input_dim + mapping2.input_dim, mapping1.output_dim
        super().__init__(input_dim=input_dim, output_dim=output_dim)
        self.mapping1 = mapping1
        self.mapping2 = mapping2
        # self.kernel_cond = isinstance(mapping1, Kernel) and isinstance(mapping1, Kernel)
        # if self.kernel_cond :
        #     print("act_dims1: ", active_dims1)
        #     print("act_dims2: ", active_dims2)
        #     assert active_dims1 == mapping1.kern.active_dims, "Inconsistent kernel active_dims of mapping1"
        #     assert active_dims2 == mapping2.kern.active_dims, "Inconsistent kernel active_dims of mapping2" 
        self.active_dims1 = active_dims1
        self.active_dims2 = active_dims2
        
        self.link_parameters(self.mapping1, self.mapping2)

    def _activate_dims(self, X):
        # X1 = np.take(X, self.active_dims1, axis=1)
        # X2 = np.take(X, self.active_dims2, axis=1)
        # if self.kernel_cond:
        #     return X, X
        X1 = X[:, self.active_dims1]
        X2 = X[:, self.active_dims2]
        return X1, X2

    def f(self, X):
        X1, X2 = self._activate_dims(X)
        return self.mapping1.f(X1) + self.mapping2.f(X2)

    def update_gradients(self, dL_dF, X):
        X1, X2 = self._activate_dims(X)
        self.mapping1.update_gradients(dL_dF, X1)
        self.mapping2.update_gradients(dL_dF, X2)

    def gradients_X(self, dL_dF, X):
        X1, X2 = self._activate_dims(X)
        dL_dX1 = self.mapping1.gradients_X(dL_dF, X1)
        dL_dX2 = self.mapping2.gradients_X(dL_dF, X2)
        # if self.kernel_cond:
        #     return dL_dX1 + dL_dX2 
        return np.hstack([dL_dX1, dL_dX2])
