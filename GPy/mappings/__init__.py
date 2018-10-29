# Copyright (c) 2013, 2014 GPy authors (see AUTHORS.txt).
# Licensed under the BSD 3-clause license (see LICENSE.txt)

from .kernel import Kernel
from .linear import Linear
from .mlp import MLP
from .mlpext import MLPext
from .additive import Additive
from .compound import Compound
from .constant import Constant
from .identity import Identity
from .piecewise_linear import PiecewiseLinear

from.kernel import Kernel_with_A
from .additive import Additive_with_active_dims