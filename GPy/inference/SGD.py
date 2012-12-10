import numpy as np
from optimization import Optimizer
from scipy import linalg, optimize
import copy
import sys

class opt_SGD(Optimizer):
    """
    Optimize using stochastic gradient descent.

    *** Parameters ***
    model: reference to the model object
    iterations: number of iterations
    learning_rate: learning rate
    momentum: momentum

    """

    def __init__(self, start, f_fp, f, fp, iterations = 10, learning_rate = 1e-4, momentum = 0.9, model = None, messages = False, batch_size = 1, self_paced = False, **kwargs):
        self.opt_name = "Stochastic Gradient Descent"

        self.f = f
        self.fp = fp
        self.f_fp = f_fp
        self.model = model
        self.iterations = iterations
        self.momentum = momentum
        self.learning_rate = learning_rate
        self.x_opt = None
        self.f_opt = None
        self.messages = messages
        self.batch_size = batch_size
        self.self_paced = self_paced

        num_params = len(self.model.get_param())
        if isinstance(self.learning_rate, float):
            self.learning_rate = np.ones((num_params,)) * self.learning_rate

        assert (len(self.learning_rate) == num_params), "there must be one learning rate per parameter"

    def __str__(self):
        status = "\nOptimizer: \t\t\t %s\n" % self.opt_name
        status += "f(x_opt): \t\t\t %.4f\n" % self.f_opt
        status += "Number of iterations: \t\t %d\n" % self.iterations
        status += "Learning rate: \t\t\t max %.3f, min %.3f\n" % (self.learning_rate.max(), self.learning_rate.min())
        status += "Momentum: \t\t\t %.3f\n" % self.momentum
        status += "Batch size: \t\t\t %d\n" % self.batch_size
        status += "Time elapsed: \t\t\t %s\n" % self.time
        return status

    def non_null_samples(self, data):
        return np.isnan(data).sum(axis=1) == 0

    def check_for_missing(self, data):
        return np.isnan(data).sum() > 0

    def subset_parameter_vector(self, x, samples, param_shapes):
        subset = np.array([], dtype = int)
        x = np.arange(0, len(x))
        i = 0

        for s in param_shapes:
            N, Q = s
            X = x[i:N*Q].reshape(N, Q)
            X = X[samples]
            subset = np.append(subset, X.flatten())
            i += N*Q

        subset = np.append(subset, x[i:])

        return subset

    def shift_constraints(self, j):
        # back them up
        bounded_i = copy.deepcopy(self.model.constrained_bounded_indices)
        bounded_l = copy.deepcopy(self.model.constrained_bounded_lowers)
        bounded_u = copy.deepcopy(self.model.constrained_bounded_uppers)

        for b in range(len(bounded_i)): # for each group of constraints
            for bc in range(len(bounded_i[b])):
                pos = np.where(j == bounded_i[b][bc])[0]
                if len(pos) == 1:
                    pos2 = np.where(self.model.constrained_bounded_indices[b] == bounded_i[b][bc])[0][0]
                    self.model.constrained_bounded_indices[b][pos2] = pos[0]
                else:
                    if len(self.model.constrained_bounded_indices[b]) == 1:
                        # if it's the last index to be removed
                        # the logic here is just a mess. If we remove the last one, then all the
                        # b-indices change and we have to iterate through everything to find our
                        # current index. Can't deal with this right now.
                        raise NotImplementedError

                    else: # just remove it from the indices
                        mask = self.model.constrained_bounded_indices[b] != bc
                        self.model.constrained_bounded_indices[b] = self.model.constrained_bounded_indices[b][mask]


        positive = self.model.constrained_positive_indices.copy()
        for p in range(len(positive)):
            pos = np.where(j == self.model.constrained_positive_indices[p])[0][0]
            self.model.constrained_positive_indices[p] = pos

        return (bounded_i, bounded_l, bounded_u), positive

    def restore_constraints(self, b, p):
        self.model.constrained_bounded_indices = b[0]
        self.model.constrained_bounded_lowers = b[1]
        self.model.constrained_bounded_uppers = b[2]
        self.model.constrained_positive_indices = p

    def get_param_shapes(self, N = None, Q = None):
        model_name = self.model.__class__.__name__
        if model_name == 'GPLVM':
            return [(N, Q)]
        else:
            raise NotImplementedError

    def step_with_missing_data(self, X, Y, step, shapes):
        N, Q = X.shape
        samples = self.non_null_samples(self.model.Y)
        j = self.subset_parameter_vector(self.x_opt, samples, shapes)
        self.model.N = samples.sum()
        self.model.X = X[samples]
        self.model.Y = self.model.Y[samples]

        if self.model.N == 0:
            return 0, step, self.model.N

        b,p = self.shift_constraints(j)

        momentum_term = self.momentum * step[j]
        f, fp = self.f_fp(self.x_opt[j])
        step[j] = self.learning_rate[j] * fp
        self.x_opt[j] -= step[j] + momentum_term

        self.restore_constraints(b, p)

        return f, step, self.model.N

    def opt(self):
        self.x_opt = self.model.get_param()
        X, Y = self.model.X.copy(), self.model.Y.copy()
        N, Q = self.model.X.shape
        D = self.model.Y.shape[1]
        self.trace = []
        missing_data = self.check_for_missing(self.model.Y)
        self.model.Youter = None # this is probably not very efficient

        for it in range(self.iterations):
            if it == 0 or self.self_paced is False:
                features = np.random.permutation(Y.shape[1])
            else:
                features = np.argsort(LL)[::-1]

            b = len(features)/self.batch_size
            features = [features[i::b] for i in range(b)]
            step = np.zeros_like(self.model.get_param())
            LL = []
            count = 0

            for j in features:
                count += 1
                self.model.D = len(j)
                self.model.Y = Y[:, j]

                if missing_data:
                    shapes = self.get_param_shapes(N, Q)
                    f, step, Nj = self.step_with_missing_data(X, Y, step, shapes)
                else:
                    Nj = N
                    momentum_term = self.momentum * step # compute momentum using update(t-1)
                    f, fp = self.f_fp(self.x_opt)
                    step = self.learning_rate * fp # compute update(t)
                    self.x_opt -= step + momentum_term

                if self.messages == 2:
                    status = "evaluating {feature: 5d}/{tot: 5d} \t f: {f: 2.3f} \t non-missing: {nm: 4d}\r".format(feature = count, tot = len(features), f = -f, nm = Nj)
                    sys.stdout.write(status)
                    sys.stdout.flush()

                LL.append(-f)

            # should really be a sum(), but earlier samples in the iteration will have a very crappy ll
            self.f_opt = np.mean(LL)
            self.model.N = N
            self.model.Y = Y
            self.model.X = X
            self.model.D = D
            # self.model.Youter = np.dot(Y, Y.T)
            self.trace.append(self.f_opt)
            if self.messages != 0:
                sys.stdout.write('\r' + ' '*len(status)*2 + '  \r')
                status = "SGD Iteration: {0: 3d}/{1: 3d}  f: {2: 2.3f}\n".format(it, self.iterations, self.f_opt)
                sys.stdout.write(status)
                sys.stdout.flush()
