import math

from marquetry.model import Model


# ===========================================================================
# Optimizer base class
# ===========================================================================
class Optimizer(object):
    """Base class for all optimizers.

        The Optimizer class provides basic functionality for all optimizers
        used to adjust model parameters to fit user-provided data.
        To use an optimizer, a target model must be registered using the 'prepare' method,
        and the 'update' method is called to update the model's parameters based on a loss function
        and the optimization method defined in each specific optimizer class.

        Optimizer objects also support the use of hooks, which are registered using the 'add_hook' method.
        Hooks are functions that are called in advance of the actual parameter update process.

        Attributes:
            target: The target model to optimize, registered via the 'prepare' method.
            additional_hooks: A list of additional hooks to be executed before the parameter optimization process.
    """
    def __init__(self):
        self.target = None
        self.additional_hooks = []

    def prepare(self, target):
        """Set a target model to optimize the model parameters.

            Args:
                target (Model): Model which you want to optimize the parameters.

            Returns:
                Optimizer: Optimizer object.
        """
        self.target = target

        return self

    def update(self):
        """Updates the parameters.

            When user optimize the model parameter, user call this method.
        """
        params = [p for p in self.target.params() if p.grad is not None]

        for f in self.additional_hooks:
            f(params)

        for param in params:
            self._update_one(param)

    def _update_one(self, param):
        """Perform the update for a single parameter (to be implemented by subclasses).

            In the subclass, the optimize mechanism is implemented in this method.

            Args:
                param(marquetry.container.Parameter): The parameter to update.
        """

        raise NotImplementedError()

    def add_hook(self, hook):
        """Add hooks to be executed in advance of parameter optimization.

            If you want to add hooks in advance the optimizing model parameters, you can add hooks via this method.

            Args:
                hook: Hook function you want to execute in advance of the optimizing.
        """
        self.additional_hooks.append(hook)


# ===========================================================================
# Hooks
# ===========================================================================
class WeightDecay(object):
    """Optimizer hook function for weight decay regularization.

        This hook function adds a scaled parameter to the corresponding gradient,
        which can be used as a form of regularization to prevent overfitting.

        Args:
            rate (float):  Coefficient for weight decay regularization.
    """

    def __init__(self, rate):
        self.rate = rate

    def __call__(self, params):
        for param in params:
            param.grad.data += self.rate * param.data


class ClipGrad(object):
    """Optimizer hook function for gradient clipping.

        This hook function scales the gradient array to a user-defined L2 norm threshold.
        Gradient clipping is often used to prevent exploding gradients during training.

        Args:
            max_norm (float): L2 norm threshold for gradient clipping.
    """

    def __init__(self, max_norm):
        self.max_norm = max_norm

    def __call__(self, params):
        total_norm = 0
        for param in params:
            total_norm += (param.grad.data ** 2).sum()

        total_norm = math.sqrt(float(total_norm))

        rate = self.max_norm / (total_norm + 1e-15)

        if rate < 1:
            for param in params:
                param.grad.data *= rate
