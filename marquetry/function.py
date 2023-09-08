import weakref

import numpy as np

import marquetry


def no_backprop_mode():
    """Make a context manager which disables back-propagation.

        In this context, Marquetry doesn't record the computation graph.
        :class:`marquetry.Variable` created in this context does not have
        reference to the :class:`marquetry.Function` which created the variable as creator.

        When you use this context, you can't do backpropagation but the memory will be released.

        In this example, ``y`` is created in this context. So you cannot call
        :func:`marquetry.Variable.backward`.

        Examples:
            >>> x = marquetry.Variable(np.array([1,], 'f'))
            >>> with marquetry.no_backprop_mode():
            ...   y = x + 1

    """
    return marquetry.using_config("enable_backprop", False)


def test_mode():
    """Make a context manager which train mode.

        In this context, Marquetry use test mode behavior if the function has different behavior in train and test mode.

        For example, :class:marquetry.functions.BatchNormalization store moving-mean and moving-variant in train mode,
        and it is used as normalize parameter in test mode.

        When you use this context, in the area is recognized as test mode.


        Examples:
            >>> x = marquetry.Variable(np.array([1,], 'f'))
            >>> with marquetry.test_mode():
            ...   y = x + 1

        """
    return marquetry.using_config("train", False)


# ===========================================================================
# Function
# ===========================================================================
class Function(object):
    """Function on variables with backpropagation ability.

        All function implementations defined in :mod:`marquetry.functions` inherit
        this class.

        The main feature of this class is recording the computation graph for back prop.
        When a function is applied to :class:`marquetry.Variable` objects,
        its :meth:`forward` method is called on :data:`Variable.data` fields of
        input variables, and at the same time it chains references from output
        variable nodes to the function and from the function to its input nodes.


        For functions that do not need a part of inputs in backward computation,
        there is a way to possibly reduce the memory consumption by quickly
        releasing the input arrays after the forward propagation. This is done by
        calling :meth:`retain_inputs` from inside :meth:`forward`.

        if you don't need the input data in backward computation, you should specify as :code: self.input_retain(()).
        The input data is remained and output data is released as a default.

        Attributes:
            generation (int): An identifier to track the generation of this function in the computation graph.
            inputs (tuple): A tuple of input variable nodes.
            outputs (tuple): A tuple of weak references to output variable nodes.
            output_data (tuple): A tuple of output data arrays.

    """

    generation = 0
    _input_indexes_to_retain = None
    _output_indexes_to_retain = None
    _output_retain_ever = None

    inputs = None
    outputs = None
    output_data = None

    def __call__(self, *inputs):
        inputs = [marquetry.as_variable(x) for x in inputs]

        xs = [x.data for x in inputs]

        xp = marquetry.cuda_backend.get_array_module(xs[0])

        ys = self.forward(*xs)
        if not isinstance(ys, tuple):
            ys = (ys,)
        outputs = [marquetry.Variable(marquetry.as_array(y, xp)) for y in ys]

        if marquetry.configuration.config.enable_backprop:
            self.generation = max([x.generation for x in inputs])
            for output in outputs:
                output.set_creator(self)

            self.inputs = tuple([x.node for x in inputs])
            self.outputs = tuple([weakref.ref(output.node) for output in outputs])

            input_indexes_to_retain = self._input_indexes_to_retain
            if input_indexes_to_retain is None:
                input_indexes_to_retain = range(len(inputs))
            for index in input_indexes_to_retain:
                inputs[index].retain_data()

            self._input_indexes_to_retain = None

            output_indexes_to_retain = self._output_indexes_to_retain
            if output_indexes_to_retain is not None:
                for index in output_indexes_to_retain:
                    outputs[index].retain_data()

            self._output_indexes_to_retain = None

        return outputs if len(outputs) > 1 else outputs[0]

    @property
    def name(self):
        """Get the name of the function class.

            Returns:
                str: The name of the function class.
        """
        return self.__class__.__name__

    def unchain(self):
        """Break references between this function and its input and output variable nodes."""
        for y in self.outputs:
            y_ref = y()
            if y_ref is not None:
                y_ref.unchain()

        self.inputs = None

    def forward(self, *xs):
        """Perform the forward computation of the function.

            Args:
                *xs: Input data arrays.

            Returns:
                marquetry.Variable: Output data arrays.

            Note:
                Generally, this class shouldn't be called by manually because `forward` is called by `__call__`.
        """
        raise NotImplementedError()

    def backward(self, *grad_ys):
        """Perform the backward computation of the function.

            Args:
                *grad_ys (:class:`marquetry.Variable`): Gradient data arrays.

            Returns:
                marquetry.Variable: Gradient data arrays with respect to the input data arrays.
        """

        raise NotImplementedError()

    def retain_inputs(self, indexes):
        """
            Specify which input data should be retained after forward propagation.

            Args:
                indexes (tuple or None): A list of indexes specifying which input data should be retained.
                    If None, all input data is retained.

        """
        self._input_indexes_to_retain = indexes

    def retain_outputs(self, indexes, retain_ever=False):
        """
            Specify which output data should be retained after forward propagation,
            and whether it should be retained forever.

            Args:
                indexes (tuple or None): A list of indexes specifying which output data should be retained.
                    If None, no output data is retained.
                retain_ever (bool): If True, the retained output data will be kept forever and not released.
                    If False, the retained output data will be released once it's no longer needed.

        """
        self._output_indexes_to_retain = indexes
        if retain_ever:
            self._output_retain_ever = retain_ever
