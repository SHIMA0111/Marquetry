import weakref

import marquetry


def no_backprop_mode():
    return marquetry.using_config("enable_backprop", False)


def test_mode():
    return marquetry.using_config("train", False)


class Function(object):
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
        return self.__class__.__name__

    def unchain(self):
        for y in self.outputs:
            y_ref = y()
            if y_ref is not None:
                y_ref.unchain()

        self.inputs = None

    def forward(self, *xs):
        raise NotImplementedError()

    def backward(self, *grad_ys):
        raise NotImplementedError()

    def retain_inputs(self, indexes):
        self._input_indexes_to_retain = indexes

    def retain_outputs(self, indexes, retain_ever=False):
        self._output_indexes_to_retain = indexes
        if retain_ever:
            self._output_retain_ever = retain_ever
