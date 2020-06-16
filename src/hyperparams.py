
import sys

class Hyperparams:

    def __init__(self):

        self._hyperparameter_to_value = {}
        for arg in sys.argv[1:]:
            name, value = arg.split('=')
            self._hyperparameter_to_value[name] = value
        self._requested_hyperparameters = set()

    def __call__(self, default, name, converter=float):

        # Retrieve the value of the given parameter specified on the command-line, or return default
        if name in self._hyperparameter_to_value:
            value = converter(self._hyperparameter_to_value[name])
            use_default = False
        else:
            value = default
            use_default = True
        if name not in self._requested_hyperparameters:
            print('hyper: {} = {}{}'.format(name, value, ' (default)' if use_default else ''))
            self._requested_hyperparameters.add(name)
        return value

    def verify_args(self):

        # Check that no hyperparameters were given on the command-line, that were not also requested through hyper(...)
        missing = []
        for name in self._hyperparameter_to_value:
            if name not in self._requested_hyperparameters:
                missing.append(name)
        if len(missing) > 0:
            raise RuntimeError('the following unrecognised hyperparameters were specified: ' + ', '.join(missing))


