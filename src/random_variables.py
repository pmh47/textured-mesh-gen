
# This is based on the equivalent functionality in edward

import inspect
from tensorflow_probability import edward2
from tensorflow_probability.python.edward2.generated_random_variables import *
import distributions

_globals = globals()
for _name in sorted(dir(distributions)):
    _candidate = getattr(distributions, _name)
    if (
        inspect.isclass(_candidate) and
        issubclass(_candidate, distributions.Distribution) and
        _candidate.__module__ == 'distributions'  # i.e. not tensorflow.contrib.distributions!
    ):
        _globals[_name] = lambda *params, _candidate=_candidate, **kwargs: edward2.as_random_variable(_candidate(*params, **kwargs))

    del _candidate

