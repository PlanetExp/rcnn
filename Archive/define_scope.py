""" Custom decorators

These decorators enable lazy instanciating of the tensorflow function
variable_scope()

Usage:
@define_scope
	is equvalent of using with variable_scope with its function as name

@define_scope(name, keywords)
	set a custom name or pass keyword argumets to the variable_scope
	function
"""
from functools import wraps  # custom decorators
import tensorflow as tf


def doublewrap(function):
	"""
    A decorator decorator, allowing to use the decorator to be used without
    parentheses if not arguments are provided. All arguments must be optional.

    Source: https://gist.github.com/danijar/8663d3bbfd586bffecf6a0094cd116f2
    """
	@wraps(function)
	def decorator(*args, **kwargs):
		if len(args) == 1 and len(kwargs) == 0 and callable(args[0]):
			return function(args[0])
		else:
			return lambda wrapee: function(wrapee, *args, **kwargs)
	return decorator


@doublewrap
def define_scope(function, scope=None, *args, **kwargs):
	"""
    A decorator for functions that define TensorFlow operations. The wrapped
    function will only be executed once. Subsequent calls to it will directly
    return the result so that operations are added to the graph only once.
    
    The operations added by the function live within a tf.variable_scope(). If
    this decorator is used with arguments, they will be forwarded to the
    variable scope. The scope name defaults to the name of the wrapped
    function.

    To sum up, the decorator works as a @property decorator with extra
    functionality of lazy loading the function and forwarding parameters to the
    tensorflow variable_scope.

    Source: https://gist.github.com/danijar/8663d3bbfd586bffecf6a0094cd116f2
    """
	attribute = '_cache_' + function.__name__
	name = scope or function.__name__

	@property
	@wraps(function)
	def decorator(self):
		if not hasattr(self, attribute):
			with tf.variable_scope(name, *args, **kwargs):  # pylint: disable=undefined-variable
				setattr(self, attribute, function(self))
		return getattr(self, attribute)

	return decorator
