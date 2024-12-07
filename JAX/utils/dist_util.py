
import jax
import jax.numpy as jnp


def pmean(x, axis_name='batch'):
  """ all_reduce if pmap axis_name exist
  """
  frames = jax.core.thread_local_state.trace_state.axis_env
  for frame in frames:
    if frame.name == axis_name:
      return jax.lax.pmean(x, axis_name=axis_name)
  return x


def all_gather(x, axis_name='batch'):
  """ all_gather if pmap axis_name exist
  """
  frames = jax.core.thread_local_state.trace_state.axis_env
  for frame in frames:
    if frame.name == axis_name:
      return jax.lax.all_gather(x, axis_name=axis_name)
  return x


def SyncBatchNorm(x, eps=1.e-6):
  """ without gamma/beta
  """
  mean_x = jnp.mean(x, axis=0, keepdims=True)
  mean_x2 = jnp.mean(x**2, axis=0, keepdims=True)
  mean_x = pmean(mean_x, axis_name='batch')
  mean_x2 = pmean(mean_x2, axis_name='batch')
  var = mean_x2 - mean_x**2
  x = (x - mean_x) / (var + eps)**.5
  return x
