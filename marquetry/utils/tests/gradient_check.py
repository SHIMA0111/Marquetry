import marquetry
from marquetry import as_container
from marquetry import cuda_backend


def gradient_check(f, x, *args, rtol=1e-4, atol=1e-5, **kwargs):
    """Check the function backward gradient is correct or not by comparing the numerical gradient."""
    x = as_container(x)
    xp = cuda_backend.get_array_module(x)
    x.data = x.data.astype(xp.float64)

    num_grad = marquetry.utils.numerical_grad(f, x, *args, **kwargs)
    y = f(x, *args, **kwargs)
    y.backward()
    bp_grad = x.grad.data

    assert bp_grad.shape == num_grad.shape
    res = marquetry.utils.array_close(num_grad, bp_grad, rtol=rtol, atol=atol)

    grad_diff = xp.abs(bp_grad - num_grad).sum()
    if not res:
        print("")
        print("========== FAILED (Gradient Check) ==========")
        print("Back propagation for {} failed.".format(f.__class__.__name__))
        print("Grad Diff: {}".format(grad_diff))
        print("=============================================")
    else:
        print("")
        print("========== OK (Gradient Check) ==========")
        print("Grad Diff: {}".format(grad_diff))
        print("=============================================")

    return res
