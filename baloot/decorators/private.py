import sys
from functools import wraps


def private(func):
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        caller_frame = sys._getframe(1)
        if caller_frame.f_locals.get("self") is not self:
            raise PermissionError(
                f"Access denied: '{func.__name__}' is a private method and cannot be called from outside the class."
            )
        return func(self, *args, **kwargs)

    return wrapper
