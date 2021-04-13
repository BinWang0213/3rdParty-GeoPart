import dolfin


def create_timer(name):
    return dolfin.Timer(f"geopart: {name}")


def apply_dolfin_timer(meth):
    name = f"{meth.__module__}.{meth.__qualname__}"

    def timed_method(*args, **kwargs):
        with create_timer(name):
            return meth(*args, **kwargs)

    return timed_method
