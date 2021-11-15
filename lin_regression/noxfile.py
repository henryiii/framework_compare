import nox

@nox.session
def purepython(session):
    session.install("numpy")
    session.run("python", "purepython.py")


@nox.session
def purepyjion(session):
    session.install("pyjion", "numpy")
    session.run("pyjion", "purepython.py", env={"PYTHONPATH": "."})


@nox.session(python="pypy3.7")
def purepypy(session):
    session.install("numpy")
    session.run("python", "purepython.py")


@nox.session
def purenumpy(session):
    session.install("numpy")
    session.run("python", "purenumpy.py")


@nox.session
def np_inv(session):
    session.install("numpy")
    session.run("python", "np_inv.py")


@nox.session
def np_pinv(session):
    session.install("numpy")
    session.run("python", "np_pinv.py")


@nox.session(python="3.9")
def purenumba(session):
    session.install("numpy", "numba")
    session.run("python", "purenumba.py")


@nox.session(python="3.9")
def vectornumba(session):
    session.install("numpy", "numba", "scipy")
    session.run("python", "vectornumba.py")
