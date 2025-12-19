from setuptools import setup, Extension
import numpy

module = Extension(
    "minimax_agent_c",
    sources=["minimax_agent_c.c"],
    include_dirs=[numpy.get_include()],
)

setup(
    name="minimax_agent_c",
    version="1.0",
    description="C implementation of minimax agent for Take5",
    ext_modules=[module],
)
