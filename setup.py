from skbuild import setup
from setuptools_scm import get_version

project_name = "GMatTensor"

setup(
    name = project_name,
    description = "Tensor operations and unit tensors support GMat models",
    long_description = "Tensor operations and unit tensors support GMat models",
    version = get_version(),
    license = "MIT",
    author = "Tom de Geus",
    author_email = "tom@geus.me",
    url = f"https://github.com/tdegeus/{project_name}",
    packages = [f"{project_name}"],
    package_dir = {"": "python"},
    cmake_install_dir = f"python/{project_name}",
    cmake_minimum_required_version = "3.13...3.21",
)
