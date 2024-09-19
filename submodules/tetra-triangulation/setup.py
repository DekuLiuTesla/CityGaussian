import os
import re
import subprocess
import sys
from distutils.cmd import Command
from pathlib import Path

from setuptools import Extension, setup, find_packages

# Convert distutils Windows platform specifiers to CMake -A arguments
PLAT_TO_CMAKE = {
    "win32": "Win32",
    "win-amd64": "x64",
    "win-arm32": "ARM",
    "win-arm64": "ARM64",
}


# A CMakeExtension needs a sourcedir instead of a file list.
# The name must be the _single_ output extension from the CMake build.
# If you need multiple extensions, see scikit-build.
class CMakeExtension(Extension):
    def __init__(self, name: str, sourcedir: str = "") -> None:
        super().__init__(name, sources=[])
        self.sourcedir = os.fspath(Path(sourcedir).resolve())


class CMakeBuild(Command):
    user_options = [("debug", "g", "compile/link with debugging information")]

    boolean_options = ["debug"]

    def initialize_options(self):
        """Set default values for all the options that this command
        supports.  Note that these defaults may be overridden by other
        commands, by the setup script, by config files, or by the
        command-line.  Thus, this is not the place to code dependencies
        between options; generally, 'initialize_options()' implementations
        are just a bunch of "self.foo = None" assignments.
        """
        self.debug = None

    def finalize_options(self):
        """Set final values for all the options that this command supports.
        This is always called as late as possible, ie.  after any option
        assignments from the command-line or from other commands have been
        done.  Thus, this is the place to code option dependencies: if
        'foo' depends on 'bar', then it is safe to set 'foo' from 'bar' as
        long as 'foo' still has the same value it was assigned in
        'initialize_options()'.
        """
        self.set_undefined_options("build", ("debug", "debug"))

    def run(self):
        """A command's raison d'etre: carry out the action it exists to
        perform, controlled by the options initialized in
        'initialize_options()', customized by other commands, the setup
        script, the command-line, and config files, and finalized in
        'finalize_options()'.  All terminal output and filesystem
        interaction should be done by 'run()'.
        """
        for ext in self.distribution.ext_modules:
            self.build_extension(ext)

    def build_extension(self, ext) -> None:
        # Using this requires trailing slash for auto-detection & inclusion of
        # auxiliary "native" libs

        debug = int(os.environ.get("DEBUG", 0)) if self.debug is None else self.debug
        cfg = "Debug" if debug else "Release"

        # Set Python_EXECUTABLE instead if you use PYBIND11_FINDPYTHON
        # EXAMPLE_VERSION_INFO shows you how to pass a value into the C++ code
        # from Python.
        cmake_args = [
            f"-DPYTHON_EXECUTABLE={sys.executable}",
            f"-DCMAKE_BUILD_TYPE={cfg}",  # not used on MSVC, but no harm
        ]
        build_args = []
        # Adding CMake arguments set as environment variable
        # (needed e.g. to build for ARM OSx on conda-forge)
        if "CMAKE_ARGS" in os.environ:
            cmake_args += [item for item in os.environ["CMAKE_ARGS"].split(" ") if item]

        # In this example, we pass in the version to C++. You might not need to.
        cmake_args += [f"-DEXAMPLE_VERSION_INFO={self.distribution.get_version()}"]  # type: ignore[attr-defined]

        # # CMake lets you override the generator - we need to check this.
        # # Can be set with Conda-Build, for example.
        # cmake_generator = os.environ.get("CMAKE_GENERATOR", "")

        if sys.platform.startswith("darwin"):
            # Cross-compile support for macOS - respect ARCHFLAGS if set
            archs = re.findall(r"-arch (\S+)", os.environ.get("ARCHFLAGS", ""))
            if archs:
                cmake_args += ["-DCMAKE_OSX_ARCHITECTURES={}".format(";".join(archs))]

        # Set CMAKE_BUILD_PARALLEL_LEVEL to control the parallel build level
        # across all generators.
        if "CMAKE_BUILD_PARALLEL_LEVEL" not in os.environ:
            # self.parallel is a Python 3 only way to set parallel jobs by hand
            # using -j in the build_ext call, not supported by pip or PyPA-build.
            if hasattr(self, "parallel") and self.parallel:
                # CMake 3.12+ only.
                build_args += [f"-j{self.parallel}"]

        subprocess.run(["cmake", ext.sourcedir, *cmake_args], cwd=ext.sourcedir, check=True)
        subprocess.run(["cmake", "--build", ".", *build_args], cwd=ext.sourcedir, check=True)

    def get_source_files(self):
        file_list = []
        for ext in self.distribution.ext_modules:
            path = Path(ext.sourcedir)
            file_list.append(str(path / "CMakeLists.txt"))
            file_list.extend(map(str, path.glob("src/**")))
            file_list.extend(map(str, path.glob("cmake/**")))
        return file_list


# The information here can also be placed in setup.cfg - better separation of
# logic and declaration, and simpler if you include description/version in a file.
setup(
    name="tetra-nerf",
    version="0.1.1",
    author="Jonas Kulhanek",
    author_email="jonas.kulhanek@live.com",
    description="Official implementation of Tetra-NeRF paper",
    long_description="",
    # ext_modules=[CMakeExtension("cmake_example")],
    # cmdclass={"build_ext": CMakeBuild},
    zip_safe=False,
    python_requires=">=3.7",
    install_requires=["trimesh>=3.20.2"],
    packages=find_packages(),
    package_data={"tetranerf.utils.extension": ["py.typed", "**/*.pyi"]},
)
