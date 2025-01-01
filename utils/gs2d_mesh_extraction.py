import add_pypath
import os
if "OMP_NUM_THREADS" not in os.environ:
    os.environ["OMP_NUM_THREADS"] = "4"
from internal.entrypoints.gs2d_mesh_extraction import main


if __name__ == "__main__":
    main()
