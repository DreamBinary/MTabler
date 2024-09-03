import os

unitable_src = "/bohr/unii-7sxm/v1/unitable/src"
os.system(f"cp -r {unitable_src} .")
pkgs_path = "/bohr/pkgs-7x29/v18/pkgs"

os.system(f"pip install {pkgs_path}/numpy*.whl")
os.system(f"pip install {pkgs_path}/numba*.whl")