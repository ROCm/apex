import glob
import os
import site


SITE_PACKAGES_FOLDERS = site.getsitepackages()[0]

#count the number of *.so files in the folder
so_files = glob.glob(os.path.join(SITE_PACKAGES_FOLDERS, "**/*.so"), recursive=True)
count = len(so_files)
print(count)
