from setuptools import setup

setup(name='cmap',
      version='0.1.dev',
      description='Overlay parametric color maps on base images',
      author='Brendan Moloney',
      author_email='moloney@ohsu.edu',
      py_modules=['cmap'],
      install_requires=['numpy', 'nibabel', 'matplotlib', 'scipy'],
      entry_points = {'console_scripts' : ['cmap = cmap:main']},
     )
