====
cmap
====

Generates color overlay from parametric map ontop of a base image, on 
a slice by slice basis.  Expects Nifti input, though it may work with
other imaging formats (if supported by nibabel).

Basic Usage
===========

There are just two required inputs, the base image and the parametric 
map to overlay.

::

    $ cmap path/to/base.nii path/to/param_map.nii

By default the output file `cmap_overlay.pdf` will be created in the 
current working directory. You can specify a different output path 
and/or file type using the `--output` option.

::

   $ cmap path/to/base.nii path/to/param_map.nii --output param_overlay.svg

Use the `--help` option to see other more advanced options.
