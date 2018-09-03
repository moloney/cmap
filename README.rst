====
cmap
====

Generates color overlay from parametric map ontop of a base image, on 
a slice by slice basis.  Expects Nifti input, though it may work with
other imaging formats (if supported by nibabel).

Basic CLI Usage
===============

There are just two required inputs, the base image and the parametric 
map to overlay.

::

    $ cmap base.nii param_map.nii

By default the output file `cmap_overlay.pdf` will be created in the 
current working directory. You can specify a different output path 
and/or file type using the `--output` option.

::

    $ cmap base.nii param_map.nii --output overlays/param.svg

If the default min/max values for the color mapping aren't to your 
liking you can override them with the `--data-range` (`-d`) option. 
Say that we want our color map to cover the data range between `0.01` 
and `0.5`.

::

    $ cmap base.nii param_map.nii -d 0.01,0.5

Values below the minimum will be made transparent by default so that
you can see the base image. Another common case is supplying a mask 
so that the parameter overlay will only be visible within a certain 
region of interest.

::

    $ cmap base.nii param_map.nii -m my_roi.nii

Use the `--help` option to see other more advanced options.
