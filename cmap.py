#! /usr/bin/env python
import sys, os, math, argparse, warnings
with warnings.catch_warnings():
    warnings.simplefilter('ignore')
    import nibabel as nb
    import numpy as np
    from matplotlib import pylab
    from matplotlib import gridspec
    from matplotlib.colorbar import ColorbarBase
    from matplotlib import colors
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    # TODO: Drop this dependency
    from scipy.ndimage import rotate


def check_dim(img):
    if img.ndim not in (2, 3):
        raise ValueError("Images must be 2D/3D arrays")


def robust_min_max(arr, iqr_coeff=1.5):
    '''Get min/max that ignores outliers

    Values are thresholed by the distance to the mean. The threshold is
    given by multiplying inter-quartile range by the `iqr_coeff`.
    '''
    first_quart, third_quart = np.percentile(arr, (25, 75))
    thresh = (third_quart - first_quart) * iqr_coeff
    mask = (arr > first_quart - thresh) & (arr < third_quart + thresh)
    incl_data = arr[mask]
    return np.min(incl_data), np.max(incl_data)


class AlphaMap(object):
    '''Map data to an alpha channel'''
    def __init__(self, alpha_range, data_range=None, 
                 alpha_bounds=None):
        if isinstance(alpha_range, int):
            alpha_range = float(alpha_range)
        if isinstance(alpha_range, float):
            self._alpha_range = (alpha_range, alpha_range)
        else:
            self._alpha_range = tuple(alpha_range)
            if len(self._alpha_range) != 2:
                raise ValueError("alpha_range must be scalar or two element tuple")
        self._data_range = data_range
        if self._data_range is None:
            self._data_range = (None, None)
        if alpha_bounds is None:
            self._alpha_bounds = (0.0, self._alpha_range[1])
        else:
            if len(alpha_bounds) != 2:
                raise ValueError("alpha_bounds must be two element tuple")
            self._alpha_bounds = alpha_bounds
        
    
    def apply(self, data):
        '''Return the alpha map for the given data'''
        min_data, max_data = self._data_range
        if min_data is None:
            min_data = np.min(data)
        if max_data is None:
            max_data = np.max(data)
        alpha = np.empty_like(data, dtype=np.float32)
        alpha[data < min_data] = self._alpha_bounds[0]
        alpha[data > max_data] = self._alpha_bounds[1]
        min_alpha, max_alpha = self._alpha_range
        in_range = (min_data <= data) & (data <= max_data)
        alpha[in_range] = (data[in_range] - min_data) / max_data
        alpha[in_range] *= max_alpha - min_alpha
        alpha[in_range] += min_alpha
        return alpha


class Overlay(object):
    '''Represent single data overlay'''
    def __init__(self, data, mask=None, cmap=None, cmap_bounds=None,
                 alpha_map=None):
        check_dim(data)
        self.data = np.atleast_3d(data)
        if mask is not None:
            check_dim(mask)
            self.mask = np.atleast_3d(mask)
        else:
            self.mask = np.ones_like(self.data)
        self.cmap = cmap
        if cmap_bounds is None:
            incl_data = self.data[self.mask == 1]
            cmap_bounds = robust_min_max(incl_data)
        self.cmap_bounds = cmap_bounds
        if alpha_map is None:
            self.alpha_map = AlphaMap(1.0, cmap_bounds)
        else:
            self.alpha_map = alpha_map
        
    def transpose(self, new_order):
        self.data = np.transpose(self.data, new_order)
        self.mask = np.transpose(self.mask, new_order)
    
    def get_rgba(self, index):
        d_sub = self.data[index].copy()
        alpha = self.alpha_map.apply(d_sub)
        alpha[self.mask[index] == 0] = 0.0
        alpha[alpha > 1] = 1.0
        alpha[alpha < 0] = 0.0
        cmap_lb, cmap_ub = self.cmap_bounds
        d_sub[d_sub > cmap_ub] = cmap_ub
        d_sub[d_sub < cmap_lb] = cmap_lb
        if self.cmap is None:
            cmap = getattr(pylab.cm, default_cmaps[0])
        else:
            cmap = getattr(pylab.cm, self.cmap)
        rgba = cmap((d_sub - cmap_lb) / cmap_ub)
        rgba[...,3] = alpha
        # matplotlib chokes on -0, so take absolute value
        return np.abs(rgba)
        


default_cmaps = ['plasma',
                 'virdis',
                 'copper',
                ]


def plot_slices(bg_img, overlays, slice_dim=None, title=None):
    if bg_img.ndim not in (2, 3):
        raise ValueError("Images must be 2D/3D arrays")
    bg_img = np.atleast_3d(bg_img)
    shape = bg_img.shape
    for overlay in overlays:
        if overlay.data.shape != shape:
            raise ValueError("Background / foreground image dimensions don't match")

    # Choose slice axis if none was specified
    if slice_dim is None:
        min_dim = np.argmin(shape)
        if shape[min_dim] < shape[2]:
            slice_dim = min_dim
        else:
            slice_dim = 2
    
    # TODO: Should determine slice orientation relative to the patient
    #       and then lay out rows / cols in radiological or neurological
    #       standard.  Instead of transposing, probably better to just 
    #       slice the array appropriately, so we don't need to update the
    #       affine
    # move slice dimension to the end
    new_order = tuple(ax for ax in range(3) if ax != slice_dim) + (slice_dim,)
    bg_img = np.transpose(bg_img, new_order)
    for overlay in overlays:
        overlay.transpose(new_order)

    # Compute plot grid size
    n_slices = bg_img.shape[2]
    n_cols = int(math.ceil(math.sqrt(n_slices)))
    n_rows = int(math.ceil(float(n_slices) / n_cols))
    
    # Compute display scaling factor
    v_max = np.mean(bg_img) * 4
    
    # Make sure each overlay has a color map, try to make them unique
    used_cmaps = [o.cmap for o in overlays if o.cmap is not None]
    for overlay in overlays:
        if overlay.cmap is None:
            for cmap in default_cmaps:
                if cmap not in used_cmaps:
                    overlay.cmap = cmap
                    break
            else:
                warnings.warn("Overlay colormaps collide")
                overlay.cmap = default_cmaps[0]
    
    # Build the matplotlib figure
    fig = pylab.figure()
    gs = gridspec.GridSpec(n_rows, n_cols)
    gs.update(wspace=0.05, hspace=0.05)
    ax_array = [fig.add_subplot(gs[i]) for i in range(n_slices)]
    for slice_idx in range(n_slices):
        ax = ax_array[slice_idx]
        if title is not None:
            #ax.set_title(title, fontsize=4)
            ax.text(.5, 1.03, title, fontsize=3,
                    horizontalalignment='center',
                    transform=ax.transAxes, color='k')
        ax.tick_params(axis='both',
                       which='both',
                       labelleft='off',
                       labelbottom='off',
                       bottom='off',
                       top='off',
                       left='off',
                       right='off')
        ax.set_axis_off()
        bg_slice = rotate(bg_img[...,slice_idx], 90)
        bg_plt = ax.imshow(bg_slice, cmap='gray', vmin=0, vmax=v_max)
        for overlay in overlays:
            fg_rgba = rotate(overlay.get_rgba((Ellipsis,slice_idx)), 90)
            fg_rgba = np.abs(fg_rgba)
            img = ax.imshow(fg_rgba)
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.05)
            cax.tick_params(labelsize=2, length=1, width=0.25, pad=1)
            cb = ColorbarBase(cax,
                              cmap=overlay.cmap,
                              norm=colors.Normalize(*overlay.cmap_bounds))
            cb.outline.set_linewidth(0.25)
    # Ironically this seems to "loosen" the layout and prevent 
    # overlapping of titles and images
    gs.tight_layout(fig)
    return fig


prog_descrip = \
'''Overlay one or more Niftis ontop of a base image using color maps'''

def main(argv=sys.argv):
    arg_parser = argparse.ArgumentParser(description=prog_descrip)
    arg_parser.add_argument('bg_img', help="Image to use as background")
    arg_parser.add_argument('fg_img', help="Image to overlay as color map")
    arg_parser.add_argument('--output', default='./cmap_overlay.pdf',
                            help="Output file (default: %(default)s)")
    arg_parser.add_argument('-t', '--title', default=None,
                            help="Title to display above each slice")
    arg_parser.add_argument('-s', '--slice-dim', default=None,
                            help="The dimension index to slice on")
    arg_parser.add_argument('-m', '--mask', default=None,
                            help="A mask to apply to the fg_img")
    arg_parser.add_argument('-c', '--color-map', default=None,
                            help="The name of the color map to use")
    arg_parser.add_argument('-d', '--data-range', default=None,
                            help='Range of data in fg_img to color')
    arg_parser.add_argument('-a', '--alpha', default=None,
                            help="Single alpha value comma separated "
                            "lower,upper values")
    arg_parser.add_argument('--alpha-range', default=None,
                            help="Range of data to map alpha values on")
    arg_parser.add_argument('--alpha-bounds', default=None,
                            help=("Alpha values to use for data values "
                                  "outside the mapped range"))
    args = arg_parser.parse_args()
    if args.slice_dim is not None:
        args.slice_dim = int(args.slice_dim)
    if args.data_range is not None:
        args.data_range = tuple(float(x) for x in args.data_range.split(','))
    if args.alpha is not None:
        args.alpha = tuple(float(x) for x in args.alpha.split(','))
        if len(args.alpha) == 1:
            args.alpha = args.alpha[0]
    if args.alpha_range is not None:
        args.alpha_range = tuple(float(x) for x in args.alpha_range.split(','))
    if args.alpha_bounds is not None:
        args.alpha_bounds = tuple(float(x) for x in args.alpha_bounds.split(','))
    
    bg_nii = nb.load(args.bg_img)
    bg_img = bg_nii.get_data()
    
    fg_nii = nb.load(args.fg_img)
    if args.mask:
        mask = nb.load(args.mask).get_data()
    else:
        mask = None
    if args.alpha is not None:
        alpha_map = AlphaMap(args.alpha, 
                             args.alpha_range, 
                             args.alpha_bounds)
    else:
        alpha_map = None
    overlay = Overlay(fg_nii.get_data(), 
                      mask, 
                      args.color_map, 
                      args.data_range,
                      alpha_map)
    
    fig = plot_slices(bg_img, [overlay], args.slice_dim, title=args.title)
    pylab.savefig(args.output, dpi=900)


if __name__ == '__main__':
    sys.exit(main())
