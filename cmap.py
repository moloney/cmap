#! /usr/bin/env python
import sys, math, argparse, warnings
from typing import Iterable, List, Optional, Tuple
with warnings.catch_warnings():
    warnings.simplefilter('ignore')
    import nibabel as nb
    import numpy as np
    from matplotlib import pylab
    from matplotlib import gridspec
    from matplotlib.colorbar import ColorbarBase
    from matplotlib import colors
    from matplotlib.backends.backend_pdf import PdfPages
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    # TODO: Drop this dependency
    from scipy.ndimage import rotate


def check_dim(img):
    if img.ndim not in (2, 3):
        raise ValueError("Images must be 2D/3D arrays")


def get_slice_dim(shape: Tuple[int, ...]) -> int:
    min_dim = np.argmin(shape)
    if shape[min_dim] < shape[2]:
        return min_dim
    else:
        return 2


def robust_min_max(arr, iqr_coeff=1.5):
    '''Get min/max that ignores outliers

    Values are thresholed by the distance to the mean. The threshold is
    given by multiplying inter-quartile range by the `iqr_coeff`.
    '''
    first_quart, third_quart = np.percentile(arr, (25, 75))
    thresh = (third_quart - first_quart) * iqr_coeff
    mask = (arr >= first_quart - thresh) & (arr <= third_quart + thresh)
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
            self._alpha_bounds = self._alpha_range
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
        alpha[in_range] = (data[in_range] - min_data) / (max_data - min_data)
        alpha[in_range] *= max_alpha - min_alpha
        alpha[in_range] += min_alpha
        return alpha


class Overlay(object):
    '''Overlay of scalar image passed through color map / alpha map'''

    def __init__(self, data, mask=None, cmap=None, cmap_bounds=None,
                 alpha_map=None, interpolation=None):
        check_dim(data)
        self.data = np.atleast_3d(data)
        if mask is not None:
            check_dim(mask)
            self.mask = np.atleast_3d(mask).copy()
        else:
            self.mask = np.ones_like(self.data)
        self.cmap = cmap
        if cmap_bounds is None:
            if self.mask.sum() == 0:
                cmap_bounds = (0.0, 0.0)
            else:
                cmap_bounds = robust_min_max(self.data[self.mask == 1])
        self.cmap_bounds = cmap_bounds
        if alpha_map is None:
            self.alpha_map = AlphaMap(1.0, cmap_bounds)
        else:
            self.alpha_map = alpha_map
        self.interpolation = interpolation
        
    def transpose(self, new_order):
        self.data = np.transpose(self.data, new_order)
        self.mask = np.transpose(self.mask, new_order)
        
    def is_empty(self, index):
        return np.count_nonzero(self.mask[index]) == 0
    
    def get_rgba(self, index):
        d_sub = self.data[index].copy()
        alpha = self.alpha_map.apply(d_sub)
        alpha[self.mask[index] == 0] = 0.0
        alpha[alpha > 1] = 1.0
        alpha[alpha < 0] = 0.0
        cmap_lb, cmap_ub = self.cmap_bounds
        d_sub[d_sub > cmap_ub] = cmap_ub
        d_sub[d_sub < cmap_lb] = cmap_lb
        d_sub -= cmap_lb
        d_range = (cmap_ub - cmap_lb)
        if d_range:
            d_sub /= d_range
        if self.cmap is None:
            cmap = getattr(pylab.cm, default_cmaps[0])
        else:
            cmap = getattr(pylab.cm, self.cmap)
        rgba = cmap(d_sub)
        del d_sub
        rgba[...,3] = alpha
        del alpha
        # matplotlib chokes on -0, so take absolute value
        return np.abs(rgba)
    
    def plot_color_hist(self, index, ax, hist_kwargs=None, scale=1.0):
        if hist_kwargs is None:
            hist_kwargs = {}
        if 'bins' not in hist_kwargs:
            hist_kwargs['bins'] = 64
        hist_kwargs['range'] = self.cmap_bounds
        cmap_lb, cmap_ub = self.cmap_bounds
        if self.cmap is None:
            cmap = getattr(pylab.cm, default_cmaps[0])
        else:
            cmap = getattr(pylab.cm, self.cmap)
        d_sub = self.data[index].copy()
        # We want our histogram to reflect what is visible in the 
        # overlay, so we need to consider the alpha map
        alpha = self.alpha_map.apply(d_sub)
        alpha[self.mask[index] == 0] = 0.0
        alpha[alpha > 1] = 1.0
        alpha[alpha < 0] = 0.0
        incl_data = d_sub[alpha != 0.0]
        if incl_data.size == 0:
            return
        incl_data[incl_data < cmap_lb] = cmap_lb
        incl_data[incl_data > cmap_ub] = cmap_ub
        n, bins, patches = ax.hist(incl_data, **hist_kwargs)
        bin_centers = 0.5 * (bins[:-1] + bins[1:])
        col = bin_centers - cmap_lb
        d_range = cmap_ub - cmap_lb
        if d_range:
            col /= d_range
        for c, p in zip(col, patches):
            pylab.setp(p, 'facecolor', cmap(c))
            pylab.setp(p, 'edgecolor', 'k')
            pylab.setp(p, 'linewidth', 0.5*scale)
    
    def plot(self, data_index, ax, histo_bins=50, scale=1.0):
        fg_rgba = rotate(self.get_rgba(data_index), 90)
        img = ax.imshow(fg_rgba, interpolation=self.interpolation)
        divider = make_axes_locatable(ax)
        if histo_bins:
            hax = divider.append_axes("right", size="15%", pad=0.01*scale)
            hax.set_axis_off()
            hax.margins(x=0.01, y=0.001)
            self.plot_color_hist(data_index, 
                                 hax, 
                                 hist_kwargs={'orientation' : 'horizontal',
                                              'align' : 'left',
                                              'density' : 1,
                                              'bins' : histo_bins,
                                             },
                                 scale=scale,
                                )
            hax.invert_xaxis()
            cax_pad = 0.001
        else:
            cax_pad = 0.1 * scale
        cax = divider.append_axes("right", size="5%", pad=cax_pad)
        cax.tick_params(labelsize=10*scale, length=7*scale, width=1.5*scale, pad=5*scale)
        cb = ColorbarBase(cax,
                          cmap=self.cmap,
                          norm=colors.Normalize(*self.cmap_bounds))
        cb.outline.set_linewidth(1.0*scale)


default_cmaps = ['plasma',
                 'virdis',
                 'copper',
                ]
'''Default selection of colormaps are all perceptually linear'''


def check_shapes(bg_img, overlays):
    """Check background / overlays have same shape"""
    if bg_img.ndim not in (2, 3):
        raise ValueError("Images must be 2D/3D arrays")
    bg_img = np.atleast_3d(bg_img)
    shape = bg_img.shape
    for overlay in overlays:
        if overlay.data.shape != shape:
            raise ValueError("Background / foreground image dimensions don't match")
    return bg_img, shape


def get_slice_multi_idx(slice_idx, slice_dim):
    """Get tuple for indexing slice of data"""
    slice_multi_idx = [slice(None) for _ in range(3)]
    slice_multi_idx[slice_dim] = slice_idx
    return tuple(slice_multi_idx)


def sample_slices(
    imgs: List[np.ndarray], 
    slice_dim: Optional[int] = None, 
    max_slices: Optional[int] = None, 
    exclude_empty: bool = False, 
    weight_by_val: bool = False
):
    """Choose slices from `img` that fit the contraints"""
    n_dims = 3
    img = imgs[0]
    if slice_dim is None:
        slice_dim = get_slice_dim(img.shape)
    if len(img.shape) < 2:
        raise ValueError("The img must be 2D+")
    if any(i.shape != img.shape for i in imgs[1:]):
        raise ValueError("All input images must have same shape")
    if n_dims == 2 or img.shape[slice_dim] == 1:
        return [0]
    n_slices = img.shape[slice_dim]
    if (max_slices is None or max_slices >= n_slices) and not exclude_empty:
        return list(range(n_slices))
    slice_densities = np.zeros(n_slices)
    for img in imgs:
        for slice_idx in range(n_slices):
            slice_multi_idx = get_slice_multi_idx(slice_idx, slice_dim)
            if weight_by_val:
                slice_densities[slice_idx] += img[slice_multi_idx].sum() 
            else:
                slice_densities[slice_idx] += (img[slice_multi_idx] != 0).sum()
    nz_slices = slice_densities != 0
    if max_slices is None or nz_slices.sum() <= max_slices:
        return [i for i, nz in enumerate(nz_slices) if nz]
    slice_spread = (nz_slices.sum() // max_slices) // 2
    res = []
    potential_mask = nz_slices.copy()
    while len(res) < max_slices and potential_mask.sum():
        for slice_idx in range(n_slices):
            if not potential_mask[slice_idx]:
                slice_densities[slice_idx] = 0.0
        max_idx = np.argmax(slice_densities)
        res.append(max_idx)
        potential_mask[max_idx] = False
        # TODO: Would be better to do some weighting here instead of hard cutoff, allow
        #       for slice_spread to be fractional and thus do better when max_slices is
        #       just over 'n_slices / 2'
        for n in range(slice_spread):
            if max_idx + n < n_slices:
                potential_mask[max_idx + n] = False
            if max_idx - n > 0:
                potential_mask[max_idx - n] = False
    res.sort()
    return res


def gen_slice_plots(
    bg_img: np.ndarray, 
    overlays: Iterable[Overlay], 
    slice_dim: Optional[int] = None, 
    bg_range: Optional[Tuple[float, float]] = None,
    title: Optional[str] = None, 
    exclude_empty: bool = False, 
    loc_info: bool = False, 
    bg_interpolation: str = 'bilinear', 
    histo_bins: int = 64, 
    max_slice_per_fig: Optional[int] = None, 
    max_slices: Optional[int] = None,
):
    '''Generate one or more image overlay figures with one or more slices per figure
    '''
    bg_img, shape = check_shapes(bg_img, overlays)
    # Choose slice axis if none was specified
    if slice_dim is None:
        slice_dim = get_slice_dim(shape)
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
    slice_dim = 2
    n_slices = bg_img.shape[slice_dim]
    # Determine which slices to include
    incl_slices = sample_slices(
        [o.data for o in overlays], slice_dim, max_slices, exclude_empty
    )
    n_incl_slices = len(incl_slices)
    # Compute plot grid size
    if max_slice_per_fig is not None:
        n_fig = n_incl_slices // max_slice_per_fig
        if n_incl_slices % max_slice_per_fig != 0:
            n_fig += 1
        slice_per_fig = min(n_incl_slices, max_slice_per_fig)
    else:
        n_fig = 1
        slice_per_fig = n_incl_slices
    n_cols = int(math.ceil(math.sqrt(slice_per_fig)))
    n_rows = int(math.ceil(float(slice_per_fig) / n_cols))
    if slice_per_fig > 1:
        scale = 1.0 / math.sqrt(slice_per_fig)
    else:
        scale = 1.0
    # Compute display scaling factor
    # TODO: Need something better here
    if bg_range is None:
        bg_min, bg_max = robust_min_max(bg_img)
        bg_range = (bg_min / 2, bg_max)
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
    curr_slice_idx = 0
    for fig_idx in range(n_fig):
        fig = pylab.figure()
        fig_slices = []
        if slice_per_fig > 1:
            gs = gridspec.GridSpec(n_rows, n_cols)
            gs.update(
                left=0.02, 
                right=.98, 
                bottom=0.02, 
                top=0.98, 
                wspace=0.8*scale, 
                hspace=0.1
            )
            ax_array = [fig.add_subplot(gs[i]) for i in range(slice_per_fig)]
        else:
            ax_array = [fig.gca()]
        for slice_fig_idx in range(slice_per_fig):
            slice_idx = incl_slices[curr_slice_idx]
            fig_slices.append(slice_idx)
            ax = ax_array[slice_fig_idx]
            ax.set_axis_off()
            ax.margins(x=0.02*scale, y=0.02*scale)
            d_idx = (Ellipsis, slice_idx)
            if title is not None:
                ax.text(.5, 1.01, title, fontsize=16*scale,
                        horizontalalignment='center',
                        transform=ax.transAxes, color='k')
            if loc_info:
                loc_text = 'Slice %d' % slice_idx
                ax.text(.5, -0.035, loc_text, fontsize=11*scale,
                        horizontalalignment='center',
                        transform=ax.transAxes, color='k')
            bg_slice = rotate(bg_img[d_idx], 90)
            bg_plt = ax.imshow(bg_slice, 
                               cmap='gray', 
                               vmin=bg_range[0], 
                               vmax=bg_range[1],
                               interpolation=bg_interpolation)
            for overlay in overlays:
                overlay.plot(d_idx, ax, histo_bins, scale=scale)
            curr_slice_idx += 1
            if curr_slice_idx == len(incl_slices):
                break
        yield fig_slices, fig, scale


prog_descrip = \
'''Create high quality and highly quantitative colormap overlays from Nifti images'''


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
    arg_parser.add_argument('-b', '--bg-range', default=None,
                            help="Give lower,upper bounds for grayscale "
                            "background")
    arg_parser.add_argument('--bg-interp', default='bilinear',
                            help="Interpolation mode for bg image")
    arg_parser.add_argument('--fg-interp', default='bilinear',
                            help='Interpolation mode for fg image')
    arg_parser.add_argument('-e', '--exclude-empty', action='store_true',
                            help="Exclude slices with no masked voxels")
    arg_parser.add_argument('-l', '--location', action='store_true',
                            help="Add slice location under each image")
    arg_parser.add_argument('--alpha-range', default=None,
                            help="Range of data to map alpha values on")
    arg_parser.add_argument('--alpha-bounds', default=None,
                            help=("Alpha values to use for data values "
                                  "outside the mapped range"))
    arg_parser.add_argument('--histo-bins', default=64, type=int,
                            help=("Number of bins for histogram-"
                                  "colorbar, set to zero to disable"))
    arg_parser.add_argument('--slices-per-fig', default=None, 
                            help="Split up slices onto multiple figures")
    arg_parser.add_argument("--multi-page", action="store_true",
                            help="Create a single multi-page output file instead of "
                            "one per figure. Only works with PDF output")
    arg_parser.add_argument('--dpi', default=300, type=int,
                            help='Resolution of output image. We scale '
                            'this by sqrt of # slices in figure.')
    args = arg_parser.parse_args()
    if args.multi_page:
        if not args.output.endswith(".pdf"):
            print("The '--multi-page' option only works with PDF output")
            return 1
        mp_pdf = PdfPages(args.output)
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
    if args.bg_range is not None:
        args.bg_range = tuple(float(x) for x in args.bg_range.split(','))
    if args.slices_per_fig is not None:
        args.slices_per_fig = int(args.slices_per_fig)  
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
                      alpha_map,
                      args.fg_interp)
    
    
    figs = gen_slice_plots(bg_img, 
                           [overlay], 
                           slice_dim=args.slice_dim, 
                           bg_range=args.bg_range,
                           title=args.title, 
                           exclude_empty=args.exclude_empty,
                           loc_info=args.location,
                           histo_bins=args.histo_bins,
                           max_slice_per_fig=args.slices_per_fig)
    for fig_idx, (slice_indices, fig, scale) in enumerate(figs):
        fig_dpi = int(args.dpi / scale)
        if args.slices_per_fig is None:
            assert fig_idx == 0
            fig.savefig(args.output, dpi=fig_dpi)
        elif args.multi_page:
            mp_pdf.savefig(fig, dpi=fig_dpi)
        else:
            toks = args.output.split('.')
            base = '.'.join(toks[:-1])
            ext = toks[-1]
            #full_out = base + ('-%03d' % fig_idx) + '.' + ext
            if args.slices_per_fig == 1:
                assert len(slice_indices) == 1
                full_out = '%s_slice_%03d.%s' % (base, slice_indices[0], ext)
            else:
                full_out = '%s_fig_%03d.%s' % (base, fig_idx, ext)
            fig.savefig(full_out, dpi=fig_dpi)
        fig.clear()
        pylab.close(fig)

if __name__ == '__main__':
    sys.exit(main())
