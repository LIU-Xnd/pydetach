"""pydetach: Detaching spatial Trascriptomics into Annotated Cells with High confidence"""

__author__ = "Xindong Liu"
__version__ = "1.0.2"

from .utils import (
    chunk_spatial,
    reinit_index,
    read_from_csv,
    write_to_csv,
)
from .data import (
    binX,
    annotate_mt,
    annotate_ribosomal,
    merge_gene_version,
    scale_genes,
    downsample_cells,
    sort_by_coords,
)
from .classifier import (
    SVM,
)
from .spatial import (
    rw_aggregate,
    # rw_aggregate_sequential,
    extract_celltypes_full,
    extract_cell_sizes_full,
    cluster_spatial_domain,
    spatial_distances,
    # SpatialTypeAnnCntMtx,
    # celltype_refined_bin,
    ctrbin_cellseg,
    ctrbin_cellseg_parallel,
    SpTypeSizeAnnCntMtx,
    aggregate_spots_to_cells,
    aggregate_spots_to_cells_parallel,
    NucleiMasks,
    vonoroi_indices,
    align_coords,
    transfer_label,
)
from .plot import (
    rcParams_setupStr,
    pie,
    plot_stacked_barplot,
    plot_stacked_barplot_circular,
    SpAnnPoints,
    get_boundaries,
    plot_boundaries,
    plot_boundaries_on_grids,
)

from . import recipe
