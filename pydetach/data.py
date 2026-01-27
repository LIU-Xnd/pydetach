from .types import (
    _AnnData,
    _csr_matrix,
    _Literal,
    _Iterable,
)
from collections import defaultdict as _defaultdict

import scanpy as _sc
import pandas as _pd
import numpy as _np

from scipy.sparse import hstack as _hstack


from tqdm import tqdm as _tqdm

from collections import Counter as _Counter

def downsample_cells(
    adata: _AnnData,
    n_samples: int = 10000,
    colname_celltype: str = "cell_type",
    reindex: bool = True,
) -> tuple[_AnnData, _Counter]:
    """Downsample a fraction of cells from adata.

    Return Tuple of (result_adata, Counter of downsampled cell-types)"""
    ids_sampled = _np.random.choice(
        _np.arange(adata.shape[0]), size=n_samples, replace=False
    )
    result = adata[ids_sampled, :].copy()
    if reindex:
        result.obs.index = _np.arange(result.shape[0]).astype(str)
    counter = _Counter(result.obs[colname_celltype].astype(str).values)
    return (result, counter)


def binX(
    adata: _sc.AnnData,
    binsize: int = 48,
    obsm_name_spatial_coords: str = 'spatial',
    key_added: str | None = None,
    verbose: bool = False,
) -> _sc.AnnData:
    """
    Bin a spatial trx anndata.

    Args:
        adata (AnnData): spatial trx.

        binsize (int): size of bin.

        obsm_name_spatial_coords (str): name of key in obsm of spatial coordinates.

        key_added (str | None): an auxiliary obs column will be added to adata, with name
        defaulting to 'spatial_binX' (when this parameter set to None) where X is the binsize.
    
    Return:
        AnnData: binned anndata with spatial coordinates saved in obsm with the same name as adata.
    """
    assert isinstance(binsize, int)
    if key_added is None:
        key_added = f'spatial_bin{binsize}'
    adata.obsm[key_added] = adata.obsm[obsm_name_spatial_coords] // binsize
    # Build index
    index_loc2id = dict()
    if verbose:
        itor_ = _tqdm(range(adata.shape[0]),desc='Building index')
    else:
        itor_ = range(adata.shape[0])
    for i in itor_:
        loc = tuple(adata.obsm[key_added][i,:])
        if loc not in index_loc2id:
            index_loc2id[loc] = [i]
        else:
            index_loc2id[loc].append(i)
    n_new = len(index_loc2id)
    cols_ = []
    rows_ = []
    if verbose:
        itor_ = enumerate(_tqdm(index_loc2id, desc='Binning'))
    else:
        itor_ = enumerate(index_loc2id)
    for i_new, loc in itor_:
        cols_.extend(index_loc2id[loc])
        rows_.extend([i_new for _ in range(len(index_loc2id[loc]))])
        
    W_leftmul = _csr_matrix(
        (
            _np.ones(adata.shape[0]),
            (
                rows_,
                cols_,
            ),
        ),
        shape=(n_new, adata.shape[0]),
    )
    X_new = (W_leftmul @ adata.X).astype(adata.X.dtype)
    coords_new = _np.array(tuple(index_loc2id.keys()))
    return _sc.AnnData(
        X=X_new,
        var=_pd.DataFrame(index=adata.var.index),
        obsm={obsm_name_spatial_coords: coords_new},
    )

def annotate_mt(
    adata: _AnnData,
    startswith: str | _Iterable[str] = "MT-",
    key_added: str = 'mt',
) -> None:
    """
    Annotate mitochondrial genes in an AnnData object.

    Args:
        adata (_AnnData): The AnnData object to process.
        startswith (str): The prefix of mitochondrial genes to remove. Defaults to "MT-".
        key_added (str): The key under which to store the annotation in `adata.var`.
            Defaults to 'mt'.
    """
    assert isinstance(adata, _AnnData)
    if isinstance(startswith, str):
        startswith = [startswith]
    startswith = tuple(startswith)
    mt_genes = adata.var.index.str.startswith(startswith)
    adata.var[key_added] = mt_genes
    if mt_genes.sum() == 0:
        _tqdm.write(f"Warning: No mitochondrial genes found starting with '{startswith}'!")
    else:
        _tqdm.write(f"Annotated {mt_genes.sum()} mitochondrial genes starting with '{startswith}' in adata.var['{key_added}'].")
    return

def annotate_ribosomal(
    adata: _AnnData,
    startswith: str | _Iterable[str] = ("RPL", "RPS"),
    key_added: str = 'ribosomal',
) -> None:
    """
    Annotate ribosomal genes in an AnnData object.

    Args:
        adata (_AnnData): The AnnData object to process.
        startswith (str | Iterable[str]): The prefix of ribosomal genes to remove.
            Defaults to ("RPL", "RPS").
        key_added (str): The key under which to store the annotation in `adata.var`.
            Defaults to 'ribosomal'.
    """
    assert isinstance(adata, _AnnData)
    if isinstance(startswith, str):
        startswith = [startswith]
    ribosomal_genes = adata.var.index.str.startswith(tuple(startswith))
    adata.var[key_added] = ribosomal_genes
    if ribosomal_genes.sum() == 0:
        _tqdm.write(f"Warning: No ribosomal genes found starting with {startswith}!")
    else:
        _tqdm.write(f"Annotated {ribosomal_genes.sum()} ribosomal genes starting with {startswith} in adata.var['{key_added}'].")
    return

def annotate_uncharacterized_genes(
    adata: _AnnData,
    startswith: str | _Iterable[str] = ("RPL", "RPS", "MT-", "AC", "AL", "LOC"),
    key_added: str = 'uncharacterized',
) -> None:
    """
    Annotate uncharacterized genes in an AnnData object.

    Args:
        adata (_AnnData): The AnnData object to process.
        startswith (str | Iterable[str]): The prefix of uncharacterized genes to remove.
            Defaults to ("RPL", "RPS", "MT-", "AC", "AL", "LOC").
        key_added (str): The key under which to store the annotation in `adata.var`.
            Defaults to 'uncharacterized'.
    """
    assert isinstance(adata, _AnnData)
    if isinstance(startswith, str):
        startswith = [startswith]
    bad_genes = adata.var.index.str.startswith(tuple(startswith))
    adata.var[key_added] = bad_genes
    if bad_genes.sum() == 0:
        _tqdm.write(f"Warning: No genes found starting with {startswith}!")
    else:
        _tqdm.write(f"Annotated {bad_genes.sum()} ribosomal genes starting with {startswith} in adata.var['{key_added}'].")
    return

def merge_gene_version(
    adata: _AnnData,
    version_sep: str = ".",
) -> _AnnData:
    """
    Merge gene versions in an AnnData object. Keep the maximum counts among versions of gene.
    Note that .layers and .var will be empty.

    Args:
        adata (_AnnData): The AnnData object to process.
        version_sep (str): The separator used to split gene names and versions. Defaults to ".".

    Returns:
        _AnnData: A new AnnData object with merged gene versions.
    """
    
    var_names = adata.var_names
    base_names = var_names.str.split(version_sep).str[0]

    # Build indices of the same base gene
    gene_to_indices = _defaultdict(list)
    for idx, name in enumerate(base_names):
        gene_to_indices[name].append(idx)
    
    # Prepare for building new matrix
    merged_columns = []
    new_var_names = []

    _tqdm.write('Converting to column-efficient mode..')
    X_old = adata.X.copy().tocsc()

    for base_name, idxs in _tqdm(gene_to_indices.items()):
        if len(idxs) == 1:
            col = X_old[:, idxs[0]]
        else:
            cols = X_old[:, idxs].tocsc()
            col = cols.max(axis=1)
        merged_columns.append(col)
        new_var_names.append(base_name)
    
    # Stack
    X_new = _csr_matrix(_hstack(merged_columns, format='csr'))
    
    adata_merged = _AnnData(
        X=X_new,
        obs=adata.obs.copy(),
        var=_pd.DataFrame(index=new_var_names),
        obsm=adata.obsm.copy(),
        obsp=adata.obsp.copy(),
        uns=adata.uns.copy(),
    )

    return adata_merged


def scale_genes(
    adata: _AnnData,
) -> None:
    """Scale each gene in adata.X so that the maximum value per gene is 1.

    Modifies adata.X in place.

    Args:
        adata (_AnnData): AnnData object with expression matrix in adata.X
    """
    # First check if there's zero gene
    X = adata.X.copy().tocsc()

    max_per_gene = X.max(axis=0).toarray()
    whr_zero = (max_per_gene==0)
    if whr_zero.sum() > 0:
        _tqdm.write('Warning: exist genes with 0 counts!')
        max_per_gene[whr_zero] = 1
    X = _csr_matrix(
        X.multiply(
            1.0 / max_per_gene
        )
    )
    adata.X = X
    return


def sort_by_coords(
    anndata: _AnnData,
    major_axis: _Literal['x', 'y', 0, 1] = 'x',
) -> _AnnData:
    """
    major_axis: if 'x' or 0, sort by x first; if 'y' or 1, sort by y first.

    NOTE:
        This function DOES NOT re-init obs indices after operation.
    """
    assert 'spatial' in anndata.obsm_keys()
    assert major_axis in ['x', 'y', 0, 1]
    keys = [
        anndata.obsm['spatial'][:,1],
        anndata.obsm['spatial'][:,0],
    ]
    if major_axis in ['y', 1]:
        keys = [keys[1], keys[0]]
    
    ix_sort = _np.lexsort(keys=keys)
    return anndata[ix_sort].copy()
