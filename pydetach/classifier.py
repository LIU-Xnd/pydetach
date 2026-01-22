from .types import (
    _AnnData,
    _NDArray,
    _1DArrayType,
    _NumberType,
    _Iterable,
    _csr_matrix,
    _UndefinedType,
    _UNDEFINED,
)
from typing import Any
from sklearn.svm import LinearSVC as _SVC
from sklearn.calibration import CalibratedClassifierCV as _CalibratedClassifierCV

import numpy as _np
from scipy.sparse import issparse as _issparse
from scipy.sparse.linalg import svds as _svds
from .utils import rearrange_count_matrix as _rearrange_count_matrix


# >>> ---- Local Classifier ----
class _LocalClassifier:
    """This classifier would predict probabilities for each class

    .fit(), .predict(), and .predict_proba() are specially built, but
     often the last two methods are not to be called manually.

    Predicted integer i indicates the class self._classes[i].

    Needs overwriting.

    Args
    ----------
    threshold_confidence : float, default=0.75
        Confidence according to which whether the classifier predicts a sample
        to be in a class."""

    def __init__(
        self,
        threshold_confidence: float = 0.75,
        log1p: bool = True,
        normalize: bool = True,
        target_sum: _NumberType = 1e3,
        on_PCs: bool = False,
        n_PCs: int = 30,
        **kwargs,
    ):
        self._threshold_confidence: float = threshold_confidence
        self._genes: _NDArray | _UndefinedType = _UNDEFINED
        self._classes: _NDArray| _UndefinedType = _UNDEFINED
        self._normalize: bool = normalize
        self._target_sum: _NumberType = target_sum
        self._log1p: bool = log1p
        self._PC_loadings: _NDArray | _UndefinedType = _UNDEFINED
        self._n_PCs: int = n_PCs
        self._on_PCs: bool = on_PCs
        return None

    @property
    def threshold_confidence(self) -> float:
        return self._threshold_confidence

    def set_threshold_confidence(self, value: float = 0.75):
        self._threshold_confidence = value
        return self

    @property
    def genes(self) -> _NDArray | _UndefinedType:
        return self._genes.copy()

    @property
    def classes(self) -> _NDArray | _UndefinedType:
        return self._classes.copy()

    def classId_to_className(self, class_id: int) -> str:
        """Returns self._classes[class_id]."""
        assert isinstance(self._classes, _np.ndarray)
        return self._classes[class_id]

    def className_to_classId(self, class_name: str) -> int:
        """Returns the index where `class_name` is in self._classes."""
        return _np.where(self._classes == class_name)[0][0]

    def classIds_to_classNames(self, class_ids: _Iterable[int]) -> _NDArray:
        return _np.array(
            list(map(lambda cid: self.classId_to_className(cid), class_ids))
        )

    def classNames_to_classIds(self, class_names: _Iterable[str]) -> _NDArray:
        return _np.array(
            list(map(lambda cnm: self.className_to_classId(cnm), class_names))
        )

    def fit(
        self,
        sn_adata: _AnnData,
        colname_classes: str = "cell_type",
        to_dense: bool = False,
    ) -> dict | Any:
        """Trains the local classifier using the AnnData (h5ad) format
        snRNA-seq data.

        Args:
            sn_adata (AnnData): snRNA-seq h5ad data. Must have attributes:
             .X, the sample-by-gene count matrix;
             .obs['cell_type'] or named otherwise, that indicates the label of
             each sample;
             .var, whose index indicates the genes used for training.

            colname_classes (str): the name of the column in .obs that
             indicates the cell types (classes).

        Return:
            dict(X: 2darray, y: array): data ready to train.

        Return (overwritten):
            self (Model): a trained model (self).
        """
        assert _np.all(
            sn_adata.obs.index.astype(_np.int64) == _np.arange(sn_adata.shape[0])
        ), "sn_adata needs tidying using reinit_index()"
        self._genes = _np.array(sn_adata.var.index)
        if self._genes.shape[0] > 10_000:
            print(
                f"Warning: genes exceed 10,000, might encounter memory issue. You might want to filter genes first."
            )
        self._classes = _np.sort(
            _np.array((sn_adata.obs[colname_classes]).astype('category').cat.categories).astype(str)
        )  # sorted alphabetically
        # Prepare y: convert classNames into classIds
        class_ids: _1DArrayType = self.classNames_to_classIds(
            _np.array(sn_adata.obs[colname_classes].astype(str))
        )
        X_train: _NDArray | _csr_matrix = sn_adata.X.copy()
        if isinstance(X_train, _csr_matrix) and to_dense:
            X_train = X_train.toarray()

        return dict(X=X_train, y=class_ids)

    def predict_proba(
        self,
        X: _NDArray | _csr_matrix,
        genes: _Iterable[str] | None = None,
        to_dense: bool = False,
    ) -> dict | Any:
        """Predicts the probabilities for each
        sample to be of each class.

        Args:
            X (_NDArray | _csr_matrix): input count matrix.

            genes (_Iterable[str] | None): list of genes corresponding to
             X's columns. If None, set to pretrained snRNA-seq's gene list.

        Return:
            dict(X: 2darray): X ready to be predictors.

        Return (overwritten):
            2darray[float]: probs of falling into each class;
             each row is a sample and each column is a class.

        Needs overwriting."""

        assert len(X.shape) == 2, "X must be a sample-by-gene matrix"
        assert isinstance(self._genes, _Iterable)
        genes_: list[str] = []
        if genes is None:
            genes_ = list(self._genes)
        else:
            assert isinstance(genes, _Iterable)
        genes_ = _np.array(genes)
        assert len(genes_) == X.shape[1], "genes must be compatible with X.shape[1]"
        # Select those genes that appear in self._genes
        X_new: _csr_matrix = _rearrange_count_matrix(X, genes_, self._genes)
        if to_dense:
            X_new = X_new.toarray()
        return {"X": X_new}

    def predict(
        self,
        X: _csr_matrix | _NDArray,
        genes: _Iterable[str] | None = None,
    ) -> _NDArray:
        """Predicts classes of each sample. For example, if prediction is i,
         then the predicted class is self.classes[i].
         For those below confidence threshold,
         predicted classes are set to -1.
        You can set .threshold_confidence by
        .set_threshold_confidence().

        Args:
            X (_NDArray | _csr_matrix): input count matrix.

            genes (_Iterable[str] | None): list of genes corresponding to
             those of X's columns. If None, set to pretrained gene list.

        Return:
            _NDArray: an array of predicted classIds.

        Needs overwriting."""

        probas: _NDArray | dict[str, _NDArray] = self.predict_proba(X, genes)
        assert isinstance(probas, _NDArray), ".predict_proba() needs overwriting!"
        classes_pred = _np.argmax(probas, axis=1)
        probas_max = probas[_np.arange(probas.shape[0]), classes_pred]
        where_notConfident = probas_max < self.threshold_confidence
        classes_pred[where_notConfident] = -1
        return classes_pred


# ---- Local Classifier ---- <<<


class SVM(_LocalClassifier):
    """Based on sklearn.svm.linearSVC (See relevant reference there),
     specially built for snRNA-seq data training.
    This classifier would predict probabilities for each class
    An OVR (One-versus-Rest) strategy is used.

    .fit(), .predict(), and .predict_proba() are specially built, but
     often the last two methods are not to be called manually.

    Predicted integer i indicates the class self._classes[i].

    Note: this is the same model as in K. Benjamin's TopACT (
    https://gitlab.com/kfbenjamin/topact) classifier SVCClassifier.

    Args
    ----------
    threshold_confidence : float, default=0.75
        Confidence according to which whether the classifier predicts a sample
        to be in a class.

    C : float, default=1.0
        Regularization parameter. The strength of the regularization is
        inversely proportional to C. Must be strictly positive. The penalty
        is a squared l2 penalty. For an intuitive visualization of the effects
        of scaling the regularization parameter C, see
        :ref:`sphx_glr_auto_examples_svm_plot_svm_scale_c.py`.

    tol : float, default=1e-4
        Tolerance for stopping criterion.

    random_state : int, RandomState instance or None, default=None
        Controls the pseudo random number generation for shuffling the data for
        probability estimates. Ignored when `probability` is False.
        Pass an int for reproducible output across multiple function calls.
        See :term:`Glossary <random_state>`."""

    def __init__(
        self,
        threshold_confidence: float = 0.75,
        log1p: bool = True,
        normalize: bool = True,
        target_sum: _NumberType = 1e3,
        on_PCs: bool = False,
        n_PCs: int = 30,
        C: float = 1.0,
        tol: float = 1e-4,
        random_state: int | None = None,
        **kwargs,
    ):
        _model = _SVC(
            C=C,
            tol=tol,
            random_state=random_state,
            dual=False,
            **kwargs,
        )
        self._model = _CalibratedClassifierCV(_model)
        self._normalize: bool = normalize
        self._target_sum: _NumberType = target_sum
        self._log1p: bool = log1p
        self._PC_loadings: _NDArray | _UndefinedType = _UNDEFINED
        self._n_PCs: int = n_PCs
        self._on_PCs: bool = on_PCs
        return super().__init__(threshold_confidence=threshold_confidence)

    def fit(
        self,
        sn_adata: _AnnData,
        colname_classes: str = "cell_type",
        to_dense: bool = False,
    ):
        """Trains the local classifier using the AnnData (h5ad) format
        snRNA-seq data.

        Args:
            sn_adata (AnnData): snRNA-seq h5ad data. Must have attributes:
             .X, the sample-by-gene count matrix;
             .obs['cell_type'] or named otherwise, that indicates the label
              of each sample;
             .var, whose index indicates the genes used for training.

            colname_classes (str): the name of the column in .obs that
             indicates the cell types (classes).

        Return:
            self (model)."""

        X_y_ready: dict = super().fit(
            sn_adata=sn_adata,
            colname_classes=colname_classes,
            to_dense=to_dense,
        )
        X_ready: _csr_matrix | _NDArray = X_y_ready["X"]
        if self._normalize:
            if _issparse(X_ready):
                # Normalize using sparse-safe operations
                row_sums = X_ready.sum(
                    axis=1
                ).A1  # .A1 to convert the sum to a 1D array
                row_inv = 1.0 / _np.maximum(row_sums, 1e-8)  # Avoid division by zero
                X_ready = _csr_matrix.multiply(X_ready, row_inv[:, _np.newaxis]) * self._target_sum
            else:
                X_ready = self._target_sum * _np.divide(
                    X_ready, _np.maximum(_np.sum(X_ready, axis=1).reshape(-1, 1), 1e-8)
                )
        if self._log1p:
            if _issparse(X_ready):
                X_ready.data = _np.log1p(X_ready.data)
            else:
                X_ready = _np.log1p(X_ready)
        # Get reference PC loadings
        if _issparse(X_ready):
            _, _, Vh = _svds(X_ready, k=self._n_PCs)
            self._PC_loadings = _np.real(Vh)
        else:
            # Non-centered PCA
            self._PC_loadings = _np.real(  # in case it is complex, but barely
                _np.linalg.svd(
                    a=X_ready, full_matrices=False, compute_uv=True, hermitian=False
                ).Vh  # the loading matrix, PC by gene
            )[: self._n_PCs, :]
        if self._on_PCs:
            X_ready = X_ready @ self._PC_loadings.T
        self._model.fit(X=X_ready, y=X_y_ready["y"])
        return self

    def predict_proba(
        self,
        X: _csr_matrix | _NDArray,
        genes: _Iterable[str] | None = None,
    ) -> _NDArray:
        """Predicts the probabilities for each
         sample to be of each class.

        Args:
            X (_NDArray | _csr_matrix): input count matrix.

            genes (_Iterable[str] | None): list of genes corresponding to
             X's columns. If None, set to pretrained snRNA-seq's gene list.

        Return:
            2darray[float]: probs of falling into each class;
             each row is a sample and each column is a class."""

        X_ready: _csr_matrix | _NDArray = super().predict_proba(X, genes)["X"]
        if self._normalize:
            if _issparse(X_ready):
                # Normalize using sparse-safe operations
                row_sums = X_ready.sum(
                    axis=1
                ).A1  # .A1 to convert the sum to a 1D array
                row_inv = 1.0 / _np.maximum(row_sums, 1e-8)  # Avoid division by zero
                X_ready = _csr_matrix.multiply(X_ready, row_inv[:, _np.newaxis]) * self._target_sum
            else:
                X_ready = self._target_sum * _np.divide(
                    X_ready, _np.maximum(_np.sum(X_ready, axis=1).reshape(-1, 1), 1e-8)
                )
        if self._log1p:
            if _issparse(X_ready):
                X_ready.data = _np.log1p(X_ready.data)
            else:
                X_ready = _np.log1p(X_ready)
        if self._on_PCs:
            assert isinstance(self._PC_loadings, _NDArray)
            X_ready = X_ready @ self._PC_loadings.T
        return self._model.predict_proba(X_ready)

    def predict(
        self,
        X: _csr_matrix | _NDArray,
        genes: _Iterable[str] | None = None,
    ) -> _NDArray:
        """Predicts classes of each sample. For example, if prediction is i,
         then the predicted class is self.classes[i].
         For those below confidence threshold,
         predicted classes are set to -1.

        Args:
            X (_NDArray | _csr_matrix): input count matrix.

            genes (_Iterable[str] | None): list of genes corresponding to
             those of X's columns. If None, set to pretrained gene list.

        Return:
            _NDArray: an array of predicted classIds."""

        return super().predict(X, genes)

