import ctypes
import os
import sys
import copy
import math
from abc import ABC, abstractmethod
from enum import IntEnum
from dataclasses import dataclass
from typing import Any, Union, Callable, TypeVar, List, Tuple, Optional

import numpy as np
from pydantic import BaseModel, validate_call
from pydantic_core import core_schema


# --- Library Names ---

_SUNCORE_LIB_NAME = "libsundials_core.so"
_NVECSERIAL_LIB_NAME = "libsundials_nvecserial.so"
_SUNMATRIXDENSE_LIB_NAME = "libsundials_sunmatrixdense.so"
_SUNLINSOLDENSE_LIB_NAME = "libsundials_sunlinsoldense.so"
_DD_LIB_NAME = "libdd.so"

# --- Re-Define C Constants

_SUN_COMM_NULL = 0
IDALS_LMEM_NULL = -2


class IDASolveID(IntEnum):
    NORMAL = 1
    ONE_STEP = 2


class IDAStatus(IntEnum):
    SUCCESS = 0
    TSTOP_RETURN = 1
    ROOT_RETURN = 2


# --- Define C Types ---

_SUNErrCode = ctypes.c_int
_sunrealtype = ctypes.c_double
_sunindextype = ctypes.c_int


class SUNContext_STRUCT(ctypes.Structure):
    pass


SUNContext_PTR = ctypes.POINTER(SUNContext_STRUCT)


class N_Vector_STRUCT(ctypes.Structure):
    pass  # Opaque


N_Vector_PTR = ctypes.POINTER(N_Vector_STRUCT)


class SUNMatrix_STRUCT(ctypes.Structure):
    pass  # Opaque


SUNMatrix_PTR = ctypes.POINTER(SUNMatrix_STRUCT)


class DDStruc_STRUCT(ctypes.Structure):
    pass  # Opaque


DDStruc_PTR = ctypes.POINTER(DDStruc_STRUCT)


class DDMatrix_STRUCT(ctypes.Structure):
    pass  # Opaque


DDMatrix_PTR = ctypes.POINTER(DDMatrix_STRUCT)


class SUNLinSol_STRUCT(ctypes.Structure):
    pass  # Opaque


SUNLinSol_PTR = ctypes.POINTER(SUNLinSol_STRUCT)


class DDMem_STRUCT(ctypes.Structure):
    pass  # Opaque


DDMem_PTR = ctypes.POINTER(DDMem_STRUCT)

DDResFn_CFUN = ctypes.CFUNCTYPE(
    ctypes.c_int, _sunrealtype, N_Vector_PTR, N_Vector_PTR, ctypes.py_object
)

DDJacFn0_CFUN = ctypes.CFUNCTYPE(
    ctypes.c_int, _sunrealtype, N_Vector_PTR, SUNMatrix_PTR, ctypes.py_object
)

DDLsJacFn1_CFUN = ctypes.CFUNCTYPE(
    ctypes.c_int,
    _sunrealtype,
    N_Vector_PTR,
    N_Vector_PTR,
    SUNMatrix_PTR,
    ctypes.py_object,
    N_Vector_PTR,
    N_Vector_PTR,
    N_Vector_PTR,
)

DDLsJacFn2_CFUN = ctypes.CFUNCTYPE(
    ctypes.c_int,
    _sunrealtype,
    _sunrealtype,
    N_Vector_PTR,
    N_Vector_PTR,
    SUNMatrix_PTR,
    N_Vector_PTR,
    ctypes.py_object,
    N_Vector_PTR,
    N_Vector_PTR,
    N_Vector_PTR,
)


class DDLsJacFnId(IntEnum):
    DD_JAC_1 = 0
    DD_JAC_2 = 1


class DDLsJacFnUnion(ctypes.Union):
    _fields_ = [("jacfn1", DDLsJacFn1_CFUN), ("jacfn2", DDLsJacFn2_CFUN)]


class DDLsJacFn(ctypes.Structure):
    _fields_ = [("id", ctypes.c_int), ("fn", DDLsJacFnUnion)]


class DDPivotResult(IntEnum):
    PIVOT_SUCCESS = 0
    PIVOT_UNNECESSARY = 1
    PIVOT_FAIL = -1


# --- Define Function Signatures ---

# For now we add the shared libraries manually to the load path based on a
# hardcoded path.
_lib_path = os.path.abspath("/tmp/sundials/lib64")
_original_ld_path = os.environ.get("LD_LIBRARY_PATH", "")
os.environ["LD_LIBRARY_PATH"] = f"{_lib_path}:{_original_ld_path}"

try:
    _suncore_lib = ctypes.CDLL(os.path.join(_lib_path, _SUNCORE_LIB_NAME))
    _nvecserial_lib = ctypes.CDLL(os.path.join(_lib_path, _NVECSERIAL_LIB_NAME))
    _sunmatrixdense_lib = ctypes.CDLL(
        os.path.join(_lib_path, _SUNMATRIXDENSE_LIB_NAME)
    )
    _sunlinsoldense_lib = ctypes.CDLL(
        os.path.join(_lib_path, _SUNLINSOLDENSE_LIB_NAME)
    )
    _dd_lib = ctypes.CDLL(os.path.join(_lib_path, _DD_LIB_NAME))

    _SUNContext_Create = _suncore_lib.SUNContext_Create
    _SUNContext_Create.argtypes = [ctypes.c_int, ctypes.POINTER(SUNContext_PTR)]
    _SUNContext_Create.restype = _SUNErrCode

    _SUNContext_Free = _suncore_lib.SUNContext_Free
    _SUNContext_Free.argtypes = [ctypes.POINTER(SUNContext_PTR)]

    _N_VNew_Serial = _nvecserial_lib.N_VNew_Serial
    _N_VNew_Serial.argtypes = [_sunindextype, SUNContext_PTR]
    _N_VNew_Serial.restype = N_Vector_PTR

    _N_VDestroy = _suncore_lib.N_VDestroy
    _N_VDestroy.argtypes = [N_Vector_PTR]

    _N_VClone = _suncore_lib.N_VClone
    _N_VClone.argtypes = [N_Vector_PTR]
    _N_VClone.restype = N_Vector_PTR

    _N_VGetLength = _suncore_lib.N_VGetLength
    _N_VGetLength.argtypes = [N_Vector_PTR]
    _N_VGetLength.restype = _sunindextype

    _N_VGetArrayPointer = _suncore_lib.N_VGetArrayPointer
    _N_VGetArrayPointer.argtypes = [N_Vector_PTR]
    _N_VGetArrayPointer.restype = ctypes.POINTER(_sunrealtype)

    _N_VConst = _suncore_lib.N_VConst
    _N_VConst.argtypes = [ctypes.c_double, N_Vector_PTR]
    _N_VConst.restype = None

    _N_VLinearSum = _suncore_lib.N_VLinearSum
    _N_VLinearSum.argtypes = [
        _sunrealtype,
        N_Vector_PTR,
        _sunrealtype,
        N_Vector_PTR,
        N_Vector_PTR,
    ]
    _N_VLinearSum.restype = None

    _N_VScale = _suncore_lib.N_VScale
    _N_VScale.argtypes = [_sunrealtype, N_Vector_PTR, N_Vector_PTR]
    _N_VScale.restype = None

    _SUNDenseMatrix = _sunmatrixdense_lib.SUNDenseMatrix
    _SUNDenseMatrix.argtypes = [_sunindextype, _sunindextype, SUNContext_PTR]
    _SUNDenseMatrix.restype = SUNMatrix_PTR

    _SUNDenseMatrix_Rows = _sunmatrixdense_lib.SUNDenseMatrix_Rows
    _SUNDenseMatrix_Rows.argtypes = [SUNMatrix_PTR]
    _SUNDenseMatrix_Rows.restype = _sunindextype

    _SUNDenseMatrix_Columns = _sunmatrixdense_lib.SUNDenseMatrix_Columns
    _SUNDenseMatrix_Columns.argtypes = [SUNMatrix_PTR]
    _SUNDenseMatrix_Columns.restype = _sunindextype

    _SUNDenseMatrix_Column = _sunmatrixdense_lib.SUNDenseMatrix_Column
    _SUNDenseMatrix_Column.argtypes = [SUNMatrix_PTR]
    _SUNDenseMatrix_Column.restype = ctypes.POINTER(_sunrealtype)

    _SUNDenseMatrix_Data = _sunmatrixdense_lib.SUNDenseMatrix_Data
    _SUNDenseMatrix_Data.argtypes = [SUNMatrix_PTR]
    _SUNDenseMatrix_Data.restype = ctypes.POINTER(_sunrealtype)

    _SUNDenseMatrix_LData = _sunmatrixdense_lib.SUNDenseMatrix_LData
    _SUNDenseMatrix_LData.argtypes = [SUNMatrix_PTR]
    _SUNDenseMatrix_LData.restype = _sunindextype

    _SUNMatDestroy = _suncore_lib.SUNMatDestroy
    _SUNMatDestroy.argtypes = [SUNMatrix_PTR]
    _SUNMatDestroy.restype = None

    _SUNMatClone = _suncore_lib.SUNMatClone
    _SUNMatClone.argtypes = [SUNMatrix_PTR]
    _SUNMatClone.restype = SUNMatrix_PTR

    _SUNMatCopy = _suncore_lib.SUNMatCopy
    _SUNMatCopy.argtypes = [SUNMatrix_PTR, SUNMatrix_PTR]
    _SUNMatCopy.restype = _SUNErrCode

    _SUNLinSolFree = _suncore_lib.SUNLinSolFree
    _SUNLinSolFree.argtypes = [SUNLinSol_PTR]
    _SUNLinSolFree.restype = None

    _SUNLinSol_Dense = _sunlinsoldense_lib.SUNLinSol_Dense
    _SUNLinSol_Dense.argtypes = [N_Vector_PTR, SUNMatrix_PTR, SUNContext_PTR]
    _SUNLinSol_Dense.restype = SUNLinSol_PTR

    _DDMatWrapDense = _dd_lib.DDMatWrapDense
    _DDMatWrapDense.argtypes = [SUNMatrix_PTR]
    _DDMatWrapDense.restype = DDMatrix_PTR

    _DDMatDestroy = _dd_lib.DDMatDestroy
    _DDMatDestroy.argtypes = [DDMatrix_PTR]
    _DDMatDestroy.restype = None

    _STCreate = _dd_lib.STCreate
    _STCreate.argtypes = [
        _sunindextype,
        ctypes.POINTER(ctypes.c_uint8),
        ctypes.POINTER(ctypes.c_uint8),
        ctypes.POINTER(ctypes.c_char_p),
        ctypes.POINTER(ctypes.c_char_p),
    ]
    _STCreate.restype = DDStruc_PTR

    _STDestroy = _dd_lib.STDestroy
    _STDestroy.argtypes = [DDStruc_PTR]
    _STDestroy.restype = None

    _DDCreate = _dd_lib.DDCreate
    _DDCreate.argtypes = [SUNContext_PTR]
    _DDCreate.restype = DDMem_PTR

    _DDInit = _dd_lib.DDInit
    _DDInit.argtypes = [
        DDMem_PTR,
        DDStruc_PTR,
        _sunrealtype,
        DDJacFn0_CFUN,
        DDMatrix_PTR,
        DDResFn_CFUN,
        _sunrealtype,
        N_Vector_PTR,
    ]
    _DDInit.restype = ctypes.c_int

    _DDSolve = _dd_lib.DDSolve
    _DDSolve.argtypes = [
        DDMem_PTR,
        _sunrealtype,
        ctypes.POINTER(_sunrealtype),
        N_Vector_PTR,
        ctypes.c_int,
    ]
    _DDSolve.restype = ctypes.c_int

    _DDPivot = _dd_lib.DDPivot
    _DDPivot.argtypes = [DDMem_PTR]
    _DDPivot.restype = ctypes.c_int

    _DDSetLinearSolver = _dd_lib.DDSetLinearSolver
    _DDSetLinearSolver.argtypes = [DDMem_PTR, SUNLinSol_PTR, SUNMatrix_PTR]
    _DDSetLinearSolver.restype = ctypes.c_int

    _DDSetJacFn = _dd_lib.DDSetJacFn
    _DDSetJacFn.argtypes = [DDMem_PTR, DDLsJacFn]
    _DDSetJacFn.restype = ctypes.c_int

    _DDSetUserData = _dd_lib.DDSetUserData
    _DDSetUserData.argtypes = [DDMem_PTR, ctypes.py_object]
    _DDSetUserData.restype = ctypes.c_int

    _DDSSTolerances = _dd_lib.DDSSTolerances
    _DDSSTolerances.argtypes = [DDMem_PTR, _sunrealtype, _sunrealtype]
    _DDSSTolerances.restype = ctypes.c_int

    _DDSetStopTime = _dd_lib.DDSetStopTime
    _DDSetStopTime.argtypes = [DDMem_PTR, _sunrealtype]
    _DDSetStopTime.restype = ctypes.c_int

    _DDFree = _dd_lib.DDFree
    _DDFree.argtypes = [ctypes.POINTER(DDMem_PTR)]

except OSError as e:
    print(e)
    os.environ["LD_LIBRARY_PATH"] = _original_ld_path
    sys.exit(1)

os.environ["LD_LIBRARY_PATH"] = _original_ld_path


# --- Wrapper Classes ---


class SUNError(Exception):
    """Exception for Sundials failures."""

    @validate_call
    def __init__(self, message: str, error_code: int):
        super().__init__(message)
        self.error_code = error_code


class SUNContext:
    """Simple Sundials context for logging and profiling"""

    @classmethod
    def __get_pydantic_core_schema__(cls, source_type: Any, handler):
        """Teaches Pydantic to validate this type by instance check."""
        return core_schema.is_instance_schema(cls)

    def __init__(self):
        self.sunctx = SUNContext_PTR()

        code = _SUNContext_Create(_SUN_COMM_NULL, ctypes.byref(self.sunctx))
        if code < 0:
            raise SUNError("Failed to create Sundials context", error_code=code)

    def _free(self):
        if self.sunctx:
            _SUNContext_Free(ctypes.byref(self.sunctx))
            self.sunctx = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._free()

    def __del__(self):
        self._free()


class NVector(ABC):
    """Wraps Sundials NVectors"""

    @abstractmethod
    def view(self, nvector) -> "NVector":
        pass

    @classmethod
    def __get_pydantic_core_schema__(cls, source_type: Any, handler):
        """Teaches Pydantic to validate this type by instance check."""
        return core_schema.is_instance_schema(cls)

    @validate_call
    def __init__(self, nvector, sunctx: SUNContext, is_view: bool = False):
        if not isinstance(nvector, N_Vector_PTR):
            raise TypeError("Expected N_Vector_PTR")

        self.nvector = nvector

        # Keep a strong reference to avoid premature GC.
        self.sunctx = sunctx

        self._is_view = is_view
        self._ptr = _N_VGetArrayPointer(nvector)
        size = len(self)
        if size > 0:
            self.np_array = np.ctypeslib.as_array(self._ptr, shape=(size,))
        else:
            self.np_array = np.empty(0)

    def __len__(self) -> int:
        return _N_VGetLength(self.nvector)

    def __iadd__(self, other):
        if not isinstance(other, NVector):
            return NotImplemented
        if len(self) != len(other):
            raise ValueError("NVectors must have the same length to be added.")

        _N_VLinearSum(1, self.nvector, 1, other.nvector, self.nvector)
        return self

    def __imul__(self, scalar: Union[int, float]):
        if not isinstance(scalar, (int, float)):
            return NotImplemented

        _N_VScale(scalar, self.nvector, self.nvector)
        return self

    def __add__(self, other):
        if not isinstance(other, NVector):
            return NotImplemented

        v = copy.deepcopy(self)
        _N_VLinearSum(1, self.nvector, 1, other.nvector, v.nvector)
        return v

    def __mul__(self, scalar: Union[int, float]):
        if not isinstance(scalar, (int, float)):
            return NotImplemented

        v = copy.deepcopy(self)
        _N_VScale(scalar, self.nvector, v.nvector)
        return v

    def __rmul__(self, scalar: Union[int, float]):
        return self.__mul__(scalar)

    def _check_index(self, index):
        if not isinstance(index, int):
            raise TypeError("NVector object only support integer indexing.")
        if not 0 <= index < len(self):
            raise IndexError("Index out of bounds")

    def __getitem__(self, index: int) -> float:
        self._check_index(index)

        return self._ptr[index]

    def __setitem__(self, index: int, value: Union[float, int]) -> None:
        if not isinstance(value, (int, float)):
            raise TypeError("NVectors can only hold numbers.")
        self._check_index(index)

        self._ptr[index] = value

    def __copy__(self):
        raise TypeError("NVector objects cannot be shallow copied.")

    def __enter__(self):
        return self

    def _free(self):
        if (not self.nvector is None) and (not self._is_view):
            _N_VDestroy(self.nvector)
            self.nvector = None

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._free()

    def __del__(self):
        self._free()


class NVectorSerial(NVector):

    @validate_call
    def __init__(self, nvector, sunctx: SUNContext, is_view: bool):
        super().__init__(nvector, sunctx, is_view)

    @classmethod
    @validate_call
    def empty(cls, size: int, sunctx: SUNContext):
        nvector = _N_VNew_Serial(size, sunctx.sunctx)
        return cls(nvector, sunctx, is_view=False)

    @classmethod
    @validate_call
    def from_list(cls, seq: List[Union[int, float]], sunctx: SUNContext):
        size = len(seq)
        nvector = cls.empty(size, sunctx)
        nvector.np_array[:] = seq
        return nvector

    @classmethod
    @validate_call
    def from_nvector(cls, nvector: NVector):
        return cls(nvector.nvector, nvector.sunctx, is_view=False)

    def view(self, nvector) -> "NVectorSerial":
        return NVectorSerial(nvector, self.sunctx, is_view=True)

    def __repr__(self) -> str:
        return (
            f"NVectorSerial({len(self)}, {[self[i] for i in range(len(self))]})"
        )

    def __deepcopy__(self, memo):
        v = NVectorSerial.empty(len(self), self.sunctx)
        _N_VScale(1, self.nvector, v.nvector)
        return v


class SUNMatrix(ABC):

    @abstractmethod
    def view(self, sunmatrix) -> "SUNMatrix":
        pass

    @classmethod
    def __get_pydantic_core_schema__(cls, source_type: Any, handler):
        """Teaches Pydantic to validate this type by instance check."""
        return core_schema.is_instance_schema(cls)

    @validate_call
    def __init__(
        self,
        sunmatrix,
        sunctx: SUNContext,
        is_view: bool = False,
    ):
        if not isinstance(sunmatrix, SUNMatrix_PTR):
            raise TypeError("Expected SUNMatrix_PTR")

        # Keep a strong reference to avoid premature GC.
        self.sunctx = sunctx

        self.sunmatrix = sunmatrix
        self._is_view = is_view

    def __copy__(self):
        raise TypeError("SUNMatrix objects cannot be shallow copied.")

    def __enter__(self):
        return self

    def _free(self):
        if (not self.sunmatrix is None) and (not self._is_view):
            _SUNMatDestroy(self.sunmatrix)
            self.sunmatrix = None

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._free()

    def __del__(self):
        self._free()


class SUNDenseMatrix(SUNMatrix):

    @validate_call
    def __init__(self, sunmatrix, sunctx: SUNContext, is_view: bool):
        super().__init__(sunmatrix, sunctx, is_view)
        self.size = _SUNDenseMatrix_LData(sunmatrix)
        self._data = _SUNDenseMatrix_Data(sunmatrix)
        if self.size > 0:
            self.np_array = np.ctypeslib.as_array(
                self._data, shape=(self.rows, self.columns)
            )
        else:
            self.np_array = np.empty(0)

    @classmethod
    @validate_call
    def empty(cls, rows: int, cols: int, sunctx: SUNContext):
        sunmatrix = _SUNDenseMatrix(rows, cols, sunctx.sunctx)
        return cls(sunmatrix, sunctx, is_view=False)

    @classmethod
    @validate_call
    def from_sunmatrix(cls, sunmatrix: SUNMatrix):
        return cls(sunmatrix.sunmatrix, sunmatrix.sunctx, is_view=False)

    def view(self, sunmatrix):
        return SUNDenseMatrix(sunmatrix, self.sunctx, is_view=True)

    def __deepcopy__(self, memo):
        m = SUNDenseMatrix.empty(self.rows, self.columns, self.sunctx)
        code = _SUNMatCopy(self.sunmatrix, m.sunmatrix)
        assert code >= 0
        return m

    def __repr__(self) -> str:
        return f"SUNDenseMatrix({self.rows}, {self.columns}, {[self._data[i] for i in range(self.size)]})"

    @property
    def rows(self):
        return _SUNDenseMatrix_Rows(self.sunmatrix)

    @property
    def columns(self):
        return _SUNDenseMatrix_Columns(self.sunmatrix)

    def _check_index(self, index: Tuple[int, int]) -> None:
        if not isinstance(index, tuple) or len(index) != 2:
            raise TypeError("SUNMatrix indexing requires two integer indices.")
        i, j = index
        if not isinstance(i, int) or not isinstance(j, int):
            raise TypeError("SUNMatrix indices must be integers.")
        if not (0 <= i < self.rows and 0 <= j < self.columns):
            raise IndexError("Index out of bounds")

    def __getitem__(self, index: Tuple[int, int]) -> float:
        self._check_index(index)

        i, j = index
        return _SUNDenseMatrix_Column(self.sunmatrix, j)[i]

    def __setitem__(
        self, index: tuple[int, int], value: Union[int, float]
    ) -> None:
        self._check_index(index)

        i, j = index
        _SUNDenseMatrix_Column(self.sunmatrix, j)[i] = value


class SUNLinSol(ABC):

    @classmethod
    def __get_pydantic_core_schema__(cls, source_type: Any, handler):
        """Teaches Pydantic to validate this type by instance check."""
        return core_schema.is_instance_schema(cls)

    def __init__(self, sunlinsol):
        if not isinstance(sunlinsol, SUNLinSol_PTR):
            raise TypeError("Expected SUNLinSol_PTR")

        self.sunlinsol = sunlinsol

    def __copy__(self):
        raise TypeError("SUNLineSol objects cannot be shallow copied.")

    def __deepcopy__(self, memo):
        raise TypeError("SUNLineSol objects cannot be deep copied.")

    def __enter__(self):
        return self

    def _free(self):
        if self.sunlinsol:
            _SUNLinSolFree(self.sunlinsol)
            self.sunlinsol = None

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._free()

    def __del__(self):
        self._free()


class SUNLinSol_Dense(SUNLinSol):

    @validate_call
    def __init__(
        self, nvector: NVector, sunmatrix: SUNMatrix, sunctx: SUNContext
    ):
        # Keep a strong reference to avoid premature GC.
        self._nvector = nvector
        self._sunmatrix = sunmatrix
        self._sunctx = sunctx

        sunlinsol = _SUNLinSol_Dense(
            nvector.nvector, sunmatrix.sunmatrix, sunctx.sunctx
        )
        super().__init__(sunlinsol)


class DAEStructure:

    @classmethod
    def __get_pydantic_core_schema__(cls, source_type: Any, handler):
        """Teaches Pydantic to validate this type by instance check."""
        return core_schema.is_instance_schema(cls)

    @staticmethod
    def _check_offsets(offsets):
        if not isinstance(offsets, list):
            raise TypeError("Offsets must be a list.")
        if not all(
            (isinstance(offset, int) and (offset >= 0)) for offset in offsets
        ):
            raise TypeError("Offset must be a positive integer.")
        if len(offsets) < 1:
            raise ValueError("There must be at least one offset")

    def __init__(self, eqn_offsets: List[int], var_offsets: List[int]):
        DAEStructure._check_offsets(eqn_offsets)
        DAEStructure._check_offsets(var_offsets)
        if len(eqn_offsets) != len(var_offsets):
            raise ValueError(
                "The number of equations and variables must be the same"
            )
        CArray = ctypes.c_uint8 * len(eqn_offsets)
        self.struc = _STCreate(
            len(eqn_offsets),
            CArray(*eqn_offsets),
            CArray(*var_offsets),
            None,
            None,
        )

    def __copy__(self):
        raise TypeError("Struc objects cannot be shallow copied.")

    def __deepcopy__(self, memo):
        raise TypeError("Struc objects cannot be deep copied.")

    def __enter__(self):
        return self

    def _free(self):
        if self.struc:
            _STDestroy(self.struc)
            self.struc = None

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._free()

    def __del__(self):
        self._free()


class DDMatrix(ABC):

    @classmethod
    def __get_pydantic_core_schema__(cls, source_type: Any, handler):
        """Teaches Pydantic to validate this type by instance check."""
        return core_schema.is_instance_schema(cls)

    @validate_call
    def __init__(self, ddmatrix, sunmatrix: SUNMatrix):
        if not isinstance(ddmatrix, DDMatrix_PTR):
            raise TypeError("Expected DDMatrix_PTR")

        self.ddmatrix = ddmatrix
        self.sunmatrix = sunmatrix

    def __copy__(self):
        raise TypeError("DDMatrix objects cannot be shallow copied.")

    def __deepcopy__(self, memo):
        raise TypeError("DDMatrix objects cannot be deep copied.")

    def __enter__(self):
        return self

    def _free(self):
        if self.ddmatrix:
            _DDMatDestroy(self.ddmatrix)
            self.ddmatrix = None

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._free()

    def __del__(self):
        self._free()


class DDMatrixDense(DDMatrix):
    def __init__(self, sunmatrix: SUNDenseMatrix):
        if not isinstance(sunmatrix, SUNDenseMatrix):
            raise TypeError("DDMatrixDense can only wrap SUNDenseMatrix")

        ddmatrix = _DDMatWrapDense(sunmatrix.sunmatrix)
        super().__init__(ddmatrix, sunmatrix)


class SolverError(SUNError):
    """Exception for solver failures."""


@dataclass
class DDJacFn1:
    fn: Callable[
        [float, NVector, NVector, SUNMatrix, Any, NVector, NVector, NVector],
        int,
    ]


@dataclass
class DDJacFn2:
    fn: Callable[
        [
            float,
            float,
            NVector,
            NVector,
            SUNMatrix,
            NVector,
            Any,
            NVector,
            NVector,
            NVector,
        ],
        int,
    ]


class DDMem:

    @classmethod
    def __get_pydantic_core_schema__(cls, source_type: Any, handler):
        """Teaches Pydantic to validate this type by instance check."""
        return core_schema.is_instance_schema(cls)

    @validate_call
    def __init__(self, sunctx: SUNContext):
        # Keep a strong reference to avoid premature GC.
        self._sunctx = sunctx

        self._struc: Optional[DAEStructure] = None
        self._J0: Optional[DDMatrix] = None
        self._jacfn0_ptr: Any = None
        self._resfn_ptr: Any = None
        self._Y0: Optional[NVector] = None
        self._sunlinsol: Optional[SUNLinSol] = None
        self._J: Optional[SUNMatrix] = None
        self._jacfn_ptr: Any = None

        self._ddmem = _DDCreate(sunctx.sunctx)
        self._initialized = False

    def __copy__(self):
        raise TypeError("DDMem objects cannot be shallow copied.")

    def __deepcopy__(self, memo):
        raise TypeError("DDMem objects cannot be deep copied.")

    @validate_call
    def init(
        self,
        struc: DAEStructure,
        ptol: float,
        jacfn0: Callable[[float, NVector, SUNMatrix, Any], int],
        J0: DDMatrix,
        resfn: Callable[[float, NVector, NVector, Any], int],
        t0: float,
        Y0: NVector,
    ):
        jacfn0_ptr = DDJacFn0_CFUN(
            lambda t, Y, J, user_data: jacfn0(
                t, Y0.view(Y), J0.sunmatrix.view(J), user_data
            )
        )
        resfn_ptr = DDResFn_CFUN(
            lambda t, Y, R, user_data: resfn(
                t, Y0.view(Y), Y0.view(R), user_data
            )
        )

        code = _DDInit(
            self._ddmem,
            struc.struc,
            ptol,
            jacfn0_ptr,
            J0.ddmatrix,
            resfn_ptr,
            t0,
            Y0.nvector,
        )
        if code < 0:
            raise SolverError("Failed to initialize solver", error_code=code)

        # Keep a strong reference to avoid premature GC.
        self._struc = struc
        self._J0 = J0
        self._jacfn0_ptr = jacfn0_ptr
        self._resfn_ptr = resfn_ptr
        self._Y0 = Y0
        self._initialized = True

    def _solve(
        self, tstop: Union[int, float], Y: NVector, solve_id: IDASolveID
    ) -> Tuple[IDAStatus, float]:
        tret = _sunrealtype(tstop)

        code = _DDSolve(
            self._ddmem,
            tstop,
            ctypes.byref(tret),
            Y.nvector,
            solve_id,
        )
        if code < 0:
            raise SolverError("Failed to solve", error_code=code)
        return (IDAStatus(code), tret.value)

    @validate_call
    def solve_normal(
        self, tstop: Union[int, float], Y: NVector
    ) -> Tuple[IDAStatus, float]:
        return self._solve(tstop, Y, IDASolveID.NORMAL)

    @validate_call
    def solve_step(
        self, tstop: Union[int, float], Y: NVector
    ) -> Tuple[IDAStatus, float]:
        return self._solve(tstop, Y, IDASolveID.ONE_STEP)

    @validate_call
    def pivot(self) -> DDPivotResult:
        return _DDPivot(self._ddmem)

    @validate_call
    def set_linear_solver(self, sunlinsol: SUNLinSol, sunmatrix: SUNMatrix):
        code = _DDSetLinearSolver(
            self._ddmem, sunlinsol.sunlinsol, sunmatrix.sunmatrix
        )
        if code < 0:
            raise SolverError("Failed to set linear solver", error_code=code)

        # Keep a strong reference to avoid premature GC.
        self._sunlinsol = sunlinsol
        self._J = sunmatrix

    @validate_call
    def set_jacfn(self, jacfn: Union[DDJacFn1, DDJacFn2]) -> None:
        code = IDALS_LMEM_NULL
        nvector = self._Y0
        sunmatrix = self._J

        if nvector is not None and sunmatrix is not None:
            nv_view = nvector.view
            sm_view = sunmatrix.view

            match jacfn:
                case DDJacFn1(fn=fn1):
                    fn_ptr = DDLsJacFn1_CFUN(
                        lambda t, Y, R, J, user_data, tmp1, tmp2, tmp3: fn1(
                            t,
                            nv_view(Y),
                            nv_view(R),
                            sm_view(J),
                            user_data,
                            nv_view(tmp1),
                            nv_view(tmp2),
                            nv_view(tmp3),
                        )
                    )
                    fn_id = DDLsJacFnId.DD_JAC_1
                case DDJacFn2(fn=fn2):
                    fn_ptr = DDLsJacFn2_CFUN(
                        lambda t, cj, Y, R, J, var_id, user_data, tmp1, tmp2, tmp3: fn2(
                            t,
                            cj,
                            nv_view(Y),
                            nv_view(R),
                            sm_view(J),
                            nv_view(var_id),
                            user_data,
                            nv_view(tmp1),
                            nv_view(tmp2),
                            nv_view(tmp3),
                        )
                    )
                    fn_id = DDLsJacFnId.DD_JAC_2
                case _:
                    TypeError("Expected Union[DDJacFn1, DDJacFn2]")

            code = _DDSetJacFn(
                self._ddmem,
                DDLsJacFn(id=fn_id, fn=DDLsJacFnUnion(fn_ptr)),
            )

        if code < 0:
            raise SolverError(
                "Failed to set Jacobian function", error_code=code
            )

        self._jacfn_ptr = fn_ptr

    @validate_call
    def set_user_data(self, user_data) -> None:
        code = _DDSetUserData(self._ddmem, user_data)
        if code < 0:
            raise SolverError("Failed to set user data", error_code=code)

    @validate_call
    def set_tolerances(self, reltol: float, abstol: float) -> None:
        code = _DDSSTolerances(self._ddmem, reltol, abstol)
        if code < 0:
            raise SolverError("Failed to set tolerances", error_code=code)

    @validate_call
    def set_stop_time(self, tstop: Union[int, float]) -> None:
        code = _DDSetStopTime(self._ddmem, tstop)
        if code < 0:
            raise SolverError("Failed to set stop time", error_code=code)

    def _free(self):
        if self._initialized and self._ddmem:
            _DDFree(ctypes.byref(self._ddmem))
            self._ddmem = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._free()

    def __del__(self):
        self._free()
