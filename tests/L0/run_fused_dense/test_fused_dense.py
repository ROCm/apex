import torch
import unittest
import torch.nn.functional as F
from apex import fused_dense
from torch import nn
from apex import amp

class FusedDenseTestBase:
    blas_library = None

    def setUp(self, seed=0):
        self.prev_blas_library = torch.backends.cuda.preferred_blas_library()
        torch.backends.cuda.preferred_blas_library(self.blas_library)
        # GPUs without hipBLASLt silently stay on hipBLAS. Restore here: tearDown is skipped too.
        if self.blas_library == "hipblaslt" and torch.backends.cuda.preferred_blas_library() != torch._C._BlasBackend.Cublaslt:
            torch.backends.cuda.preferred_blas_library(self.prev_blas_library)
            self.skipTest("hipBLASLt is not supported on this GPU")
        torch.manual_seed(seed)
        # torch.cuda.manual_seed_all(seed)

        self.seq_length   = 512
        self.sequences    = 3
        self.hidden_dim   = 1024

        self.ref_inputs = torch.randn(self.sequences*self.seq_length, self.hidden_dim,
                                      dtype=torch.float16, device=torch.device("cuda")).half().requires_grad_(True)

        self.tst_inputs = self.ref_inputs.clone().detach().requires_grad_(True)
        self.dense = fused_dense.FusedDense(1024, 3072)
        self.dense.half()
        self.dense.cuda()

    def tearDown(self):
        torch.backends.cuda.preferred_blas_library(self.prev_blas_library)


    def test_fused_dense(self) :
        y_tst = self.dense(self.tst_inputs)
        y_ref = torch.matmul(self.ref_inputs,self.dense.weight.t())+self.dense.bias
        dy = torch.randn_like(y_tst).half()
        y_tst.backward(dy)
        dw_ref = torch.matmul(dy.t(), self.ref_inputs)
        dx_ref = torch.matmul(dy, self.dense.weight.clone())
        db_ref = dy.sum(0, False)

        self.assertTrue(torch.allclose(self.ref_inputs,  self.tst_inputs,        atol=1e-5, rtol=1e-5))
        self.assertTrue(torch.allclose(y_ref,            y_tst,                  atol=1e-3, rtol=1e-3, equal_nan=True))
        self.assertTrue(torch.allclose(dw_ref,           self.dense.weight.grad, atol=1e-3, rtol=1e-3, equal_nan=True))
        self.assertTrue(torch.allclose(dx_ref,           self.tst_inputs.grad,   atol=1e-3, rtol=1e-3, equal_nan=True))
        self.assertTrue(torch.allclose(db_ref,           self.dense.bias.grad,   atol=1e-3, rtol=1e-3, equal_nan=True))


class FusedDenseHipblasTest(FusedDenseTestBase, unittest.TestCase):
    blas_library = "hipblas"


class FusedDenseHipblasltTest(FusedDenseTestBase, unittest.TestCase):
    blas_library = "hipblaslt"


if __name__ == '__main__':
    unittest.main()