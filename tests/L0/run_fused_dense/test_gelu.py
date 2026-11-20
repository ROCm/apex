from apex import fused_dense
import torch
import torch.nn.functional as F
import unittest


class FusedDenseGeluDenseTestBase:
    blas_library = None

    def setUp(self):
        self.prev_blas_library = torch.backends.cuda.preferred_blas_library()
        torch.backends.cuda.preferred_blas_library(self.blas_library)
        # GPUs without hipBLASLt silently stay on hipBLAS. Restore here: tearDown is skipped too.
        if self.blas_library == "hipblaslt" and torch.backends.cuda.preferred_blas_library() != torch._C._BlasBackend.Cublaslt:
            torch.backends.cuda.preferred_blas_library(self.prev_blas_library)
            self.skipTest("hipBLASLt is not supported on this GPU")

    def tearDown(self):
        torch.backends.cuda.preferred_blas_library(self.prev_blas_library)

    def test_fused_dense_gelu_dense(self) :
        seed = 0
        torch.manual_seed(seed)
        batch_size   = 4
        in_features  = 3
        intermediate_features = 3
        out_features = 2

        #tst_dtype = torch.float8_e4m3
        # tst_dtype = torch.float8_e5m2
        tst_dtype = torch.float16

        I = torch.randn(batch_size, in_features, dtype=tst_dtype, device='cuda').requires_grad_(True)

        denseGelu = fused_dense.FusedDenseGeluDense(in_features, intermediate_features, out_features)
        denseGelu.to(dtype=tst_dtype)
        denseGelu.cuda()

        #get weight and bias from the denseGelu module
        W1 = denseGelu.weight1
        b1 = denseGelu.bias1
        W2 = denseGelu.weight2
        b2 = denseGelu.bias2

        y_tst = denseGelu(I.clone().detach().requires_grad_(True))

        C1  = torch.matmul(I, W1.t())+b1
        gelu_output = F.gelu(C1)
        y_ref = torch.matmul(gelu_output, W2.t())+b2
        torch.testing.assert_close(y_ref,  y_tst,  atol=1e-3, rtol=1e-3, equal_nan=True)

    def test_fused_dense_gelu_dense_backward(self) :
        seed = 0
        torch.manual_seed(seed)
        batch_size   = 4
        in_features  = 3
        intermediate_features = 3
        out_features = 2

        tst_dtype = torch.float16

        I = torch.randn(batch_size, in_features, dtype=tst_dtype, device='cuda')

        denseGelu = fused_dense.FusedDenseGeluDense(in_features, intermediate_features, out_features)
        denseGelu.to(dtype=tst_dtype)
        denseGelu.cuda()

        I_tst = I.clone().requires_grad_(True)
        I_ref = I.clone().requires_grad_(True)
        W1 = denseGelu.weight1.detach().clone().requires_grad_(True)
        b1 = denseGelu.bias1.detach().clone().requires_grad_(True)
        W2 = denseGelu.weight2.detach().clone().requires_grad_(True)
        b2 = denseGelu.bias2.detach().clone().requires_grad_(True)

        y_tst = denseGelu(I_tst)

        C1  = torch.matmul(I_ref, W1.t())+b1
        gelu_output = F.gelu(C1, approximate="tanh")
        y_ref = torch.matmul(gelu_output, W2.t())+b2

        dy = torch.randn_like(y_tst)
        y_tst.backward(dy)
        y_ref.backward(dy)

        torch.testing.assert_close(I_ref.grad,  I_tst.grad,              atol=1e-3, rtol=1e-3, equal_nan=True)
        torch.testing.assert_close(W1.grad,     denseGelu.weight1.grad,  atol=1e-3, rtol=1e-3, equal_nan=True)
        torch.testing.assert_close(b1.grad,     denseGelu.bias1.grad,    atol=1e-3, rtol=1e-3, equal_nan=True)
        torch.testing.assert_close(W2.grad,     denseGelu.weight2.grad,  atol=1e-3, rtol=1e-3, equal_nan=True)
        torch.testing.assert_close(b2.grad,     denseGelu.bias2.grad,    atol=1e-3, rtol=1e-3, equal_nan=True)


class FusedDenseGeluDenseHipblasTest(FusedDenseGeluDenseTestBase, unittest.TestCase):
    blas_library = "hipblas"


class FusedDenseGeluDenseHipblasltTest(FusedDenseGeluDenseTestBase, unittest.TestCase):
    blas_library = "hipblaslt"

    # Known hipBLASLt-path bug: forward doesn't save the pre-activation; backward math is wrong.
    @unittest.expectedFailure
    def test_fused_dense_gelu_dense_backward(self):
        super().test_fused_dense_gelu_dense_backward()


if __name__ == '__main__':
    unittest.main()
