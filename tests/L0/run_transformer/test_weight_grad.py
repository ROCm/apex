
import torch
import fused_weight_gradient_mlp_cuda
import torch.nn as nn
import sys
import math
import unittest

class TestFusedWeightGradDenseCuda(unittest.TestCase):
    SEED: int = 123456
    configs = [
        #BATCH_SIZE, SEQUENCE_LENGTH, HIDDEN_SIZE, OUTPUT_SIZE
        (1, 14, 1531, 500),
        (1, 32, 1536, 256),
        (1, 64, 3584, 512),
        (1, 128, 4096, 1024),
        (8, 32, 1536, 256),
        (8, 64, 3584, 2048),
        (8, 128, 4096, 4096),       
    ]
    
    def backward_matrix_operation(self, alpha, beta, A, B, C):

        A_sizes = A.size()
        if A.dim() > 2:
            new_shape = (-1, A_sizes[-1])
            reshaped_A = A.view(new_shape)
        else:
            reshaped_A = A   
        
        B_sizes = B.size()
        if B.dim() > 2:
            new_shape = (-1, B_sizes[-1])
            reshaped_B = B.view(new_shape)
        else:
            reshaped_B = B  
        
        AB = reshaped_B.float().t().matmul(reshaped_A.float())
        scaled_AB = alpha * AB
        
        output_type = C.dtype
        C_temp = C.float()

        C_temp.data.mul_(beta)
        C_temp.data.add_(scaled_AB) 

        C.copy_(C_temp).to(dtype=output_type, device="cuda")
        return C


    def RunWeightGradDense(self, ab_type, cd_type, tol):
        for BATCH_SIZE, SEQUENCE_LENGTH, HIDDEN_SIZE, OUTPUT_SIZE in self.configs:
            #print(f"BATCH_SIZE = {BATCH_SIZE}, SEQUENCE_LENGTH = {SEQUENCE_LENGTH}, HIDDEN_SIZE = {HIDDEN_SIZE}, OUTPUT_SIZE = {OUTPUT_SIZE}")
            input_shape = (SEQUENCE_LENGTH, BATCH_SIZE, HIDDEN_SIZE) # k*m
            output_shape = (SEQUENCE_LENGTH, BATCH_SIZE, OUTPUT_SIZE) # k*n
            weight_shape = (OUTPUT_SIZE, HIDDEN_SIZE)
            
            torch.manual_seed(self.SEED)

            input = torch.randn(input_shape, requires_grad=False, device="cuda") # A
            grad_output = torch.randn(output_shape, requires_grad=False, device="cuda") #B
            grad_weight = torch.randn(weight_shape, requires_grad=False, device="cuda") # C

            ref_grad_weight = grad_weight.clone()
            
            if ab_type == 'fp16':
                input = input.half()
                grad_output = grad_output.half()
            elif ab_type == 'bf16':
                input = input.bfloat16()
                grad_output = grad_output.bfloat16()
        
            if cd_type == 'fp16':
                grad_weight = grad_weight.half()
                ref_grad_weight = ref_grad_weight.half()
            elif cd_type == 'bf16':
                grad_weight = grad_weight.bfloat16()
                ref_grad_weight = ref_grad_weight.bfloat16()

            if cd_type == 'fp32':
                fused_weight_gradient_mlp_cuda.wgrad_gemm_accum_fp32(
                    input, grad_output, grad_weight
                )
            else: 
                fused_weight_gradient_mlp_cuda.wgrad_gemm_accum_fp16(
                    input, grad_output, grad_weight
                )

            alpha = 1.0
            beta = 1.0
            self.backward_matrix_operation(alpha, beta, input, grad_output, ref_grad_weight)

            grad_weight = grad_weight.view(-1)
            ref_grad_weight = ref_grad_weight.view(-1)

            self.assertTrue(torch.allclose(grad_weight, ref_grad_weight, atol=tol, rtol=tol))

    def test_fused_weight_grad_dense_cuda_ifp32_ofp32(self):
        self.RunWeightGradDense('fp32', 'fp32', 1e-4)
    def test_fused_weight_grad_dense_cuda_ifp16_ofp32(self):
        self.RunWeightGradDense('fp16', 'fp32', 1e-4)
    def test_fused_weight_grad_dense_cuda_ibf16_ofp32(self):
        self.RunWeightGradDense('bf16', 'fp32', 1e-4)
    def test_fused_weight_grad_dense_cuda_ifp16_ofp16(self):
        self.RunWeightGradDense('fp16', 'fp16', 1e-3)
    def test_fused_weight_grad_dense_cuda_ibf16_obf16(self):
        self.RunWeightGradDense('bf16', 'bf16', 1e-2)
    

if __name__ == "__main__":
    unittest.main()
