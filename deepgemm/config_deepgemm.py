import os

class gemm_deepgemm_masked_config:

    @staticmethod
    def get_deepgemm_dict():
        gemm_dict_80 = {
            "EXPECTED_M16": 1002,
            "EXPECTED_M32": 1002,
            "EXPECTED_M64": 1002,
        }

        gemm_dict_72 = {
            "EXPECTED_M16": 1002,
            "EXPECTED_M32": 1002,
            "EXPECTED_M64": 1002,
        }

        gemm_dict_64 = {
            "EXPECTED_M16": 1002,
            "EXPECTED_M32": 1002,
            "EXPECTED_M64": 1002,
        }

        gemm_dict_48 = {
            "EXPECTED_M16": 1002,
            "EXPECTED_M32": 1002,
            "EXPECTED_M64": 1002,
        }
        gpu_cus = int(os.environ.get("DEEPGEMM_GPU_CUS", "80"))

        if gpu_cus == 80:
            gemm_dict = gemm_dict_80
        elif gpu_cus == 72:
            gemm_dict = gemm_dict_72
        elif gpu_cus == 64:
            gemm_dict = gemm_dict_64
        elif gpu_cus == 48:
            gemm_dict = gemm_dict_48
        else:
            raise ValueError(f"Unsupported GPU CUs: {gpu_cus}")
        return gemm_dict

    
    @staticmethod
    def get_deepgemm_block_m(mode: int):
        if mode == 1000:
            return 256
        elif mode == 1001:
            return 128
        elif mode == 1002:
            return 64
        raise ValueError(f"Unsupported asm mode: {mode}")

    @staticmethod
    def get_deepgemm_block_n(mode: int):
        if mode >= 1000 and mode <= 1002:
            return 256
        raise ValueError(f"Unsupported asm mode: {mode}")

