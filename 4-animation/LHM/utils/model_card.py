MODEL_CARD = {
    "prior_model": "https://virutalbuy-public.oss-cn-hangzhou.aliyuncs.com/share/aigc3d/data/LHM/LHM_prior_model.tar",
}
ModelScope_MODEL_CARD = {
    "LHM-MINI": 'Damo_XR_Lab/LHM-MINI',
    "LHM-500M": 'Damo_XR_Lab/LHM-500M',
    "LHM-500M-HF": 'Damo_XR_Lab/LHM-500M-HF',
    "LHM-1B": 'Damo_XR_Lab/LHM-1B',
    "LHM-1B-HF": 'Damo_XR_Lab/LHM-1B-HF',
}
HuggingFace_MODEL_CARD = {
    "LHM-MINI": '3DAIGC/LHM-MINI',
    "LHM-500M": '3DAIGC/LHM-500M',
    "LHM-500M-HF": '3DAIGC/LHM-500M-HF',
    "LHM-1B": '3DAIGC/LHM-1B',
    "LHM-1B-HF": '3DAIGC/LHM-1B-HF',
}
MODEL_CONFIG={
    '1B': "./configs/inference/human-lrm-1B.yaml",
    '500M': "./configs/inference/human-lrm-500M.yaml",
    'MINI': "./configs/inference/human-lrm-mini.yaml",
}

MEMORY_MODEL_CARD={
    "LHM-MINI": 16000,  # 16G
    "LHM-500M": 18000,
    "LHM-500M-HF": 18000,
    "LHM-1B": 22000,
    "LHM-1B-HF": 22000,
}