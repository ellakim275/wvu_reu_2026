# cases/case_solver.py

from cases import r1_cd_r3, r1_cd_s3, s1_cd_r3, s1_cd_s3, delta

_CASE_MAP = {
    1: r1_cd_r3,
    2: r1_cd_s3,
    3: s1_cd_r3,
    4: s1_cd_s3,
    5: delta,
    6: delta,   
}

def get_intermediates(cfg):
    module = _CASE_MAP.get(cfg.case_num)
    if module is None:
        raise ValueError(f"Unknown case_num={cfg.case_num}")
    return module.compute_intermediates(cfg)

def get_wave_curves(cfg, intermediates, rho_range):
    module = _CASE_MAP.get(cfg.case_num)
    return module.wave_curves(cfg, intermediates, rho_range)