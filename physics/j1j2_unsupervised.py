import torch

def compute_local_energy_log(spins, model, J1, J2):
    """ Used for Real-valued models. Applies Marshall Sign Rule manually. """
    N = spins.shape[-1]
    B = spins.shape[0]
    
    log_psi = model(spins)
    E_diag = torch.zeros(B, 1, device=spins.device)
    E_off = torch.zeros(B, 1, device=spins.device)
    
    # Diagonal
    interactions = [(1, J1), (2, J2)]
    for i in range(N):
        s_i = spins[:, 0, i]
        for dist, J in interactions:
            j = i + dist
            if j < N:
                s_j = spins[:, 0, j]
                E_diag[:, 0] += (J / 4.0) * s_i * s_j

    # Off-Diagonal (Sign Rule Enforced)
    bonds = []
    for i in range(N-1): bonds.append((i, i+1, J1, True))   
    for i in range(N-2): bonds.append((i, i+2, J2, False))
        
    for (i, j, J, is_nn) in bonds:
        s_i = spins[:, 0, i]
        s_j = spins[:, 0, j]
        diff_mask = (s_i != s_j)
        
        if diff_mask.any():
            spins_prime = spins.clone()
            spins_prime[:, 0, i] = spins[:, 0, j]
            spins_prime[:, 0, j] = spins[:, 0, i]
            
            with torch.no_grad():
                log_psi_prime = model(spins_prime)
            
            ratio = torch.exp(log_psi_prime - log_psi)
            sign_fix = -1.0 if is_nn else 1.0 # Marshall Sign Rule Hard-Coded
            term = (J / 2.0) * ratio * sign_fix
            E_off += term * diff_mask.unsqueeze(1).float()

    return E_diag + E_off

def compute_energy_complex_robust(spins, model, J1, J2):
    N = spins.shape[-1]
    B = spins.shape[0]
    
    def get_components(x):
        out = model(x)
        if isinstance(out, tuple): 
            log_amp, phase = out[0], out[1]
        elif out.is_complex(): 
            log_amp, phase = out.real, out.imag
        else: 
            log_amp, phase = out, torch.zeros_like(out)
        
        if log_amp.dim() == 1: log_amp = log_amp.unsqueeze(-1)
        if phase.dim() == 1: phase = phase.unsqueeze(-1)
        return log_amp, phase

    log_amp_s, phase_s = get_components(spins)
    E_diag = torch.zeros(B, 1, device=spins.device)
    E_off = torch.zeros(B, 1, device=spins.device)
    
    # Diagonal
    interactions = [(1, J1), (2, J2)]
    for i in range(N):
        s_i = spins[:, 0, i]
        for dist, J in interactions:
            j = i + dist
            if j < N:
                s_j = spins[:, 0, j]
                E_diag[:, 0] += (J / 4.0) * s_i * s_j

    # Off-Diagonal
    bonds = []
    for i in range(N-1): bonds.append((i, i+1, J1))
    for i in range(N-2): bonds.append((i, i+2, J2))
        
    for (i, j, J) in bonds:
        s_i = spins[:, 0, i]
        s_j = spins[:, 0, j]
        diff_mask = (s_i != s_j)
        if diff_mask.any():
            spins_prime = spins.clone()
            spins_prime[:, 0, i] = spins[:, 0, j]
            spins_prime[:, 0, j] = spins[:, 0, i]
            
            log_amp_prime, phase_prime = get_components(spins_prime)
            
            # clamp exponent to avoid overflow
            log_diff = log_amp_prime - log_amp_s
            log_diff = torch.clamp(log_diff, max=10.0) 
            
            mag_ratio = torch.exp(log_diff)
            phase_diff = phase_prime - phase_s
            
            term = (J / 2.0) * mag_ratio * torch.cos(phase_diff)
            E_off += term * diff_mask.unsqueeze(1).float()

    return E_diag + E_off