import torch

def mcmc_step(current_spins, model, n_steps=10):
    device = current_spins.device
    B, _, N = current_spins.shape
    
    def get_log_prob(spins):
        out = model(spins)
        if isinstance(out, tuple): out = out[0]
        if out.is_complex(): out = out.real
        return 2.0 * out.squeeze() # |psi|^2 = exp(2*real)

    with torch.no_grad():
        log_prob = get_log_prob(current_spins)

    for _ in range(n_steps):
        flip_indices = torch.randint(0, N, (B,), device=device)
        proposed_spins = current_spins.clone()
        rows = torch.arange(B, device=device)
        proposed_spins[rows, 0, flip_indices] *= -1

        with torch.no_grad():
            new_log_prob = get_log_prob(proposed_spins)

        accept_log_prob = new_log_prob - log_prob
        rand_log_uniform = torch.log(torch.rand(B, device=device))
        accepted = rand_log_uniform < accept_log_prob
        
        current_spins[accepted] = proposed_spins[accepted]
        log_prob[accepted] = new_log_prob[accepted]

    return current_spins