import torch
import torch.optim as optim
from utils.mcmc_sampler import mcmc_step


def train_vmc_model(model_cls, energy_fn, J2_val, config):
    device = config['device']
    n_spins = config['n_spins']
    J1 = config['J1']
    
    model = model_cls(n_spins, depth_1=config['depth_1'], depth_2 = config['depth_2'], kernel_size=config['kernel_size']).to(device)
    optimizer = optim.Adam(model.parameters(), lr=config['lr'])
    spins = torch.randint(0, 2, (config['batch_size'], 1, n_spins), device=device).float() * 2 - 1
    
    mean_hist = []
    std_hist = []
    print(f"Training {model_cls.__name__}...")
    # 250 Epochs
    for epoch in range(1, config['n_epochs'] + 1):
        spins = mcmc_step(spins, model, n_steps=config['mcmc_steps'])
        
        raw_out = model(spins)
        if isinstance(raw_out, tuple): is_complex = True; log_amp = raw_out[0]
        elif raw_out.is_complex(): is_complex = True; log_amp = raw_out.real
        else: is_complex = False; log_amp = raw_out
            
        if log_amp.dim() == 1: log_amp = log_amp.unsqueeze(-1)
            
        E_loc = energy_fn(spins, model, J1, J2_val)
        E_mean = E_loc.mean()
        E_std = E_loc.std()
        
        if is_complex:
            loss = E_loc.mean() + ((E_loc.detach() - E_mean.detach()) * log_amp).mean()
        else:
            loss = ((E_loc - E_mean) * log_amp).mean()

        optimizer.zero_grad()
        loss.backward()
        # Gradient Clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        if epoch % 25 == 0 or epoch == 1:
            print(f"    Epoch {epoch:03d} | <E> = {E_mean.item():.4f}")
        mean_hist.append(E_mean.item())
        std_hist.append(E_std.item())
        
    return mean_hist, std_hist