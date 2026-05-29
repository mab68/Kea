
import numpy as np

def make_hermitian(field_k, N):
    field_k[0] = field_k[0].real
    field_k[N//2:N//2+1] = field_k[N//2:N//2+1].real
    for i in range(N):
        ii = (N - i) % N
        if i <= N//2:
            continue
        elif ii < N:
            field_k[i] = np.conj(field_k[ii])
    return field_k

def make_grid(N, D):
    Ns = [N*np.fft.fftfreq(N) for _ in range(D)]
    mgrid = np.meshgrid(*Ns, indexing='ij')
    rr = np.linalg.norm(mgrid, axis=0)
    return mgrid, rr

def normalize_field(field, var=1.):
    field = field - np.nanmean(field)
    curr_var = np.nanmean(np.abs(field)**2)
    return np.sqrt(var) * field / np.sqrt(curr_var)

def make_mff(kk, L, H, d, noise_k=None, normalize=True, seed=0):
    if seed is not None:
        np.random.seed(seed)
    D = np.float64(kk.ndim)

    if noise_k is None:
        # phases = np.random.uniform(0., 2.*np.pi, np.shape(kk))
        # noise_k = np.cos(phases) + 1j * np.sin(phases)
        noise_k = np.random.randn(*np.shape(kk)) + 1j * np.random.randn(*np.shape(kk))

    filter_k = 1./np.sqrt(kk**2 + L**2)
    omega_k = filter_k**(H + D/2.) * np.exp(-(kk/d)**2) * noise_k
    omega_k[np.isnan(omega_k)|np.isinf(omega_k)] = 0. + 1j * 0.
    if kk.ndim == 1:
        omega_k = make_hermitian(omega_k, len(omega_k))
    omega = np.fft.ifftn(omega_k, norm='backward').real
    if normalize:
        omega = normalize_field(omega)
    return omega

def make_lognormal(field, lamb=0.1):
    log_f = np.exp(field * np.sqrt(lamb))
    return log_f

def make_correlated_noise(field, seed=0):
    if seed is not None:
        np.random.seed(seed)
    n = np.random.normal(0., np.std(field), np.shape(field))
    df = field * n
    return df

def make_multi(kk, Lcross, Hcross, dcross, lamb, Lplus, Hplus, dplus, seeds=(0,1,2), normalize=False):
    field = make_mff(kk, Lplus, Hplus, dplus, normalize=True, seed=seeds[0])
    log_f = make_lognormal(field, lamb)
    corr_n = np.fft.fftshift(np.fft.fftn(make_correlated_noise(log_f, seed=seeds[1])))
    multi_field = make_mff(kk, Lcross, Hcross, dcross, corr_n, seed=seeds[2], normalize=normalize)
    return multi_field
