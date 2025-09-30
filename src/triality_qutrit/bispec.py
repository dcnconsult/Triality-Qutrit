
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def _segment_signal(x: np.ndarray, nperseg: int, noverlap: int):
    step = nperseg - noverlap
    if step <= 0:
        raise ValueError("noverlap must be < nperseg")
    nwin = 1 + (len(x) - nperseg) // step
    if nwin <= 0:
        raise ValueError("Signal shorter than nperseg.")
    windows = []
    w = np.hanning(nperseg)
    for k in range(nwin):
        s = k * step
        windows.append(x[s:s+nperseg] * w)
    return np.stack(windows, axis=0)

def bicoherence(x: np.ndarray, fs: float, nperseg: int=4096, noverlap: int=2048, nfft: int=None):
    if nfft is None:
        nfft = int(2**np.ceil(np.log2(nperseg)))
    Xw = _segment_signal(x, nperseg, noverlap)  # [nwin, nperseg]
    F = np.fft.rfft(Xw, n=nfft, axis=1)         # [nwin, nfreq]
    nwin, nfreq = F.shape

    B = np.zeros((nfreq, nfreq), dtype=np.complex128)
    P12 = np.zeros_like(B, dtype=np.float64)
    P3 = np.zeros_like(B, dtype=np.float64)

    for f1 in range(nfreq):
        for f2 in range(nfreq - f1):
            prod = F[:, f1] * F[:, f2] * np.conj(F[:, f1+f2])
            B[f1, f2] = prod.mean()
            P12[f1, f2] = (np.abs(F[:, f1] * F[:, f2])**2).mean()
            P3[f1, f2] = (np.abs(F[:, f1+f2])**2).mean()

    b2 = (np.abs(B)**2) / (P12 * P3 + 1e-15)
    freqs = np.fft.rfftfreq(nfft, d=1.0/fs)
    return freqs, b2

def main():
    ap = argparse.ArgumentParser(description="Compute bicoherence for RF timeseries CSV.")
    ap.add_argument('--in', dest='inp', required=True, help='CSV with columns t_s,x OR x only')
    ap.add_argument('--fs', type=float, default=None, help='Sample rate Hz (required if no t_s)')
    ap.add_argument('--nperseg', type=int, default=4096)
    ap.add_argument('--noverlap', type=int, default=2048)
    ap.add_argument('--nfft', type=int, default=None)
    ap.add_argument('--outdir', default='out/bispec', help='Output directory')
    ap.add_argument('--peak-topk', type=int, default=20, help='Save top-K peaks')
    args = ap.parse_args()

    df = pd.read_csv(args.inp)
    if 't_s' in df.columns and 'x' in df.columns and args.fs is None:
        t = df['t_s'].values
        if len(t) < 2:
            raise ValueError("Not enough samples to infer fs.")
        fs = 1.0 / np.median(np.diff(t))
        x = df['x'].values
    elif 'x' in df.columns and args.fs is not None:
        fs = args.fs
        x = df['x'].values
    else:
        raise ValueError("Provide either (t_s,x) or (x with --fs).")

    freqs, b2 = bicoherence(x, fs, nperseg=args.nperseg, noverlap=args.noverlap, nfft=args.nfft)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    np.save(outdir / 'bicoherence.npy', b2)
    np.save(outdir / 'freqs.npy', freqs)

    tri = np.triu(b2)
    flat_idx = np.argsort(tri, axis=None)[::-1][:args.peak_topk]
    f1_idx, f2_idx = np.unravel_index(flat_idx, tri.shape)
    rows = []
    for i in range(len(f1_idx)):
        f1 = freqs[f1_idx[i]]
        f2 = freqs[f2_idx[i]]
        f3 = f1 + f2
        rows.append(dict(rank=i+1, f1_Hz=float(f1), f2_Hz=float(f2), f3_Hz=float(f3), b2=float(tri[f1_idx[i], f2_idx[i]])))
    pd.DataFrame(rows).to_csv(outdir / 'bicoherence_peaks.csv', index=False)

    fig = plt.figure(figsize=(7,6))
    plt.imshow(tri, origin='lower', aspect='auto',
               extent=[freqs.min(), freqs.max(), freqs.min(), freqs.max()])
    plt.xlabel('f1 [Hz]'); plt.ylabel('f2 [Hz]'); plt.title('Normalized squared bicoherence')
    cbar = plt.colorbar(); cbar.set_label('b^2')
    for r in rows[:10]:
        plt.plot(r['f1_Hz'], r['f2_Hz'], 'o', markersize=3)
    fig.savefig(outdir / 'bicoherence.png', dpi=150, bbox_inches='tight')
    print(f"Saved bicoherence outputs to {outdir}")

if __name__ == '__main__':
    main()
