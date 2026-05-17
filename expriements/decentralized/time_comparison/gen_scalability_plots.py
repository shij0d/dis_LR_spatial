"""Combined scalability table + figure for m=50 and m=100."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

Js = np.array([1, 2, 4, 8, 16, 28])
N = 80000

# ── Data ─────────────────────────────────────────────────────────────
m50_fp64_w = [3.79, 2.01, 1.23, 0.92, 0.71, 0.36]
m50_fp64_e = [100, 94.4, 76.8, 51.6, 33.5, 37.8]
m50_fp32_w = [15.19, 7.80, 4.22, 2.25, 1.22, 0.76]
m50_fp32_e = [100, 97.4, 90.0, 84.3, 78.1, 71.3]
m50_fp64_s = [m50_fp64_w[0]/w for w in m50_fp64_w]
m50_fp32_s = [m50_fp32_w[0]/w for w in m50_fp32_w]

m100_fp64_w = [9.56, 5.14, 2.98, 1.99, 1.60, 0.92]
m100_fp64_e = [100, 93.1, 80.2, 60.1, 37.5, 37.2]
m100_fp32_w = [73.90, 37.95, 20.51, 10.81, 5.55, 3.36]
m100_fp32_e = [100, 97.4, 90.1, 85.4, 83.2, 78.6]
m100_fp64_s = [m100_fp64_w[0]/w for w in m100_fp64_w]
m100_fp32_s = [m100_fp32_w[0]/w for w in m100_fp32_w]

out_dir = '/home/shij0d/documents/dis_LR_spatial/expriements/decentralized/time_comparison'

# ── TSV table ─────────────────────────────────────────────────────────
with open(f'{out_dir}/scalability_table_m50_m100.tsv', 'w') as f:
    f.write("J\tN_j\t"
            "FP64 m=50 (s)\tFP64 m=50 spdup\tFP64 m=50 eff%\t"
            "FP32 m=50 (s)\tFP32 m=50 spdup\tFP32 m=50 eff%\t"
            "FP64 m=100 (s)\tFP64 m=100 spdup\tFP64 m=100 eff%\t"
            "FP32 m=100 (s)\tFP32 m=100 spdup\tFP32 m=100 eff%\n")
    for i, J in enumerate(Js):
        nj = N // J
        f.write(f"{J}\t{nj}\t"
                f"{m50_fp64_w[i]:.2f}\t{m50_fp64_s[i]:.2f}x\t{m50_fp64_e[i]:.1f}%\t"
                f"{m50_fp32_w[i]:.2f}\t{m50_fp32_s[i]:.2f}x\t{m50_fp32_e[i]:.1f}%\t"
                f"{m100_fp64_w[i]:.2f}\t{m100_fp64_s[i]:.2f}x\t{m100_fp64_e[i]:.1f}%\t"
                f"{m100_fp32_w[i]:.2f}\t{m100_fp32_s[i]:.2f}x\t{m100_fp32_e[i]:.1f}%\n")
print("TSV written")

# ── Figure: 2×2 grid ──────────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(13, 10))

# Panel (a): Wall time m=50
ax = axes[0, 0]
ax.loglog(Js, m50_fp64_w, 'o-', color='#2171b5', lw=2, ms=8, label='FP64')
ax.loglog(Js, m50_fp32_w, 's--', color='#d94801', lw=2, ms=8, label='FP32')
ideal50 = [m50_fp64_w[0]/j for j in Js]
ax.loglog(Js, ideal50, ':', color='gray', lw=1, label='Ideal')
ax.set_title('Wall time, m = 50', fontsize=12, fontweight='bold')
ax.set_xlabel('J'); ax.set_ylabel('Wall time (s)')
ax.legend(fontsize=9); ax.grid(True, alpha=0.3)
ax.set_xticks(Js); ax.set_xticklabels([str(j) for j in Js])

# Panel (b): Wall time m=100
ax = axes[0, 1]
ax.loglog(Js, m100_fp64_w, 'o-', color='#2171b5', lw=2, ms=8, label='FP64')
ax.loglog(Js, m100_fp32_w, 's--', color='#d94801', lw=2, ms=8, label='FP32')
ideal100 = [m100_fp64_w[0]/j for j in Js]
ax.loglog(Js, ideal100, ':', color='gray', lw=1, label='Ideal')
ax.set_title('Wall time, m = 100', fontsize=12, fontweight='bold')
ax.set_xlabel('J'); ax.set_ylabel('Wall time (s)')
ax.legend(fontsize=9); ax.grid(True, alpha=0.3)
ax.set_xticks(Js); ax.set_xticklabels([str(j) for j in Js])

# Panel (c): Efficiency m=50
ax = axes[1, 0]
ax.plot(Js, m50_fp64_e, 'o-', color='#2171b5', lw=2, ms=8, label='FP64')
ax.plot(Js, m50_fp32_e, 's--', color='#d94801', lw=2, ms=8, label='FP32')
ax.axhline(100, color='gray', ls=':', lw=1)
ax.set_title('Parallel efficiency, m = 50', fontsize=12, fontweight='bold')
ax.set_xlabel('J'); ax.set_ylabel('Efficiency (%)')
ax.legend(fontsize=9); ax.grid(True, alpha=0.3); ax.set_ylim(0, 110)
ax.set_xticks(Js); ax.set_xticklabels([str(j) for j in Js])

# Panel (d): Efficiency m=100
ax = axes[1, 1]
ax.plot(Js, m100_fp64_e, 'o-', color='#2171b5', lw=2, ms=8, label='FP64')
ax.plot(Js, m100_fp32_e, 's--', color='#d94801', lw=2, ms=8, label='FP32')
ax.axhline(100, color='gray', ls=':', lw=1)
ax.set_title('Parallel efficiency, m = 100', fontsize=12, fontweight='bold')
ax.set_xlabel('J'); ax.set_ylabel('Efficiency (%)')
ax.legend(fontsize=9); ax.grid(True, alpha=0.3); ax.set_ylim(0, 110)
ax.set_xticks(Js); ax.set_xticklabels([str(j) for j in Js])

fig.suptitle(f'C++/OpenMP ce_optimize_stage2 Scalability\n'
             f'N = {N:,}   T = 3   S_max = 5\n'
             f'2× Xeon E5-2680 v4 (28 cores, 2 NUMA sockets)',
             fontsize=11, style='italic')
plt.tight_layout()
plt.savefig(f'{out_dir}/scalability_figure_m50_m100.png', dpi=200, bbox_inches='tight')
plt.savefig(f'{out_dir}/scalability_figure_m50_m100.pdf', bbox_inches='tight')
print("Figure saved")
