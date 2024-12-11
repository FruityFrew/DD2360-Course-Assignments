import matplotlib.pyplot as plt
import numpy as np

# Data for float precision
float_sizes = ['1024x1024', '2048x2048', '4096x4096', '8192x8192']
# Data for plotting:
# size,h2d,kernel,d2h
# 1024,1.94,52.71,3.65
# 2048,7.19,74.94,11.66
# 4096,29.57,395.71,46.91
# 8192,113.73,2594.47,187.18
float_h2d = [1.94, 7.19, 29.57, 113.73]
float_kernel = [52.71, 74.94, 395.71, 2594.47]
float_d2h = [3.65, 11.66, 46.91, 187.18]

# Data for double precision
double_sizes = ['1024x1024', '2048x2048', '4096x4096', '8192x8192']
# Data for plotting:
# size,h2d,kernel,d2h
# 1024,3.76,24.11,6.09
# 2048,14.21,169.03,23.20
# 4096,61.09,769.79,95.84
# 8192,235.30,6402.18,374.66
double_h2d = [3.76, 14.21, 61.09, 235.30]
double_kernel = [24.11, 169.03, 769.79, 6402.18]
double_d2h = [6.09, 23.20, 95.84, 374.66]

# Create figure with two subplots
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6, 8))
fig.suptitle('Matrix Multiplication Timing Analysis')

# Width of each bar
width = 0.35

# Positions for bars
positions = np.arange(len(float_sizes))

# Plot for float precision
ax1.bar(positions, float_h2d, width, label='Host to Device', color='#8884d8')
ax1.bar(positions, float_kernel, width, bottom=float_h2d, label='Kernel Execution', color='#82ca9d')
float_bottom = np.array(float_h2d) + np.array(float_kernel)
ax1.bar(positions, float_d2h, width, bottom=float_bottom, label='Device to Host', color='#ffc658')

ax1.set_title('Float Precision')
ax1.set_xticks(positions)
ax1.set_xticklabels(float_sizes)
ax1.set_ylabel('Time (ms)')
ax1.legend()

# Add percentage labels
for i in range(len(float_sizes)):
    total = float_h2d[i] + float_kernel[i] + float_d2h[i]
    h2d_pct = float_h2d[i] / total * 100
    kernel_pct = float_kernel[i] / total * 100
    d2h_pct = float_d2h[i] / total * 100
    
    # Only show percentages for portions > 5%
    # if h2d_pct > 5:
    #     ax1.text(i, float_h2d[i]/2, f'{h2d_pct:.1f}%', ha='center', va='center')
    # if kernel_pct > 5:
    #     ax1.text(i, float_h2d[i] + float_kernel[i]/2, f'{kernel_pct:.1f}%', ha='center', va='center')
    # if d2h_pct > 5:
    #     ax1.text(i, float_bottom[i] + float_d2h[i]/2, f'{d2h_pct:.1f}%', ha='center', va='center')

# Plot for double precision
ax2.bar(positions, double_h2d, width, label='Host to Device', color='#8884d8')
ax2.bar(positions, double_kernel, width, bottom=double_h2d, label='Kernel Execution', color='#82ca9d')
double_bottom = np.array(double_h2d) + np.array(double_kernel)
ax2.bar(positions, double_d2h, width, bottom=double_bottom, label='Device to Host', color='#ffc658')

ax2.set_title('Double Precision')
ax2.set_xticks(positions)
ax2.set_xticklabels(double_sizes)
ax2.set_ylabel('Time (ms)')
ax2.legend()

# Add percentage labels
for i in range(len(double_sizes)):
    total = double_h2d[i] + double_kernel[i] + double_d2h[i]
    h2d_pct = double_h2d[i] / total * 100
    kernel_pct = double_kernel[i] / total * 100
    d2h_pct = double_d2h[i] / total * 100
    
    # Only show percentages for portions > 5%
    # if h2d_pct > 5:
    #     ax2.text(i, double_h2d[i]/2, f'{h2d_pct:.1f}%', ha='center', va='center')
    # if kernel_pct > 5:
    #     ax2.text(i, double_h2d[i] + double_kernel[i]/2, f'{kernel_pct:.1f}%', ha='center', va='center')
    # if d2h_pct > 5:
    #     ax2.text(i, double_bottom[i] + double_d2h[i]/2, f'{d2h_pct:.1f}%', ha='center', va='center')

plt.tight_layout()
plt.show()