import matplotlib.pyplot as plt
import numpy as np

# Data processing
def process_data():
    # Vector length = 1M
    s_seg_1m = [1048576, 524288, 262144, 131072, 65536, 32768, 16384, 8192, 4096]
    streamed_1m = [1.359, 1.150, 1.072, 1.080, 1.177, 1.360, 1.664, 2.198, 3.453]
    non_streamed_1m = [1.588] * len(s_seg_1m)

    # Vector length = 2M
    s_seg_2m = [2097152, 1048576, 524288, 262144, 131072, 65536, 32768, 16384, 8192]
    streamed_2m = [2.602, 2.224, 2.082, 2.024, 2.128, 2.328, 2.747, 3.396, 4.497]
    non_streamed_2m = [2.643] * len(s_seg_2m)

    # Vector length = 4M
    s_seg_4m = [4194304, 2097152, 1048576, 524288, 262144, 131072, 65536, 32768, 16384]
    streamed_4m = [5.278, 4.545, 4.033, 3.932, 3.976, 4.296, 4.888, 5.614, 6.758]
    non_streamed_4m = [5.303] * len(s_seg_4m)

    # Vector length = 8M
    s_seg_8m = [8388608, 4194304, 2097152, 1048576, 524288, 262144, 131072, 65536, 32768]
    streamed_8m = [10.074, 8.418, 7.769, 7.482, 7.543, 7.942, 8.626, 9.517, 10.902]
    non_streamed_8m = [10.111] * len(s_seg_8m)

    return {
        '1M': (s_seg_1m, streamed_1m, non_streamed_1m),
        '2M': (s_seg_2m, streamed_2m, non_streamed_2m),
        '4M': (s_seg_4m, streamed_4m, non_streamed_4m),
        '8M': (s_seg_8m, streamed_8m, non_streamed_8m)
    }

def plot_performance():
    data = process_data()
    plt.figure(figsize=(12, 8))

    colors = ['blue', 'green', 'red', 'purple']
    markers = ['o', 's', '^', 'D']
    
    for i, (vector_size, (s_seg, streamed, non_streamed)) in enumerate(data.items()):
        # Convert s_seg to KB for better readability
        s_seg_kb = [size/1024 for size in s_seg]
        
        # Plot streamed version
        plt.plot(s_seg_kb, streamed, 
                label=f'Streamed ({vector_size})', 
                color=colors[i], 
                marker=markers[i])
        
        # Plot non-streamed version (constant line)
        plt.plot(s_seg_kb, non_streamed, 
                label=f'Non-Streamed ({vector_size})', 
                color=colors[i], 
                linestyle='--')

    plt.xscale('log')  # Use log scale for x-axis
    plt.grid(True)
    plt.xlabel('Segment Size (K)')
    plt.ylabel('Execution Time (ms)')
    plt.title('Vector Addition Performance Comparison')
    plt.legend(bbox_to_anchor=(1.05, 1))
    
    # Add gridlines
    plt.grid(True, which="both", ls="-", alpha=0.2)
    
    # Adjust layout to prevent label cutoff
    plt.tight_layout()
    
    # Save the plot
    plt.savefig('vector_addition_performance.png', bbox_inches='tight', dpi=300)
    plt.show()

if __name__ == "__main__":
    plot_performance()