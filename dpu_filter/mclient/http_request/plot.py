import matplotlib.pyplot as plt

# Data
files = ['File A', 'File B', 'File C']
operations = ['Filter on Client', 'Filter on DPU', 'S3Select on Client', 'S3Select on DPU']
times = [
    [9.3, 0.75, 3.05, 2.96],
    [101.8, 3.14, 43.81, 43.3],
    [503.9, 12.67, 157.15, 157.45]
]

# Plot
fig, ax = plt.subplots()
bar_width = 0.2
opacity = 0.8
colors = ['b', 'g', 'r', 'y']

for i, operation in enumerate(operations):
    bars = [times[j][i] for j in range(len(files))]
    plt.bar([x + i * bar_width for x in range(len(files))], bars, bar_width,
            alpha=opacity,
            color=colors[i],
            label=operation)

plt.xlabel('Files')
plt.ylabel('Time (seconds)')
plt.title('Operation Time for Different Files')
plt.xticks([x + 1.5 * bar_width for x in range(len(files))], files)
plt.legend()
plt.tight_layout()
plt.show()

