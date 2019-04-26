import sys
import plot_results

if len(sys.argv) == 1:
    print("usage: $ python plot_everything_in_file.py <your_json_results_file_here>")
else:
    filenames = sys.argv[1:]
    for filename in filenames:
        filename_without_path = filename.split("/")[-1].split(".")[0]
        plot = plot_results.Plot(filename)
        plot.plot_everything_for_valid_experiments(filename_suffix="_" + filename_without_path)
