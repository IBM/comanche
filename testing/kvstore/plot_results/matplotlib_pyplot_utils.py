import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import numpy as np
import os


def get_graph_cutoff(values):
    mean_val, stddev_val = np.mean(np.asarray(values)), np.std(np.asarray(values))
    stddev_multiplier = 1.0

    cutoff_val = mean_val + (stddev_val * stddev_multiplier)
    max_val = np.asarray(values).max()

    return cutoff_val, max_val


def limit_visibility_on_graph(x_values, y_values, plot_x_divisions=10):
    axes = plt.gca()  # gca = get current axes

    # set x scale to exclude most outliers
    max_visible_x_axis, max_x = get_graph_cutoff(x_values)
    if len(x_values) > 0 and np.asarray(x_values).max() > max_visible_x_axis:
        axes.set_xlim(-10, max_visible_x_axis)

        x_labels = []
        x_locations = []

        # TODO: feature creep. Move this elsewhere
        for i in range(0, int(max_visible_x_axis), int(max_visible_x_axis / plot_x_divisions)):
            x_labels.append("%.2f" % (i / (10 ** 6)))
            x_locations.append(i)

        plt.xticks(x_locations, x_labels)

    # set y scale to exclude most outliers
    y_cutoff, max_y = get_graph_cutoff(y_values)
    max_visible_y_axis = min(y_cutoff, 3000)  # anything higher and it's difficult to see network bottleneck

    if len(y_values) > 0 and np.asarray(y_values).max() > max_visible_y_axis:
        axes.set_ylim([-10, max_visible_y_axis])  # -10 so zero-valued entries aren't clipped in half

    return max_visible_x_axis, max_visible_y_axis, max_x, max_y


def set_graph_limits(x_cutoff, y_cutoff):
    axes = plt.gca()  # gca = get current axes

    axes.set_xlim([-10, x_cutoff])
    axes.set_ylim([-10, y_cutoff])  # -10 so zero-valued entries aren't clipped in half


def create_directory_if_missing(directory):
    if not os.path.exists(directory):
        os.mkdir(directory)


# credit: https://stackoverflow.com/questions/12523586/python-format-size-application-converting-b-to-kb-mb-gb-tb
def bytes_2_human_readable(number_of_bytes, skip_lowest=False):
    if number_of_bytes < 0:
        # raise ValueError("!!! number_of_bytes can't be smaller than 0 !!! Got %s!" % number_of_bytes)
        pass

    step_to_greater_unit = 1024.

    number_of_bytes = float(number_of_bytes)
    if not skip_lowest:
        unit = 'bytes'
    else:
        unit = ""

    if (number_of_bytes / step_to_greater_unit) >= 1:
        number_of_bytes /= step_to_greater_unit
        unit = 'KB'

    if (number_of_bytes / step_to_greater_unit) >= 1:
        number_of_bytes /= step_to_greater_unit
        unit = 'MB'

    if (number_of_bytes / step_to_greater_unit) >= 1:
        number_of_bytes /= step_to_greater_unit
        unit = 'GB'

    if (number_of_bytes / step_to_greater_unit) >= 1:
        number_of_bytes /= step_to_greater_unit
        unit = 'TB'

    precision = 1
    number_of_bytes = round(number_of_bytes, precision)

    return str(number_of_bytes) + ' ' + unit


def size_2_human_readable(count):
    if count < 0:
        raise ValueError("size_2_human_readable: count should never be negative!")

    count = float(count)
    unit = ""

    if (count / 1000) >= 1:
        count /= 1000
        unit = 'k'

        if (count / 1000) >= 1:
            count /= 1000
            unit = 'M'

    return str(count) + " " + unit


def ms_to_human_readable(time_in_ms):
    if time_in_ms < 0:
        raise ValueError("time_in_ms must be positive!")

    time_amount = float(time_in_ms)
    units = 'ms'

    if (time_amount / 1000) >= 1:
        time_amount /= 1000
        units = 's'

        if (time_amount / 60) >= 1:
            time_amount /= 60
            units = 'min'

            if (time_amount / 60) >= 1:
                time_amount /= 60
                units = 'hour'

                if (time_amount / 24) >= 1:
                    time_amount /= 24
                    units = 'days'

    time_description = "%.2f %s" % (time_amount, units)

    return time_description


def fast_time_to_human_readable(time_amount):
    units = 's'
    si_units = ['ms', 'us', 'ns', 'ps', 'fs', 'as', 'zs', 'ys']
    factor = 1000

    i = 0
    can_decrement = time_amount < 1.0
    while can_decrement:
        time_amount = time_amount * factor
        units = si_units.pop(0)
        can_decrement = time_amount < 1.0 and len(si_units) > 0

    return "%.0f %s" % (time_amount, units)

# converts seconds to human readable unit of time with either 2 or 0 units precision
def seconds_to_human_readable(time_amount):
    long_time = [{'units': 's', 'duration': 1},
                 {'units': 'min', 'duration': 60},
                 {'units': 'hr', 'duration': 60},
                 {'units': 'dy', 'duration': 24},
                 {'units': 'wk', 'duration': 7},
                 {'units': 'yr', 'duration': 52}]

    if time_amount >= 1.0:
        units = 's'  # default

        while len(long_time) > 0:
            time_info = long_time.pop(0)
            temp_units, duration = time_info['units'], time_info['duration']
            temp_time = time_amount / duration
            if temp_time > 1.0:
                time_amount = temp_time
                units = temp_units
            else:
                break

        time_description = "%.2f %s" % (time_amount, units)
    elif time_amount > 0.0:
        time_description = fast_time_to_human_readable(time_amount)
    elif time_amount == 0.0:
        time_description = "0 s"
    else:
        raise ValueError("time_amount should always be positive!")

    return time_description

def save_plot(filename, label, dpi=80, clear_plot=True, plot_path="./figures"):
    create_directory_if_missing(plot_path)
    create_directory_if_missing(plot_path + "/%s" % filename)

    full_path = plot_path + "/" + filename + "/" + label
    plt.savefig(full_path, dpi=dpi)
    print("saved figure: %s.png" % full_path)

    if clear_plot:
        plt.gca().clear()
        plt.yscale('linear')  # just reset this in case we use some non-linear scale
        plt.xscale('linear')
        plt.clf()
        plt.cla()
        plt.close()


def set_axis_tick_color(color):
    axes = plt.gca()

    axes.xaxis.label.set_color(color)
    axes.yaxis.label.set_color(color)
    axes.spines['bottom'].set_color(color)
    axes.spines['top'].set_color(color)
    axes.spines['left'].set_color(color)
    axes.spines['right'].set_color(color)
    axes.tick_params(axis='x', colors=color, which='both')
    axes.tick_params(axis='y', colors=color, which='both')


def set_axis_tick_labels_to_color(color):
    # set y tick labels to be the same color as the rest of the plot
    axes = plt.gca()
    [i.set_color(color) for i in axes.get_xticklabels()]
    [j.set_color(color) for j in axes.get_yticklabels()]


def bits_2_human_readable(number_of_bits):
    return bytes_2_human_readable(number_of_bits / 8.0)


def formatter_bytes_2_human_readable(number_of_bytes, pos):
    return bytes_2_human_readable(number_of_bytes, True)


def formatter_group_long_numbers(x, pos):
    return "{:,}".format(int(x))


def formatter_seconds_to_human_readable(x, pos):
    return seconds_to_human_readable(x)


def set_log_x_axis_to_human_readable_sizes():
    plt.gca().xaxis.set_major_formatter(FuncFormatter(formatter_bytes_2_human_readable))


def set_log_y_axis_to_full_numbers():
    plt.gca().yaxis.set_major_formatter(FuncFormatter(formatter_group_long_numbers))


def set_legend_text_color(legend, color):
    for text in legend.get_texts():
        plt.setp(text, color=color)


def set_log_y_axis_to_human_readable_times():
    plt.gca().yaxis.set_major_formatter(FuncFormatter(formatter_seconds_to_human_readable))

def remove_borders_from_plot():
    # get rid of borders
    axes = plt.gca()
    for region in ['top', 'right', 'left', 'bottom']:
        axes.spines[region].set_visible(False)


def remove_axis_ticks(x_axis_labels_visible=True, y_axis_labels_visible=True):
    # get rid of ticks on x and y axis
    plt.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=x_axis_labels_visible)
    plt.tick_params(axis='y', which='both', left=False, right=False, labelleft=y_axis_labels_visible)
