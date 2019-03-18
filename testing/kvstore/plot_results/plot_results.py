
import json
import matplotlib.pyplot as plt
import numpy as np
import matplotlib_pyplot_utils as utils

class Plot:
    def __init__(self, filename, title_note=""):
        self.filename = filename
        self.title_note = title_note

        # list of parameters used in plot to read from json file
        self.key_environment = "experiment"
        self.key_component = "component"
        self.key_cores = "cores"
        self.key_pool_size = "pool_size"
        self.key_element_count = "elements"
        self.key_length_key = "key_length"
        self.key_length_value = "value_length"

        # file output info
        self.plot_path = "./figures"
        self.results = {}

        self.figure_dpi = 600
        self.figure_background = 'gray'
        self.figure_foreground = 'black'

    '''
    experiment: put_latency, get_latency, get_direct_latency, etc
    results_type: latency, start_time
    filename: json filename, with extension
    '''
    def get_experiment_results_from_filename(self, experiment, results_type, filename):
        if filename in self.results and experiment in self.results[filename] and results_type in self.results[filename][experiment]:
            return self.results[filename][results_type][experiment]
        else:
            counts = {}
            mins = {}
            maxs = {}
            means = {}
            stds = {}

            try:
                with open(self.filename) as file:
                    json_contents = json.load(file)

                    # get environment info for experiment
                    self.validate_json_info(json_contents)

                    # retrieve results
                    experiment_info = json_contents[self.key_environment]

                    try:
                        results = json_contents[experiment]
                    except KeyError:
                        raise KeyError("plot_latency_histogram found no experiment key with name '%s'" % experiment)

                    for core in results:
                        counts[core] = []
                        mins[core] = []
                        maxs[core] = []
                        means[core] = []
                        stds[core] = []

                        try:
                            bin_info = results[core][results_type]['info']
                        except KeyError:
                            raise KeyError("missing latency info key. ")

                        try:
                            for current_bin in results[core][results_type]['bins']:
                                counts[core].append(current_bin['count'])
                                mins[core].append(current_bin['min'])
                                maxs[core].append(current_bin['max'])
                                means[core].append(current_bin['mean'])
                                stds[core].append(current_bin['std'])

                        except KeyError:
                            raise KeyError("missing bins key for '%s' experiment, core %s" % (experiment, core))

                    info = {}

                    info['counts'] = counts
                    info['mins'] = mins
                    info['maxs'] = maxs
                    info['means'] = means
                    info['stds'] = stds
                    info['info'] = bin_info
                    info['setup'] = experiment_info

                    if filename not in self.results:
                        self.results[filename] = {}

                    if results_type not in self.results[filename]:
                        self.results[filename][results_type] = {}

                    self.results[filename][results_type][experiment] = info

            except IOError:
                raise IOError("tried to open file '%s' but failed. Exiting." % self.filename)

            return self.results[filename][results_type][experiment]

    '''
    experiment: put_latency, get_latency, get_direct_latency, etc
    results_type: latency or start_time
    '''
    def plot_latency_histogram(self, experiment, results_type="latency", filename_suffix="", bin_count=50):
        results = self.get_experiment_results_from_filename(experiment, results_type, self.filename)
        core_string = results['setup'][self.key_cores]
        counts = results['counts']
        bin_count_raw = results['info']['bin_count']

        bin_count = bin_count_raw # TODO: group if these don't match

        # get limits for x and y
        max_counts = []
        core_list = self.core_string_to_list(core_string)
        for core in core_list:
            max_counts += counts[str(core)]

        max_count = np.asarray(max_counts).max()
        max_count_digits = len(str(int(max_count)))

        bin_info = results['info']
        x_values, x_labels = self._get_labels_from_bin_info(bin_info['threshold_min'], bin_info['increment'], bin_count, change_last_bucket=True)
        y_values, y_labels = self._get_labels_from_bin_info(0, 0, 100, tick_count=max_count_digits+1, print_func=self.print_power_of_ten, scale='log')

        for core in core_list:
            subplot_axes = plt.subplot(len(core_list), 1, core_list.index(core) + 1)

            if core == core_list[0]:  # first loop
                plt.title("%s: %s - Latency Frequencies %s" % (results['setup'][self.key_component].capitalize(), experiment, self.title_note), color=self.figure_background, loc='left')
                text_experiment_info = "Pool size: %s \nElements: %s \nKey length: %s\nValue length: %s" % (
                    utils.bytes_2_human_readable(results['setup'][self.key_pool_size]),
                    utils.size_2_human_readable(results['setup'][self.key_element_count]),
                    results['setup'][self.key_length_key],
                    results['setup'][self.key_length_value])
                plt.figtext(x=0.75, y=0.78, s=text_experiment_info, fontsize=6, color=self.figure_background)

            if core == core_list[len(core_list) - 1]:  # last loop
                plt.xlabel("Latency bins (%s total)" % bin_count, horizontalalignment='left', x=0.0)

            plt.bar(np.arange(bin_count_raw), counts[str(core)], color=self.figure_background)
            plt.ylabel("Count (Core %s)" % core, color=self.figure_background)

            subplot_axes.set_yscale('log')
            plt.yticks(y_values, y_labels, fontsize=7, color=self.figure_background)
            subplot_axes.set_ylim((0, 10 ** (max_count_digits)))
            subplot_axes.set_yticklabels(y_labels, color=self.figure_background)
            #plt.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))

            plt.xticks(x_values, x_labels, fontsize=7)

            # make graph pretty
            utils.remove_borders_from_plot()
            utils.set_axis_tick_color(self.figure_background)
            utils.set_axis_tick_labels_to_color(self.figure_background)

        # save figure
        utils.save_plot(results['setup'][self.key_component], "%s_%s%s" % (experiment, results_type, filename_suffix), dpi=self.figure_dpi)

    '''
    experiment: put_latency, get_latency, get_direct_latency, etc
    results_type: latency or start_time
    '''
    def plot_latency_over_time(self, experiment, results_type="start_time", filename_suffix = "", bin_count=50):
        results = self.get_experiment_results_from_filename(experiment, results_type, self.filename)
        cores = results['setup'][self.key_cores]

        mins = results['mins']
        means = results['means']
        maxs = results['maxs']
        stds = results['stds']

        bin_count_raw = results['info']['bin_count']
        bin_count = bin_count_raw  # TODO: group bins for variable bin_count

        # find consistent y scale
        maxs_all = []
        for temp_max in maxs.keys():
            maxs_all += maxs[temp_max]

        maxs_all = np.asarray(maxs_all)
        y_limit = maxs_all.max()

        bin_info = results['info']
        x_values, x_labels = self._get_labels_from_bin_info(bin_info['threshold_min'], bin_info['increment'], bin_count)

        core_list = self.core_string_to_list(cores)
        for core in core_list:
            # visualized: mean, min, max
            current_figure = plt.subplot(len(core_list), 1, core_list.index(core) + 1)

            current_maxs = maxs[str(core)]
            current_mins = mins[str(core)]
            current_means = means[str(core)]
            current_stds = stds[str(core)]

            if core == core_list[0]:  # first run (top subplot)
                plt.title("%s: %s - Latency Across Experiment %s\n" % (results['setup'][self.key_component].capitalize(), experiment, self.title_note), color=self.figure_background, loc='left')

            if core == core_list[len(core_list) - 1]:  # last run (bottom subplot)
                plt.xlabel("Time since start (%s groups total)" % bin_count, horizontalalignment='left', x=0.0)

            plt.ylabel("Latency: Core %s" % core, fontsize=9)
            plt.gca().set_yscale('log')

            temp_bins = range(bin_count)
            plt.bar(temp_bins, current_maxs, label='Max', color=self.figure_background)
            plt.bar(temp_bins, current_means, label='Mean', color=self.figure_foreground)
            plt.bar(temp_bins, current_mins, label='Min', color=self.figure_background)

            if core == core_list[0]:
                plt.legend(loc='upper right', prop={'size': 6})

            plt.xticks(x_values, x_labels, fontsize=6)

            # make graph pretty
            utils.remove_borders_from_plot()
            utils.set_axis_tick_color(self.figure_background)
            utils.set_log_y_axis_to_human_readable_times()
            plt.tick_params(axis='y', labelsize=7)
            utils.set_axis_tick_labels_to_color(self.figure_background)

        # save figure
        utils.save_plot(results['setup'][self.key_component], "%s_%s%s" % (experiment, results_type, filename_suffix), dpi=self.figure_dpi)

    # bin_count may be different from raw number of bins we collected information about, so it's left as a variable
    def _get_labels_from_bin_info(self, threshold_min, increment, bin_count, change_last_bucket=False, tick_count=10, print_func=utils.seconds_to_human_readable, scale='linear'):
        labels = []
        values = []

        step = int(bin_count / tick_count)

        if scale == 'linear':
            value_range = list(range(0, bin_count, step))  # +1 to bin count so range doesn't clip out final value
            value_range.append(value_range[len(value_range)-1] + step)
        elif scale == 'log':
            value_range = range(0, tick_count)
        else:
            raise ValueError("scale type '%s' isn't supported by _get_labels_from_bin_info!" % scale)

        for i in value_range:

            if scale == 'log':
                value = threshold_min + (10**i)
                value_index = value
            else:  # linear
                value = threshold_min + (i * increment)
                value_index = i

#            if i % step == 0 or i == bin_count - 1:
            if i == 0:
                value_string = print_func(value + increment)
            elif (i == value_range[len(value_range) - 1]) and change_last_bucket:
                value_string = print_func(value + increment)
                value_string += "+"
            else:
                value_string = print_func(value)

            if scale == 'log':
                for j in range(1, 10):
                    values.append(value_index * j)
                    if j == 1:
                        labels.append(value_string)
                    else:
                        labels.append("")
            else:
                labels.append(value_string)
                values.append(value_index)

        return values, labels

    def print_scientific(self, value):
        return '{:0.2e}'.format(value)

    def print_power_of_ten(self, value):
        length = len(str(int(value))) - 1

        return r'$10^{%s}$' % length # 10^x with superscript

    def validate_json_info(self, json_contents):
        try:
            environment = json_contents[self.key_environment]
        except KeyError:
            raise KeyError("Couldn't load environmental setup info in '%s'. This should be kept in the '%s' key. Exiting." % (self.filename, self.key_environment))

        important_keys = [self.key_component, self.key_cores, self.key_length_key, self.key_length_value, self.key_pool_size, self.key_element_count]

        for key in important_keys:
            try:
                temp = environment[key]
            except ValueError:
                raise ValueError("Couldn't find environemtal info for key '%s'. Exiting." % key)

    def core_string_to_list(self, core_string):
        cores = []
        ranges = []

        # parse single cores first
        for entry in core_string.split(","):
            if "-" not in entry:  # can't use length because cores can be in the hundreds
                cores.append(int(entry))
            else:
                ranges.append(entry)

        # parse ranges
        for entry in ranges:
            split_range = entry.split("-")
            for core in range(int(split_range[0]), int(split_range[1]) + 1):
                cores.append(core)

        return cores

    def plot_everything_for_valid_experiments(self, filename_suffix=""):
        try:
            with open(self.filename) as file:
                json_contents = json.load(file)

                # get environment info for experiment
                self.validate_json_info(json_contents)

                experiments = list(json_contents.keys())
                experiments.remove(self.key_environment)
        except Exception as e:
            print("Error: %s", e)
            raise Exception

        for experiment in experiments:
            self.plot_latency_histogram(experiment, filename_suffix=filename_suffix)
            self.plot_latency_over_time(experiment, filename_suffix=filename_suffix)
