import matplotlib.pylab as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np


class Simulation:
    def __init__(self, dt, t_trial, n_trials, seed):
        self.dt = dt
        self.t_trial = t_trial  # total simulation time per trial in ms
        self.n_trials = n_trials  # number of measurement trials
        self.seed = seed
        self.source = dict()
        self.multimeter = []
        self.neurons = dict()
        self.i_syn = dict()
        self.v_m = dict()
        self.sim_time = dict()
        self.input_ = []
        self.run_time = 0.

    @staticmethod
    def plot_spike_times(spike_times, title, legend=None, xlabel="time (ms)", ylim=None, xlim=None):
        fig = plt.figure()
        line_type = ["-", "--"]
        colors = ['C{}'.format(i) for i in range(6)]
        for i, spike_time in enumerate(spike_times):
            plt.eventplot(spike_time, orientation='horizontal',
                          colors=colors[i%len(colors)], linestyles=line_type[i % len(line_type)])
        if legend is not None:
            plt.legend(legend)
        plt.title(title)
        plt.xlabel(xlabel)
        if ylim is not None:
            plt.ylim(ylim)
        if xlim is not None:
            plt.xlim(xlim)
        plt.tight_layout(h_pad=2.0)
        return fig

    @staticmethod
    def plot_results(sim_time, plots, title, legend,
                     xlabel="time (ms)", ylabel="I (nA)", ylabel_e="I (nA)",
                     xlim=None, ylim=None, show_error=False, ye_lim=None, linewidth = 1.0):
        line_type = ["-", "--"]
        fig = plt.figure()
        if show_error:
            plt.subplot(2, 1, 1)
        for i in range(len(plots)):
            axes = plt.gca()
            if xlim is not None:
                axes.set_xlim(xlim)
                dt = sim_time[i][1]-sim_time[i][0]
                n = int(round((xlim[0]-sim_time[i][0])/dt, 5))
                m = int(round((xlim[1]-sim_time[i][0])/dt, 5))
                plots[i] = plots[i][n:m]
                sim_time[i] = sim_time[i][n:m]
            if ylim is not None:
                axes.set_ylim(ylim)
            plt.plot(sim_time[i], plots[i], line_type[i % len(line_type)], linewidth = linewidth)
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.legend(legend)
        if show_error:
            axes = plt.subplot(2, 1, 2)
            legend_ = []
            end_ = len(plots) if len(plots)>2 else 1
            for i in range(end_):
                if len(plots[i]) != len(plots[(i+1)%len(plots)]):
                    continue
                legend_.append(legend[i]+" - " + legend[(i+1)%len(plots)])
                error = np.array(plots[i]) - np.array(plots[(i+1)%len(plots)])
                # error[np.abs(error) > 0.1] = 0.
                time = (sim_time[i] if (sim_time[i].size < sim_time[(i+1)%len(plots)].size)
                        else sim_time[(i+1)%len(plots)])
                plt.plot(time, error, line_type[i % len(line_type)], linewidth = linewidth)

            plt.xlabel(xlabel)
            plt.ylabel(ylabel_e)
            plt.legend(legend_)
            # plt.title("Error")
            if ye_lim is not None:
                axes.set_ylim(ye_lim)
            if xlim is not None:
                axes.set_xlim(xlim)
        plt.tight_layout(h_pad=2.0)
        return fig

    @staticmethod
    def mse_evol(vec1, vec2):
        max_val = max(np.amax(vec1), np.amax(vec2)) ** 2
        N = min(vec1.size, vec2.size)
        MSE = np.zeros(N)
        MSE[0] = ((vec1[0] - vec2[0]) ** 2)
        for i in range(1, N):
            MSE[i] = ((vec1[i] - vec2[i]) ** 2 / (i + 1)) + (1. - (1. / (i + 1))) * MSE[i - 1]
        return abs(MSE / max_val)

    @staticmethod
    def se_evol(vec1, vec2):
        max_val = max(np.amax(vec1), np.amax(vec2))
        N = min(vec1.size, vec2.size)
        ME = np.zeros(N)
        for i in range(N):
            ME[i] = (vec1[i] - vec2[i]) ** 2
        return abs(ME / max_val)

    @staticmethod
    def show_plots(save=False, name="save.pdf", plots=None, show=True):
        if plots is None:
            plots = []
        if save:
            pp = PdfPages(name)
            for plot in plots:
                pp.savefig(plot)
            pp.close()
        if show:
            plt.show()
        plt.close('all')

    @staticmethod
    def poisson_uniform(size, rate, t, dt=0.1, offset=100.):
        results = [[np.around(sp + offset,
                              int(np.log10(1.0 / dt)))] if sp < t else [] for sp in
                   -np.log(np.random.rand(size)) / rate]
        overshoot = np.array([len(sp) > 0 for sp in results])

        while overshoot.any():
            ids_to_add = np.where(overshoot)[0]
            spike_to_add = -np.log(np.random.rand(ids_to_add.size)) / rate
            for id, sp in zip(ids_to_add, spike_to_add):
                new_sp = max(np.around(results[id][-1] + sp, int(np.log10(1.0 / dt))),
                             results[id][-1] + dt)
                if new_sp > offset + t:
                    overshoot[id] = False
                else:
                    results[id].append(new_sp)
        return results

    def process_results(self):
        pass

    def create_current_input(self, spike_times, target):
        pass

    def create_synapse(self, syn_params, source, target, stochastic=False):
        pass

    def create_adex_neuron(self, neuron_params):
        pass

    def create_multimeter(self, recording_point):
        pass

    def process_mean(self):
        for i in self.v_m.keys():
            self.i_syn[i] = np.array([np.mean(self.i_syn[i][:, k]) for (k, j) in enumerate(self.i_syn[i][0, :])])
            self.v_m[i] = np.array([np.mean(self.v_m[i][:, k]) for (k, j) in enumerate(self.v_m[i][0, :])])

    def process_deterministic(self):
        for i in self.v_m.keys():
            self.i_syn[i] = self.i_syn[i][0].tolist()
            self.v_m[i] =     self.v_m[i][0].tolist()

    def reset(self):
        # Reset outputs
        self.run_time = 0.
        self.sim_time = dict()
        self.v_m = dict()
        self.i_syn = dict()

    def run(self):
        self.process_results()
