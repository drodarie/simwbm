import numpy as np
import time
import operator
from neuron import h
from simwbm.simulation import Simulation

_loaded = False


class NeuronSimulation(Simulation):
    def __init__(self, t_trial, n_trials, seed, path, dt=0.05, do_parallel=False):
        Simulation.__init__(self, dt, t_trial, n_trials, seed)
        global _loaded
        if not _loaded:
            h.nrn_load_dll(path + "/x86_64/.libs/libnrnmech.so")
            h.load_file("stdrun.hoc")
            _loaded = True

        self.do_parallel = do_parallel
        if do_parallel:
            self.pc = h.ParallelContext()
            self.id = int(self.pc.id())
            self.nhost = int(self.pc.nhost())
        else:
            self.id = 0
            self.nhost = 1
        print("I am {} of {}".format(self.id, self.nhost))
        h.celsius = 34
        h.dt = self.dt
        h.steps_per_ms = 1.0 / self.dt
        self._gif_fun = dict()
        self._inner_current = dict()
        self._connections = []
        self.synapses = dict()
        self.randoms = [seed]
        self.spikes = dict()

    def __del__(self):
        if self.do_parallel:
            self.pc.done()

    def create_adex_neuron(self, neuron_params):
        position = 0.5
        for i in range(self.id, len(neuron_params["C_m"]), self.nhost): # Round Robin to match NEST
            gid = neuron_params["gid"][i]
            self.neurons[gid] = h.Section()
            self._gif_fun[gid] = h.AdEx(self.neurons[gid](position))
            self._gif_fun[gid].C_m = neuron_params["C_m"][i]
            self._gif_fun[gid].g_L = neuron_params["g_L"][i]
            self._gif_fun[gid].V_th = neuron_params["V_th"][i]
            self._gif_fun[gid].V_reset = neuron_params["V_reset"][i]
            self._gif_fun[gid].E_L = neuron_params["E_L"][i]
            self._gif_fun[gid].V_M = neuron_params["V_m"][i] if "V_m" in list(neuron_params.keys()) else -80.0
            self._gif_fun[gid].Delta_T = neuron_params["Delta_T"][i]
            self._gif_fun[gid].V_peak = neuron_params["V_peak"][i]
            self._gif_fun[gid].t_ref = neuron_params["t_ref"][i]
            self._gif_fun[gid].a = neuron_params["a"][i]
            self._gif_fun[gid].b = neuron_params["b"][i]
            self._gif_fun[gid].tau_w = neuron_params["tau_w"][i]
            self._gif_fun[gid].I_e = neuron_params["I_e"][i] if "I_e" in neuron_params else 0.
            self.synapses[gid] = [h.NetCon(self._gif_fun[gid]._ref_spike,
                                           None, sec=self.neurons[gid])]  # Empty conn to associate cell to node
            self.synapses[gid][0].threshold = 1 # spike detection threshold 
            if self.do_parallel:
                self.pc.set_gid2node(gid, int(self.pc.id()))
                self.pc.cell(gid, self.synapses[gid][0]) 

    def create_current_input(self, spike_times, target):
        self.source[target] = h.VecStim()
        self.input_.append(h.Vector(spike_times))
        self.source[target].play(self.input_[-1], self.dt)

    def create_background_noise(self, rate, targets, weight, start=None, stop=None, index_=None, delay=None):
        t = stop-start if stop is not None else self.t_trial
        ldelay = self.dt if delay is None or delay < self.dt else delay
        spike_times = self.poisson_uniform(len(targets), rate/1000., t, self.dt, start if start is not None else 0.)
        loc_loc = index_ if index_ is not None else 1
        for i, target in enumerate(targets):
            self.source[-target] = h.VecStim()
            self.input_.append(h.Vector(spike_times[i]))
            self.source[-target].play(self.input_[-1], self.dt)

            connection = h.NetCon(self.source[-target], self._gif_fun[target])
            connection.weight[0] = -weight
            connection.weight[1] = loc_loc - 1
            connection.delay = ldelay + self.dt
            if -target not in self.synapses:
                self.synapses[-target] = []
            self.synapses[-target].append(connection)

    def create_direct_input(self, input_, targets):
        for target in targets:
            if target in self._gif_fun:
                if target not in self._inner_current:
                    self._inner_current[target] = []
                self._inner_current[target].append(h.Vector(input_))
                self._gif_fun[target].setI_stim(self._inner_current[target][-1])

    def set_receptors(self, neuron, params):
        if neuron in self._gif_fun:
            self._gif_fun[neuron].setPostsyn(h.Vector(params["E_rev"]),
                                             h.Vector(params["tau_rise"]), h.Vector(params["tau_decay"]))

    def create_synapse(self, syn_params, sources, target, stochastic=False):
        nb_synapses_ = len(syn_params["delay"])
        id_syn = self._gif_fun[target].initSynapse(nb_synapses_)
        for i in range(nb_synapses_):
            if list(sources)[i] in self.neurons or list(sources)[i] in self.source:
                weight = syn_params["weight"][i] if syn_params["weight"][i] > 1e-9 else 0.
                self._gif_fun[target].addSynapse(id_syn+i, weight, 
                                                 syn_params["tau_rec"][i], syn_params["tau_fac"][i], 
                                                 syn_params["U"][i],
                                                 syn_params["delay"][i], # Delay used for last_spike to match Nest
                                                 syn_params["x"][i], syn_params["u"][i]) 

                connection = None
                if list(sources)[i] in self.neurons:
                    connection = h.NetCon(self._gif_fun[list(sources)[i]]._ref_spike,
                                          self._gif_fun[target], sec=self.neurons[list(sources)[i]])
                elif list(sources)[i] in self.source:
                    connection = h.NetCon(self.source[list(sources)[i]], self._gif_fun[target])
                if connection is not None:
                    connection.weight[0] = id_syn+i
                    connection.weight[1] = syn_params["receptor_type"][i] - 1
                    connection.delay = syn_params["delay"][i]+self.dt
                    if list(sources)[i] not in self.synapses:
                        self.synapses[list(sources)[i]] = []
                    self.synapses[list(sources)[i]].append(connection)

    def create_multimeter(self, recording_point):
        for k,v in recording_point.items():
            multimeter = {"gid": k,
                          "v_m": h.Vector(),
                          # "spike": h.Vector(),
                          }
            multimeter["v_m"].record(self._gif_fun[k]._ref_V_M)
            # multimeter["spike"].record(self._gif_fun[k]._ref_spike)
            # if recording_point[i][0] in self.synapses:
            #     for synapse in self.synapses[recording_point[i][0]]:
            #         multimeter["i_syn"].append(h.Vector())
            #         multimeter["i_syn"][-1].record(self._gif_fun[recording_point[i][0]]._ref_dg_in_double[int(synapse.weight[0])])
            self.multimeter.append(multimeter)

    def create_spike_detector (self, recording_point):
        # for k in recording_point.keys():
        #     self.pc.spike_record(k, self.spikes[1], self.spikes[0])
        for k, v in recording_point.items():
            multimeter = {"gid": k,
                          "spike_times": h.Vector()}
            if k not in self.synapses:
                self.synapses[k] = [h.NetCon(self._gif_fun[k]._ref_spike,
                                             None, sec=self.neurons[k])]  # Empty conn to associate cell to node
            self.synapses[k][0].record(multimeter["spike_times"])
            self.multimeter.append(multimeter)

    def extract_values(self):
        for i in range(len(self.multimeter)):
            gid = self.multimeter[i]["gid"]
            if gid not in self.v_m:
                self.v_m[gid] = []
                self.i_syn[gid] = []
            if "v_m" in self.multimeter[i]:
                self.v_m[gid].append(np.array(self.multimeter[i]["v_m"], dtype=np.float64)[1:])
            elif "spike_times" in self.multimeter[i]:
                self.spikes[gid] = np.array(self.multimeter[i]["spike_times"], dtype=np.float64)
            self.i_syn[gid].append(np.zeros(int(self.t_trial/self.dt)))
            # if "spike" in self.multimeter[i]:
            #     self.i_syn[gid].append(np.array(self.multimeter[i]["spike"], dtype=np.float64)[1:])

    def process_results(self):
        sim_time = np.arange(self.dt, self.t_trial+self.dt, self.dt)
        for multimeter in self.multimeter:
            gid = multimeter["gid"]
            self.v_m[gid] = np.array(self.v_m[gid])
            self.sim_time[gid] = sim_time
            self.i_syn[gid] = np.array(self.i_syn[gid])

    def spikes_to_file(self, filename):
        with open(filename, "w") as spk_file:
            # for (k,t) in zip(self.spikes[0], self.spikes[1]):
            #     spk_file.write('%d\t%.3f\n' %(k, t))
            result = []
            for gid, spikes in self.spikes.items():
                for spike in spikes:
                    result.append((gid, spike))
            for (k,t) in sorted(result, key=operator.itemgetter(1)):
                spk_file.write('%d\t%.3f\n' %(k, t))

    def voltage_to_file(self, filename):
        with open(filename, "w") as volt_file:
            result = []
            for gid, voltages in self.v_m.items():
                for ti, voltage in enumerate(voltages[0]):
                    result.append((gid, self.sim_time[gid][ti], voltage))
            for (k, t, v) in sorted(result, key=operator.itemgetter(1)):
                volt_file.write('%d\t%.3f\t%.3f\n' %(k, t, v))

    def run(self):
        self.reset()
        h.init()
        h.tstop = self.t_trial
        # print("-------------------------- Run NEURON simulation ----------------------------")
        start_time = time.time()
        for i in range(self.n_trials):
            h.run()
            self.extract_values()
        self.run_time = time.time() - start_time
        # print("-------------------------- End NEURON simulation ----------------------------")
        Simulation.run(self)
