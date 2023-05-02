import time
import nest
import numpy as np
from os.path import isfile
from simwbm.simulation import Simulation


class NestSimulation(Simulation):
    def __init__(self, t_trial, n_trials, seed, path, dt=0.1):
        Simulation.__init__(self, dt, t_trial, n_trials, seed)

        # Reset Kernel
        nest.ResetKernel()
        self.to_file = True if path is not None else False
        nest.SetKernelStatus({"grng_seed": self.seed,
                              "resolution": self.dt,
                              "print_time": True
                              })
        if self.to_file:
            nest.SetKernelStatus({
                "overwrite_files": True,
                "data_path": path
            })
        # if "rossert_et_al" not in nest.Models():
        #     nest.Install("rossert_et_almodule")
        self.spikes = dict()

    def create_current_input(self, spike_times, target):
        self.input_.append(nest.Create('spike_generator'))
        nest.SetStatus(self.input_[-1], {'spike_times': spike_times})
        nest.Connect(self.input_[-1], target, syn_spec={'delay': self.dt})

    def create_direct_input(self, input_, target):
        self.input_.append(nest.Create('step_current_generator'))
        nest.SetStatus(self.input_[-1], {'amplitude_values': input_[1:],
                                         'amplitude_times': self.dt * np.arange(1, len(input_), dtype=np.float32)})
        # print(nest.GetStatus(self.input_[-1]))
        for i in target:
            nest.Connect(self.input_[-1], [self.neurons[i]], syn_spec={'delay': self.dt})

    def create_background_noise(self, rate, target, weight, start=None, stop=None, index_=None, delay=None):
        params={'rate': rate}
        if start is not None:
            params["start"]= start
            params["stop"] = stop

        for poisson in nest.Create('poisson_generator', len(target), params=params):
            self.input_.append([poisson])
        loc = 0 if index_ is None else index_
        ldelay = self.dt if delay is None or delay < self.dt else delay
        for i, t in enumerate(target):
            if loc > 0 or t in self.neurons and "multisynapse" in str(nest.GetStatus([self.neurons[t]])[0]['model']):
                loc_loc = loc if loc > 0 else 1
                nest.Connect(self.input_[-len(target) + i], [self.neurons[t]], syn_spec={"receptor_type": loc_loc,
                                                                                         "delay": ldelay,
                                                                                         "weight": weight})
            else:
                nest.Connect(self.input_[-len(target) + i], self.source[t], syn_spec={'delay': ldelay,
                                                                                      "weight": weight})

    def create_synapse(self, syn_params, sources, target, stochastic=False):
        for i in range(len(syn_params["delay"])):
            weight = syn_params["weight"][i] if syn_params["weight"][i] > 1e-9 else 0.
            nest_synapse_params = {"model": "tsodyks2_synapse",
                                   "U": syn_params["U"][i],
                                   "u": syn_params["u"][i],
                                   "x": syn_params["x"][i],
                                   "tau_fac": syn_params["tau_fac"][i],
                                   "tau_rec": syn_params["tau_rec"][i],
                                   "weight": weight,
                                   "delay": syn_params["delay"][i],
                                   "receptor_type": int(syn_params["receptor_type"][i])
                                   }
            if stochastic:
                nest_synapse_params["model"] = "quantal_stp_synapse"
                del nest_synapse_params['x']
                # nest_synapse_params["n"] = syn_params["n"][i]

            if list(sources)[i] in self.neurons:  # normal neurons
                nest.Connect([self.neurons[list(sources)[i]]], [self.neurons[target]],
                             syn_spec=nest_synapse_params)

            elif list(sources)[i] in self.source:  # parrot neurons
                nest.Connect(self.source[list(sources)[i]], [self.neurons[target]], syn_spec=nest_synapse_params)

    def create_adex_neuron(self, neuron_params):
        nest_neuron_params = []
        number = len(neuron_params["C_m"])
        for i in range(number):
            nest_neuron_params.append({"C_m": neuron_params["C_m"][i],
                                       "g_L": neuron_params["g_L"][i],
                                       "V_th": neuron_params["V_th"][i],
                                       "V_reset": neuron_params["V_reset"][i],
                                       "E_L": neuron_params["E_L"][i],
                                       "V_m": neuron_params["V_m"][i] if "V_m" in list(neuron_params.keys()) else -80.0,
                                       "Delta_T": neuron_params["Delta_T"][i],
                                       "V_peak": neuron_params["V_peak"][i],
                                       "t_ref": neuron_params["t_ref"][i],
                                       "a": neuron_params["a"][i],
                                       "b": neuron_params["b"][i],
                                       "tau_w": neuron_params["tau_w"][i],
                                       "I_e": neuron_params["I_e"][i] if "I_e" in list(neuron_params.keys()) else 0.,
                                       # "w": 20.
                                       })
        neurons = nest.Create("aeif_cond_beta_multisynapse", number, nest_neuron_params)
        for i in range(number):
            self.neurons[neuron_params["gid"][i]] = neurons[i]

    def setStatus(self, neuron, params):
      nest.SetStatus([neuron], params)

    def create_parrot_neuron(self, gid):
        self.source[gid] = nest.Create("parrot_neuron", 1)
        return self.source[gid]

    def create_multimeter(self, recording_point):
        params = {"interval": self.dt,
                  "record_from": ["V_m"],  # "Rand", "I_syn"],
                  "withgid": True,
                  "precision": 6
                  }
        if self.to_file:
            params["to_memory"] = False
            params["to_file"] = True
        multimeter = nest.Create("multimeter", 1, params=params)

        nest.Connect(multimeter, list(recording_point.values()), syn_spec={"delay": self.dt})
        self.multimeter.append(multimeter)

    def get_multimeters_filenames(self):
        return [nest.GetStatus(multimeter)[0]["filenames"][0] for multimeter in self.multimeter]

    def create_spike_detector(self, recording_point):
        params = {"withgid": True}
        if self.to_file:
            params["to_memory"] = False
            params["to_file"] = True
        multimeter = nest.Create("spike_detector", 1, params=params)
        nest.Connect(list(recording_point.values()), multimeter, syn_spec={'delay': self.dt})
        self.multimeter.append(multimeter)
        for gid in recording_point.keys():
            self.spikes[gid] = []

    def process_results(self):
        if self.to_file:
            self.process_from_file()
        else:
            self.process_from_mem()

    def process_from_mem(self):
        reversed_gids = dict()
        for (key, value) in self.neurons.items():
            reversed_gids[value] = str(key)

        # for (key, value) in self.source.items():
        #     reversed_gids[value[0]] = str(key)

        num_points = int(self.n_trials * self.t_trial / self.dt)
        for i in range(len(self.multimeter)):
            gids = []
            for conn in nest.GetStatus(nest.GetConnections(self.multimeter[i])):
                gids.append(int(reversed_gids[conn["target"]]))

            events = nest.GetStatus(self.multimeter[i])[0]["events"]
            for gid in gids:
                self.sim_time[gid] = events['times'][np.where(events['senders'] == self.neurons[gid])][:int(self.t_trial/self.dt)]
                self.v_m[gid] = events['V_m'][np.where(events['senders'] == self.neurons[gid])][:num_points]
                self.v_m[gid].shape = (self.n_trials, int(self.t_trial / self.dt))
                # self.rand[gid] = events['Rand'][np.where(events['senders'] == self.neurons[gid])][:num_points]
                # self.rand[gid].shape = (self.n_trials, int(self.t_trial / self.dt))
                # self.i_syn[gid] = events['I_syn'][np.where(
                #                                     events['senders'] == self.neurons[gid])][:num_points] / 1000.
                # self.i_syn[gid].shape = (self.n_trials, int(self.t_trial / self.dt))

    def process_from_file(self):
        reversed_gids = dict()
        for (key, value) in self.neurons.items():
            reversed_gids[value] = str(key)

        # for (key, value) in self.source.items():
        #     reversed_gids[value[0]] = str(key)

        num_points = int(self.n_trials * self.t_trial / self.dt)
        for file_ in self.get_multimeters_filenames():
            if isfile(file_):
                with open(file_, 'r') as f:
                    for line in f:
                        splitLine = line.split()
                        gid = int(reversed_gids[int(splitLine[0])])
                        if "multimeter-" in file_:
                            if gid not in self.v_m.keys():
                                self.v_m[gid] = []
                                self.sim_time[gid] = []
                                # self.rand[gid] = []
                                # self.i_syn[gid] = []
                            self.v_m[gid].append(float(splitLine[2]))
                            # self.rand[gid].append(float(splitLine[3]))
                            self.sim_time[gid].append(float(splitLine[1]))
                            # self.i_syn[gid].append(float(splitLine[4]))
                        elif "spike_detector-" in file_:
                            self.spikes[gid].append(float(splitLine[1]))
        for gid in self.v_m:
            self.v_m[gid] = np.array(self.v_m[gid][:num_points])
            self.v_m[gid].shape = (self.n_trials, int(self.t_trial / self.dt))
            self.sim_time[gid] = np.array(self.sim_time[gid][: int(self.t_trial / self.dt)])
            # self.rand[gid] = np.array(self.rand[gid][:num_points])
            # self.i_syn[gid] = np.array(self.i_syn[gid][:num_points]) / 1000.

    def process_deterministic(self):
        for i in self.v_m.keys():
            self.v_m[i] = self.v_m[i][0].tolist()

    def run(self):
        self.reset()
        # print("-------------------------- Run NEST simulation ----------------------------")
        start_time = time.time()
        for t in range(self.n_trials):
            t_net = nest.GetKernelStatus('biological_time')
            for j in range(len(self.input_)):
                nest.SetStatus(self.input_[j], {'origin': t_net})
            nest.Simulate(self.t_trial)
        self.run_time = time.time() - start_time
        nest.Simulate(self.dt)
        # print("-------------------------- End NEST simulation ----------------------------")
        Simulation.run(self)

    def is_neuron_local(self, curr_id):
        if curr_id in self.neurons:
            return nest.GetStatus([self.neurons[curr_id]])[0]["local"]
        return False

    def reset(self):
        Simulation.reset(self)
        nest.ResetNetwork()
