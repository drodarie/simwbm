import h5py
import datetime
import json
import time

from os import makedirs
from os.path import *
import numpy as np
import sys
from simwbm.nest_simulation import NestSimulation
from simwbm.nrn_simulation import NeuronSimulation
import shutil
import sqlite3

DATE = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")

Dver = 'v9_test' if len(sys.argv) < 2 else sys.argv[1]
ROOT_FOLDER = dirname(realpath(__file__))
NEURON_MOD_FOLDER = join(ROOT_FOLDER, "NEURON")
HIGHD_FOLDER = ROOT_FOLDER
linewidth_ = 0.1

databases_file = join(HIGHD_FOLDER, "ElecDB2.db")
conn = sqlite3.connect(databases_file)
cursor = conn.cursor()


def main():
    seed = 75599
    t_trial = 3000.0  # ms
    n_trial = 1
    dt = 0.1  # ms
    print_plots = False
    global OUTPUT_FOLDER
    assert (seed > 0 and t_trial > 0 and n_trial > 0 and dt > 0.)
    np.random.seed(seed)
    # FLAGS
    request = "select distinct name from etypes"
    cursor.execute(request)

    standalone_input_ = []
    i0 = np.random.rand(50) * 200.
    for id_, i in enumerate(i0):
        if id_ == 0:
            i = 0  # prevent errors linked to initiation
        standalone_input_.append(i * np.ones(int(t_trial / (dt * 50))))
    standalone_input_ = np.concatenate(standalone_input_)

    functions = np.fromfunction(
        lambda x: (x >= int(700.0 / dt)) & (x <= int((700.0 + 1000.0) / dt)) | (x >= int(2000.0 / dt)) & (
                    x <= int((409.3 + 2000.0) / dt)) | (x >= int(2900.0 / dt)), (int(t_trial / dt),))
    standalone_input_ = np.vstack((functions * 45., functions * 100., functions * 120., standalone_input_))

    # standalone_input_ = np.array([functions*45.])
    # standalone_input_ = np.array([np.zeros(int(t_trial / dt))])
    for etype in np.array(cursor.fetchall())[:, 0]:
        for sim in range(2):
            OUTPUT_FOLDER = join(ROOT_FOLDER, "save", Dver, etype)
            with_step_current = True if sim > 0 else False
            print(etype, with_step_current)

            # Create output
            if with_step_current:
                OUTPUT_FOLDER = join(OUTPUT_FOLDER, "random_step_current")
            else:
                OUTPUT_FOLDER = join(OUTPUT_FOLDER, "synapse_replay")

            try:
                if exists(OUTPUT_FOLDER):
                    shutil.rmtree(OUTPUT_FOLDER)
                makedirs(OUTPUT_FOLDER)
            except Exception as e:
                print(e)
                return

            standalone_input = None
            if with_step_current:
                standalone_input = standalone_input_
            simulate(seed, t_trial, n_trial, dt, OUTPUT_FOLDER, etype, standalone_input, print_plots)


def simulate(seed,  # random seed
             t_trial,  # simulation time per trial in ms
             n_trials,  # number of measurement trials
             dt,  # time precision in ms
             folder,
             etype,  # gid to test
             step_current,
             print_plots):
    start_time = time.time()
    print("------------------------- Initializing Simulators -------------------------")
    nest_sim = NestSimulation(t_trial, n_trials, seed, folder, dt)

    request = "select id from etypes where name='" + etype + "'"
    cursor.execute(request)
    etype_id = np.array(cursor.fetchall())[:, 0]
    request = "select distinct is_excitatory, stype from connections join cell_types on connections.source_celltype = cell_types.id where target_etype=" + str(
        etype_id[0])
    cursor.execute(request)
    result = np.array(cursor.fetchall())
    stype_ids = result[:, 1]
    receptorsIsExc = result[:, 0] - 1
    tested_gid = np.array([id_ for id_ in range((len(step_current) if step_current is not None else len(stype_ids)))])
    print("------------------------- Create Neurons ----------------------------------")
    request = "SELECT constants.value, parameters.name, constants.id FROM (select id, neuron_model from neuron_instances where etype=" + str(
        etype_id[
            0]) + ") as neuron_instances JOIN (SELECT * FROM neuron_models WHERE neuron_models.name like '%aeif_cond_beta_multisynapse%') as neuron_models ON neuron_instances.neuron_model=neuron_models.id JOIN neuron_instances_constants ON neuron_instances_constants.instance=neuron_instances.id JOIN constants ON neuron_instances_constants.constant=constants.id JOIN parameters ON constants.parameter=parameters.id order by constants.id"
    cursor.execute(request)
    result = cursor.fetchall()
    if len(result) < 1:
        return
    neuron_parameters = {}
    for row in result:
        if row[1] in list(neuron_parameters.keys()):
            neuron_parameters[row[1]] = np.vstack([neuron_parameters[row[1]], [float(row[0])] * tested_gid.size])
        else:
            neuron_parameters[row[1]] = [float(row[0])] * tested_gid.size
    neuron_parameters["gid"] = tested_gid
    neuron_parameters["V_m"] = [-65.0] * tested_gid.size  # mV
    neuron_parameters["I_e"] = [0.0] * tested_gid.size  # [21.85]*tested_gid.size,  # pA

    # for k, v in neuron_parameters.items():
    #     print(k, v)
    nest_sim.create_adex_neuron(neuron_parameters)

    pre_synaptic_parameters = {}
    postsyn_params = {'E_rev': neuron_parameters['E_rev'][:, 0],
                      'tau_rise': neuron_parameters['tau_rise'][:, 0],
                      'tau_decay': neuron_parameters['tau_decay'][:, 0]}
    # print(postsyn_params)
    spike_times = []
    if step_current is None:
        print("-------------------------- Create synapses ------------------------------------")
        params = None
        for stype_id in stype_ids:
            request = "SELECT constants.value, parameters.name FROM (select id, synapse_model from synapse_instances where stype=" + str(
                stype_id) + ") as synapse_instances JOIN (SELECT * FROM synapse_models WHERE synapse_models.name like '%tsodyks2_synapse%') as synapse_models ON synapse_instances.synapse_model=synapse_models.id JOIN synapse_instances_constants ON synapse_instances_constants.instance=synapse_instances.id JOIN constants ON synapse_instances_constants.constant=constants.id JOIN parameters ON constants.parameter=parameters.id order by parameters.name"
            cursor.execute(request)
            result = cursor.fetchall()
            if len(result) < 1:
                return
            elif params is None:
                params = np.array([float(row[0]) for row in result])
            else:
                params = np.vstack((params, np.array([float(row[0]) for row in result])))

        receptors = 3 - receptorsIsExc * 2  # conversion from EI to concrete receptor type
        indexes = np.where(receptors == 3)
        params[:, 3][indexes] = np.abs(params[:, 3][indexes])
        params = np.hstack((params, receptors.reshape((len(receptors), 1))))
        for neuron in nest_sim.neurons.values():
            nest_sim.setStatus(neuron, postsyn_params)
        gid = 100
        rate = 10. * dt * 1e-3
        spike_times = np.around(np.cumsum(-np.log(np.random.rand(int(rate * t_trial / dt))) / rate), 1)
        target = nest_sim.create_parrot_neuron(gid)
        nest_sim.create_current_input(spike_times, target)
        spike_times = [spike_times]
        for i in range(len(tested_gid)):
            pre_synaptic_parameters = {
                "delay": [dt],  # in ms
                "tau_rec": [params[i, 2]],  # ms
                "tau_fac": [params[i, 1]],  # ms
                "U": [params[i, 0]],
                'u': [0.5],
                'x': [0.5],
                'weight': [params[i, 3]],
                'receptor_type': [params[i, 4]]
            }
            nest_sim.create_synapse(pre_synaptic_parameters, list(nest_sim.source.keys()), tested_gid[i])
    else:
        params = None
        for i in range(tested_gid.size):
            nest_sim.create_direct_input(step_current[i], [tested_gid[i]])
        nest_sim.create_spike_detector(nest_sim.neurons)

    print("-------------------------- Create multimeter ----------------------------")
    nest_sim.create_multimeter(nest_sim.neurons)

    loadtime = time.time() - start_time
    nest_sim.run()
    nrn_sim = load_neuron(seed, t_trial, n_trials, dt, neuron_parameters,
                          params, postsyn_params, step_current, spike_times, folder)
    print("--------------------------- Recover output data -----------------------------")
    nest_sim.process_deterministic()
    nrn_sim.process_deterministic()

    plots = []
    if step_current is None:
        plots.append(nest_sim.plot_spike_times([spike_times[0]], "Presynaptic spike times", xlim=[dt, t_trial]))
    for i in range(tested_gid.size):
        if step_current is not None:
            plots.append(nest_sim.plot_results([nest_sim.sim_time[tested_gid[i]][:-1]], [step_current[i][1:]],
                                               "Current input injected at soma", ['Stim. current'],
                                               ylabel="Stim. current (pA)", linewidth=linewidth_))
            plots.append(nest_sim.plot_results(
                [np.array(nest_sim.sim_time[tested_gid[i]]), np.array(nrn_sim.sim_time[tested_gid[i]])],
                [nest_sim.v_m[tested_gid[i]], nrn_sim.v_m[tested_gid[i]]],
                etype + ": Membrane potential comparison upon clamp current",
                ["NEST", "NEURON"], ylabel="Membrane potential (mV)", show_error=True,
                ylabel_e=" Diff. membrane potential (mV)", linewidth=linewidth_)
                         )
            plots.append(nest_sim.plot_spike_times([nest_sim.spikes[tested_gid[i]], nrn_sim.spikes[tested_gid[i]]],
                                                   "Spike times", ["NEST", "NEURON"], xlim=[dt, t_trial]))
        else:
            plots.append(nest_sim.plot_results(
                [np.array(nest_sim.sim_time[tested_gid[i]]), np.array(nrn_sim.sim_time[tested_gid[i]])],
                [nest_sim.v_m[tested_gid[i]], nrn_sim.v_m[tested_gid[i]]],
                etype + ": Post synaptic membrane potential comparison",
                ["NEST", "NEURON"], ylabel="Membrane potential (mV)", show_error=True,
                ylabel_e=" Diff. membrane potential (mV)", linewidth=linewidth_)
                         )
    nest_sim.show_plots(save=True, name=join(OUTPUT_FOLDER, "Membrane_Potential.pdf"), plots=plots, show=print_plots)
    del nrn_sim


def load_neuron(seed, t_trial, n_trials, dt, gifs, presyn_params, postsyn_params, step_current, inputs, output_folder):
    sim_times = dict()
    i_syns = dict()
    v_ms = dict()
    rands = dict()
    spikes = dict()
    run_time = 0
    neurons = dict()
    for trial in range(n_trials):
        nrn_sim = NeuronSimulation(t_trial, 1, seed, NEURON_MOD_FOLDER, dt)
        nrn_sim.create_adex_neuron(gifs)
        if step_current is None:
            for neuron in nrn_sim.neurons:
                nrn_sim.set_receptors(neuron, postsyn_params)
            gid = 100
            for input_ in inputs:
                nrn_sim.create_current_input(input_, gid)
                gid += 1

            j = 0
            for i, synapse in enumerate(presyn_params):
                pre_synaptic_parameters = {
                    "delay": [dt],  # in ms
                    "tau_rec": [synapse[2]],  # ms
                    "tau_fac": [synapse[1]],  # ms
                    "U": [synapse[0]],
                    'u': [0.5],
                    'x': [0.5],
                    'weight': [synapse[3]],
                    'receptor_type': [synapse[4]]
                }
                # print(pre_synaptic_parameters)
                nrn_sim.create_synapse(pre_synaptic_parameters, list(nrn_sim.source.keys()), gifs["gid"][i])
                j += 1
        else:
            for i, target in enumerate(gifs["gid"]):
                nrn_sim.create_direct_input(step_current[i], [target])
            nrn_sim.create_spike_detector(nrn_sim.neurons)
        nrn_sim.create_multimeter(nrn_sim.neurons)
        nrn_sim.run()
        nrn_sim.spikes_to_file(join(output_folder, "spikes_" + str(trial) + ".txt"))
        for neuron in nrn_sim.neurons.keys():
            if neuron not in sim_times:
                i_syns[neuron] = []
                v_ms[neuron] = []
                spikes[neuron] = []
                rands[neuron] = []
            sim_times[neuron] = nrn_sim.sim_time[neuron]
            i_syns[neuron].append(nrn_sim.i_syn[neuron][0])
            v_ms[neuron].append(nrn_sim.v_m[neuron][0])
            if neuron in nrn_sim.spikes:
                spikes[neuron].append(nrn_sim.spikes[neuron])
            run_time += nrn_sim.run_time
        path = "."
        seed += 1
        neurons = nrn_sim.neurons
        del nrn_sim
    nrn_sim = NeuronSimulation(t_trial, 1, seed, path, dt)
    nrn_sim.sim_time = sim_times
    nrn_sim.i_syn = i_syns
    nrn_sim.v_m = v_ms
    nrn_sim.spikes = spikes
    nrn_sim.run_time = run_time
    nrn_sim.neurons = neurons
    for neuron in nrn_sim.v_m:
        nrn_sim.i_syn[neuron] = np.array(nrn_sim.i_syn[neuron])
        nrn_sim.v_m[neuron] = np.array(nrn_sim.v_m[neuron])
        nrn_sim.spikes[neuron] = np.array(nrn_sim.spikes[neuron])
    return nrn_sim


def save_output(simulations, step_current, loadtime):
    print("---------------------------- Store output data ------------------------------")
    first_sim = simulations.values()[0]
    parameters = dict()
    parameters['sim_date'] = DATE
    parameters['seed'] = first_sim.seed
    parameters['t_trial'] = first_sim.t_trial
    parameters['n_trials'] = first_sim.n_trials
    parameters['dt'] = first_sim.dt
    parameters["loadtime"] = loadtime
    parameters['runtime'] = dict()

    with h5py.File(OUTPUT_FOLDER + '/simulation.h5', 'w') as f:
        f.attrs['sim_date'] = DATE
        input_group = f.create_group('input')
        input_group.attrs['seed'] = first_sim.seed
        input_group.attrs['t_trial'] = first_sim.t_trial
        input_group.attrs['n_trials'] = first_sim.n_trials
        input_group.attrs['dt'] = first_sim.dt
        input_group.attrs['n_GIF_nrn'] = len(first_sim.neurons)
        if step_current is not None:
            input_group.create_dataset('input_current', data=np.array(step_current))
        output_group = f.create_group('output')
        for name, simulation_ in simulations.items():
            simulation_group = output_group.create_group(name)
            potential_group = simulation_group.create_group("membrane_potentials")
            random_group = simulation_group.create_group("random_values")
            syn_group = simulation_group.create_group("synapses_current")
            for neuron in simulation_.neurons.keys():
                potential_group.create_dataset(str(neuron), data=np.array(simulation_.v_m[neuron]))
                syn_group.create_dataset(str(neuron), data=np.array(simulation_.i_syn[neuron]))
                random_group.create_dataset(str(neuron), data=np.array(simulation_.rand[neuron]))
            parameters['runtime'][name] = simulation_.run_time

    with open(OUTPUT_FOLDER + '/output.json', 'w') as f:
        json.dump(parameters, f, indent=4)


if __name__ == '__main__':
    main()
