import h5py
import datetime
import json
import time
from os import makedirs
from os.path import *
import numpy as np
import sys
from simwbm.nest_simulation import NestSimulation
from mpi4py import MPI  # import mpi after nest AND before neuron !!!
from simwbm.nrn_simulation import NeuronSimulation
import shutil
import sqlite3

DATE = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
mpi_size = comm.Get_size()

Dver = 'v9' if len(sys.argv) < 2 else sys.argv[1]
SIM_NAME = "Whole_Brain" if len(sys.argv) < 3 else sys.argv[2]
CONN_PRINTING = True
ROOT_FOLDER = dirname(realpath(__file__))
OUTPUT_FOLDER = join(ROOT_FOLDER, "save", Dver, SIM_NAME)
NEURON_MOD_FOLDER = join(ROOT_FOLDER, "NEURON")
NRN_OUTPUT = join(OUTPUT_FOLDER, "neuron_output")
NEST_OUTPUT = join(OUTPUT_FOLDER, "nest_output")
HIGHD_FOLDER = join(ROOT_FOLDER, "data")

databases_file = join(HIGHD_FOLDER, 'db', "ElecDB.db")
conn = sqlite3.connect(databases_file)
cursor = conn.cursor()


def main():
    seed = 75599
    t_trial = 2100.0  # ms
    n_trial = 1
    dt = 0.1  # ms
    file_ = join(HIGHD_FOLDER, 'whole_brain_model', 'whole_brain_model', Dver, 'whole_brain_model_Nest.h5')
    record_potentials = False
    save_to_h5 = False
    regions = ["Primary somatosensory area, lower limb"]
    # regions = ["Cerebral nuclei", "Brain stem", "Cerebellum"]
    # regions = []
    assert (seed > 0 and t_trial > 0 and n_trial > 0 and dt > 0.)
    status = None
    if rank == 0:
        try:
            if exists(OUTPUT_FOLDER):
                shutil.rmtree(OUTPUT_FOLDER)
            makedirs(NRN_OUTPUT)
            makedirs(NEST_OUTPUT)
            status = True
        except Exception as e:
            print(e)
            status = False
    status = comm.bcast(status, root=0)
    if not status:
        return

    np.random.seed(seed)
    simulate(Dver, seed, t_trial, n_trial, dt, file_, regions, record_potentials, save_to_h5)


def simulate(Dver,  # version of the Brain model
             seed,  # random seed
             t_trial,  # simulation time per trial in ms
             n_trials,  # number of measurement trials
             dt,  # time precision in ms
             file_,
             regions,
             record_potentials,
             save_to_h5):
    # n_stochastic = 1
    start_time = time.time()
    if rank == 0: print("------------------------- Initializing Nest Simulator -------------------------")
    nest_sim = NestSimulation(t_trial, n_trials, seed, NEST_OUTPUT, dt)
    nrn_sim = NeuronSimulation(t_trial, n_trials, seed, NEURON_MOD_FOLDER, dt, True)
    h5file = h5py.File(file_, "r")
    gids = np.array(h5file["/neurons/default/gid"])
    ids = np.ones(gids.size, dtype=bool)
    if len(regions) > 0:
        request = "select id from regions where is_leaf=1 AND (full_name like '%" + ("%' OR full_name like '%").join(
            regions) + "%')"
        cursor.execute(request)
        regions_ids = np.array(cursor.fetchall())[:, 0]
        gid_reg = np.array(h5file['neurons/regions'])
        ids = np.zeros(gids.size, dtype=bool)
        for reg in regions_ids:
            ids = np.logical_or(ids, gid_reg == reg)
    gids = gids[ids]
    comm.Barrier()
    if rank == 0:
        print("------------------------- Create " + str(gids.size) + " Neurons ----------------------------------")
    neuron_parameters = {
        "gid": gids.tolist(),
        "C_m": np.array(h5file["/neurons/default/C_m"])[ids].tolist(),  # pF
        "E_L": np.array(h5file["/neurons/default/E_L"])[ids].tolist(),  # mV
        "g_L": np.array(h5file["/neurons/default/g_L"])[ids].tolist(),  # nS
        "V_reset": np.array(h5file["/neurons/default/V_reset"])[ids].tolist(),  # mV
        "V_th": np.array(h5file["/neurons/default/V_th"])[ids].tolist(),  # mV
        "V_peak": np.array(h5file["/neurons/default/V_peak"])[ids].tolist(),  # mV
        "Delta_T": np.array(h5file["/neurons/default/Delta_T"])[ids].tolist(),  # mV
        "a": np.array(h5file["/neurons/default/a"])[ids].tolist(),  #
        "b": np.array(h5file["/neurons/default/b"])[ids].tolist(),  #
        "tau_w": np.array(h5file["/neurons/default/tau_w"])[ids].tolist(),  # ms
        "t_ref": ([5.0] * len(gids) if "t_ref" not in h5file["/neurons/default"].keys() else
                  np.array(h5file["/neurons/default/t_ref"])[ids].tolist()),  # ms
    }
    is_excitatory = np.array(h5file["/neurons/excitatory"])
    is_excitatory[np.where(is_excitatory < 0)] = 0
    is_excitatory[np.where(is_excitatory > 1)] = 1  # Force modulatory neurons to be excitatory
    nest_sim.create_adex_neuron(neuron_parameters)
    nrn_sim.create_adex_neuron(neuron_parameters)
    if rank == 0:
        with open(OUTPUT_FOLDER + '/conversion.json', 'w') as f:
            json.dump(nest_sim.neurons, f, indent=4)

    del neuron_parameters

    ########################### Stimulation #############################################
    # radius_ = np.sqrt( ((0.8-h5file["IO/x"][:][gids])**2.0) + ((0.47-h5file["IO/y"][:][gids])**2.0) ) # whisker
    # radius_ = radius_[h5file["IO/gid"][:][gids]==11]
    # StimulationIDs  = list(gids[np.where( (h5file["IO/gid"][:][gids]==11) )])
    # StimulationIext = 1000.0 *np.exp(-radius_*radius_/0.010)

    # request = "select id from regions where is_leaf=1 AND full_name like '%"+regions[-1]+"%'"
    request = "select id from regions where is_leaf=1 AND full_name like '%Ventral posteromedial nucleus%'"
    cursor.execute(request)
    regions_ids = np.array(cursor.fetchall())[:, 0]
    ids *= False
    for reg in regions_ids:
        ids = np.logical_or(ids, gid_reg == reg)
    StimulationIDs = np.array(h5file["/neurons/default/gid"])[ids].tolist()
    print("Nb stimulation neurons: " + str(len(StimulationIDs)))
    rate = 10. * dt * 1e-3
    spike_times = np.around(
        np.cumsum(-np.log(np.random.rand(len(StimulationIDs), int(rate * (1000.) / dt))) / rate, axis=1), 1) + 100.
    with open(OUTPUT_FOLDER + '/stimulation.gdf', 'w') as f:
        for i, gid in enumerate(StimulationIDs):
            target = nest_sim.create_parrot_neuron(gid)
            nest_sim.create_current_input(spike_times[i], target)
            nrn_sim.create_current_input(spike_times[i], gid)
            for j in spike_times[i]:
                f.write(str(j) + "\t" + str(gid) + "\n")

    gids = np.concatenate((gids, StimulationIDs))
    # StimulationIext = 60.0 *np.ones(len(StimulationIDs))
    # for i in range(len(StimulationIDs)):
    #     standalone_input = StimulationIext[i] * np.fromfunction(lambda x: (x>=2000) & (x % 2000 <= 200), (int(t_trial / dt),))
    #     nest_sim.create_direct_input(standalone_input, [StimulationIDs[i]])
    #     nrn_sim.create_direct_input(standalone_input, [StimulationIDs[i]])
    start = 0
    post_gids = np.array(h5file["/presyn/default/post_gid"], int)
    filter_post = np.in1d(post_gids, gids)
    post_gids = post_gids[filter_post]
    pre_gids = np.array(h5file["/presyn/default/pre_gid"], int)[filter_post]
    filter_pre = np.in1d(pre_gids, gids)
    pre_gids = pre_gids[filter_pre]
    post_gids = post_gids[filter_pre]

    n_synapses = len(post_gids)
    comm.Barrier()
    if rank == 0:
        print("-------------------------- Create " + str(
            n_synapses * 2) + " synapses ------------------------------------")
        if CONN_PRINTING:
            sys.stdout.write("\n" * mpi_size)
            sys.stdout.flush()
    percent_done = -1
    num_created = 0
    post_synaptic_parameters = pre_synaptic_parameters = {}
    comm.Barrier()

    # WARNING: Might consume all your memory
    syn_params = [h5file["/presyn/default/delay"][filter_post][filter_pre],  # in ms
                  h5file["/presyn/default/tau_rec"][filter_post][filter_pre],  # ms
                  h5file["/presyn/default/tau_fac"][filter_post][filter_pre],  # ms
                  h5file["/presyn/default/U"][filter_post][filter_pre],
                  h5file["/presyn/default/weight"][filter_post][filter_pre]]
    postsyn_params = {'E_rev': [0.0, 0.0, -80.0, -97.0],
                      'tau_rise': [0.2, 0.29, 0.2, 3.5],
                      'tau_decay': [1.7, 43.0, 8.0, 260.9]}
    while start < n_synapses:
        current_percent = min(int(float(start) / float(n_synapses) * 100), 100)
        if CONN_PRINTING and current_percent > percent_done:
            progress_bar(current_percent, num_created)
            percent_done = current_percent
        range_ = range(start, n_synapses)
        curr_id = post_gids[start]
        i_win = 0
        for i_win in range_:
            if curr_id != post_gids[i_win]:
                break
            elif i_win == n_synapses - 1:
                i_win += 1
        if nest_sim.is_neuron_local(curr_id):
            # if curr_id in nrn_sim._gif_fun.keys():
            pre_ids = pre_gids[start:i_win]
            local_excitatory = is_excitatory[pre_ids]
            receptors = 3 - local_excitatory * 2
            weights = syn_params[4][start:i_win]
            indexes = np.where(receptors == 3)
            weights[indexes] = np.abs(weights[indexes])

            pre_synaptic_parameters = {
                "delay": syn_params[0][start:i_win],  # in ms
                "tau_rec": syn_params[1][start:i_win],  # ms
                "tau_fac": syn_params[2][start:i_win],  # ms
                "U": syn_params[3][start:i_win],
                'u': (0.5 * np.ones(i_win - start, np.float64)),
                'x': (0.5 * np.ones(i_win - start, np.float64)),
                'weight': weights,
                'receptor_type': receptors
            }
            nest_sim.setStatus(nest_sim.neurons[curr_id], postsyn_params)
            nrn_sim.set_receptors(curr_id, postsyn_params)

            nest_sim.create_synapse(pre_synaptic_parameters, pre_ids, curr_id)
            nrn_sim.create_synapse(pre_synaptic_parameters, pre_ids, curr_id)
            receptors += 1
            weights[np.where(receptors == 2)] *= 0.4 + 0.4 * (1 - is_excitatory[curr_id])
            weights[np.where(receptors == 4)] *= 0.75 * is_excitatory[curr_id]
            pre_synaptic_parameters['weight'] = weights.tolist()
            pre_synaptic_parameters['receptor_type'] = receptors.tolist()
            nest_sim.create_synapse(pre_synaptic_parameters, pre_ids, curr_id)
            nrn_sim.create_synapse(pre_synaptic_parameters, pre_ids, curr_id)
            num_created += 2 * (i_win - start)
        start = i_win
    if CONN_PRINTING:
        progress_bar(100, num_created)
    h5file.close()
    del post_synaptic_parameters, pre_synaptic_parameters, h5file, gids, pre_gids, post_gids, pre_ids, filter_post, filter_pre
    ########################### Background Noise ########################################
    comm.Barrier()
    if rank == 0: print("-------------------------- Create background noise ----------------------------")
    nest_sim.create_background_noise(0.5, nest_sim.neurons)
    if rank == 0: print("-------------------------- Create spike detector ----------------------------")
    nest_sim.create_spike_detector(nest_sim.neurons)
    nrn_sim.create_spike_detector(nrn_sim.neurons)
    # nest_sim.create_spike_detector(nest_sim.source)

    if record_potentials:
        if rank == 0: print("-------------------------- Create multimeter ----------------------------")
        nest_sim.create_multimeter(nest_sim.neurons)
        nrn_sim.create_multimeter(nrn_sim.neurons)
    comm.Barrier()
    loadtime = time.time() - start_time
    if rank == 0: print("Loading time: " + str(loadtime))
    if rank == 0: print("-------------------------- Run simulation ----------------------------")
    nest_sim.run()
    nrn_sim.run()
    nrn_sim.spikes_to_file(join(NRN_OUTPUT, "spike_detector-" + str(rank) + ".gdf"))
    if record_potentials:
        nrn_sim.voltage_to_file(join(NRN_OUTPUT, "multimeter-" + str(rank) + ".gdf"))

    if rank == 0: print("--------------------------- Recover output data -----------------------------")
    if save_to_h5:
        spikes_list = comm.gather(nest_sim.spikes, root=0)
        spikes_list_nrn = comm.gather(nrn_sim.spikes, root=0)

    if rank == 0:
        print("---------------------------- Store output data ------------------------------")
        if save_to_h5:
            nest_sim.spikes = dict()
            nrn_sim.spikes = dict()
            for x in spikes_list:
                nest_sim.spikes.update(x)
            for x in spikes_list_nrn:
                nrn_sim.spikes.update(x)

            with h5py.File(OUTPUT_FOLDER + '/simulation.h5', 'w') as f:
                f.attrs['sim_date'] = DATE
                input_group = f.create_group('input')
                input_group.attrs['seed'] = seed
                input_group.attrs['version'] = Dver
                input_group.attrs['t_trial'] = t_trial
                input_group.attrs['n_trials'] = n_trials
                input_group.attrs['dt'] = dt
                input_group.attrs['mpi_size'] = mpi_size
                input_group.create_dataset('param_files', data=file_)
                output_group = f.create_group('output')
                output_group.attrs['loadtime'] = loadtime
                output_group.attrs['nest_runtime'] = nest_sim.run_time
                output_group.attrs['nrn_runtime'] = nrn_sim.run_time
                spikes_group = output_group.create_group("nest_spike_times")
                for (gid, values) in nest_sim.spikes.items():
                    spikes_group.create_dataset(str(gid), data=np.array(values))
                if record_potentials:
                    potential_group = output_group.create_group("membrane_potentials")
                    for neuron in nest_sim.neurons.keys():
                        potential_group.create_dataset(str(neuron), (t_trial * n_trials / dt,), dtype='f8')
                spikes_group = output_group.create_group("nrn_spike_times")
                for (gid, values) in nrn_sim.spikes.items():
                    spikes_group.create_dataset(str(gid), data=np.array(values))
                if record_potentials:
                    potential_group = output_group.create_group("membrane_potentials")
                    for neuron in nrn_sim.neurons.keys():
                        potential_group.create_dataset(str(neuron), (t_trial * n_trials / dt,), dtype='f8')

        parameters = dict()
        parameters['sim_date'] = DATE
        parameters['seed'] = seed
        parameters['version'] = Dver
        parameters['t_trial'] = t_trial
        parameters['n_trials'] = n_trials
        parameters['dt'] = dt
        parameters['nest_runtime'] = nest_sim.run_time
        parameters['nrn_runtime'] = nrn_sim.run_time
        parameters['loadtime'] = loadtime
        parameters['mpi_size'] = mpi_size
        parameters['param_files'] = file_
        with open(OUTPUT_FOLDER + '/simulation.json', 'w') as f:
            json.dump(parameters, f, indent=4)

    if record_potentials and save_to_h5:
        comm.Barrier()
        for i in range(mpi_size):
            if rank == i:
                with h5py.File(OUTPUT_FOLDER + '/simulation.h5', 'r+') as f:
                    for (gid, values) in nest_sim.v_m.items():
                        f["/output/nest_membrane_potentials"][str(gid)][:] = np.array(values)
                    for (gid, values) in nrn_sim.v_m.items():
                        f["/output/nrn_membrane_potentials"][str(gid)][:] = np.array(values)

            comm.Barrier()


def progress_bar(current_percent, num_created):
    color = "\033[91m"  # red
    if current_percent > 33:
        color = "\033[93m"
    if current_percent > 66:
        color = "\033[92m"
    sys.stdout.write('\x1b[1A' * (int(mpi_size) - rank) + "\r[" + color
                     + "%s" % ("-" * current_percent + " " * (100 - current_percent)) + "\033[0m" + "] "
                     + str(current_percent) + "% " + str(num_created) + "\n" * (int(mpi_size) - rank))
    sys.stdout.flush()


if __name__ == '__main__':
    main()
