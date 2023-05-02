import h5py
import datetime
import json
import time
from os import makedirs
from os.path import join, exists, dirname, realpath
import numpy as np
import sys
from simwbm.nest_simulation import NestSimulation
from mpi4py import MPI  # import mpi after nest AND before neuron !!!
from functools import partial
# from simwbm.nrn_simulation import NeuronSimulation
import shutil
from argparse import ArgumentParser
from braindb import *


DATE = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
mpi_size = comm.Get_size()
print = partial(print, flush=True)

def parse_args(argv):
    parser = ArgumentParser(description="Simulate point neuron mouse brain network")

    parser.add_argument("--brain_version", dest="brain_version", required=True, type=str,
                        help="The version of the whole brain model.")
    parser.add_argument("--data_folder", dest="data_folder", required=True, type=str,
                        help="Folder which will contain the data and results")
    parser.add_argument("--simulation_name", dest="simulation_name", required=True, type=str,
                        help="Simulation output folder name.")
    parser.add_argument("--whole_brain_model", dest="whole_brain_model", required=False, type=str,
                        help="Whole brain model file location.")
    parser.add_argument("--database", dest="database", required=False, type=str,
                        help="Database file location.")
    parser.add_argument("--output_folder", dest="output_folder", required=False, type=str, default='save',
                        help="Output folder location.")
    parser.add_argument("--regions_to_simulate", dest="regions_to_simulate", required=False, type=str,
                        nargs='+', default=[],
                        help="Regions to simulate and record from.")
    parser.add_argument("--region_stimulus", dest="region_stimulus", required=False, type=str,
                        nargs='+', default=None,
                        help="Regions to use to stimulate the network.")
    parser.add_argument("--local_stimulus", dest="local_stimulus", required=False, type=str,
                        nargs='+', default=None,
                        help="Local to use to stimulate the network.")
    parser.add_argument("--background_noise", dest="background_noise", required=False, type=float, default=0.5,
                        help="Background noise frequency.")
    parser.add_argument("--weight_factor", dest="weight_factor", required=False, type=float, default=1.0,
                        help="Synapses weight scaling factor.")
    parser.add_argument("--print_progress", dest="print_progress", required=False, type=bool,
                        default=False, help="Display progress bar")
    parser.add_argument("--seed", dest="seed", required=False, type=int,
                        default=None, help="Seed for random generators.")
    parser.add_argument("--t_trial", dest="t_trial", required=False, type=float,
                        default=2100.0, help="Trial simulation time.")
    parser.add_argument("--n_trial", dest="n_trial", required=False, type=int,
                        default=1, help="Number of simulation trial.")
    parser.add_argument("--dt", dest="dt", required=False, type=float,
                        default=0.1, help="Simulation time step.")
    parser.add_argument("--record_potential", dest="record_potential", required=False, type=bool,
                        default=False, help="Record membrane potentials")
    parser.add_argument("--save_h5", dest="save_to_h5", required=False, type=bool,
                        default=False, help="Save results to h5 file.")

    result = parser.parse_args(argv)
    if result.whole_brain_model is None:
        result.whole_brain_model = join(result.data_folder, "whole_brain_model", 
                                        result.brain_version, "whole_brain_model_Nest.h5")
    if result.database is None:
        result.database = join(result.data_folder, "ElecDB.db")
    result.output_folder = join(result.output_folder, result.brain_version, result.simulation_name)

    if result.seed is not None:
        assert(result.seed > 0)

    assert (result.t_trial > 0 and result.n_trial > 0 and result.dt > 0.)
    return result


def poisson_uniform_2(size, rate, t, dt=0.1, offset=100.):
    nb_spikes = 2 * int(rate * t)
    spikes = np.around(np.cumsum(-np.log(np.random.rand(size, nb_spikes)) / rate, axis=1) + offset,
                     int(np.log10(1.0 / dt)))
    results = [[] for _ in range(size)]
    for i, loc_spikes in enumerate(spikes):
        for sp in loc_spikes:
            if sp < offset+t:
                results[i].append(sp)
    return results


def poisson_uniform(size, rate, t, dt=0.1, offset=100.):
    results = [[np.around(sp + offset,
                          int(np.log10(1.0 / dt)))] if sp < t else [] for sp in -np.log(np.random.rand(size)) / rate]
    overshoot = np.array([len(sp)>0 for sp in results])

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


def main(args):
    status = None
    if rank == 0:
        try:
            if exists(args.output_folder):
                shutil.rmtree(args.output_folder)
            makedirs(join(args.output_folder, "nest_output"))
            makedirs(join(args.output_folder, "neuron_output"))
            status = True
        except Exception as e:
            print(e)
            status = False
    status = comm.bcast(status, root=0)
    if not status:
        return

    np.random.seed(args.seed)

    simulate(args)


def simulate(args):
    # n_stochastic = 1
    start_time = time.time()
    if rank == 0: print("------------------------- Initializing Nest Simulator -------------------------")
    nest_sim = NestSimulation(args.t_trial, args.n_trial, args.seed, join(args.output_folder, "nest_output"), args.dt)
    # nrn_sim = NeuronSimulation(args.t_trial, args.n_trial, args.seed, join(dirname(realpath(__file__)), "NEURON"), args.dt, True)
    h5file = h5py.File(args.whole_brain_model, "r")
    gids = np.array(h5file["/neurons/default/gid"])
    db = BrainDB(args.database)
    gid_reg = np.array(h5file['neurons/regions'])
    ids = np.ones(gids.size, dtype=bool)
    if len(args.regions_to_simulate) > 0:
        request = Region.table.select("id",
                                      [Region.table.condition("full_name", args.regions_to_simulate, with_like=True),
                                       db.getMouseBrainConds(leafs=True)])
        regions_ids = np.array(db.execute(request)[0])[:, 0]
        ids = np.zeros(gids.size, dtype=bool)
        for reg in regions_ids:
            ids = np.logical_or(ids, gid_reg == reg)
    gids = gids[ids]
    inv_gids = -np.ones(gids[-1] + 1, dtype=np.int32)
    inv_gids[gids] = np.arange(gids.size)
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
    # nrn_sim.create_adex_neuron(neuron_parameters)
    if rank == 0:
        with open(join(args.output_folder, 'conversion.json'), 'w') as f:
            json.dump(nest_sim.neurons, f, indent=4)

    del neuron_parameters

    ########################### Stimulation #############################################
    if args.region_stimulus is not None:
        request = Region.table.select("id",
                                      [Region.table.condition("full_name", args.region_stimulus, with_like=True),
                                       db.getMouseBrainConds(leafs=True)])
        regions_ids = np.array(db.execute(request)[0])[:, 0]
        ids *= False
        for reg in regions_ids:
            ids = np.logical_or(ids, gid_reg == reg)
        StimulationIDs = np.array(h5file["/neurons/default/gid"])[ids].tolist()
        if rank == 0: print("Nb stimulation neurons: " + str(len(StimulationIDs)))
        for gid in StimulationIDs:
            nest_sim.create_parrot_neuron(gid)
        nest_sim.create_background_noise(17.3, StimulationIDs, weight=1.0, start=1200., stop=2200.)
        # spike_times = poisson_uniform(len(StimulationIDs), 17.3/1000., t=1000., dt=args.dt, offset=1200.)
        # for i, gid in enumerate(StimulationIDs):
        #     nrn_sim.create_current_input(spike_times[i], gid)
        gids = np.concatenate((gids, StimulationIDs))
    elif args.local_stimulus is not None:
        ptpos = np.array(
            np.float32(0.01 *np.vstack((np.array(h5file["/neurons/x"])[ids],
                                  np.array(h5file["/neurons/z"])[ids],
                                  np.array(h5file["/neurons/y"])[ids]))), order="C")
        request = Region.table.select("id",
                                      [Region.table.condition("full_name", args.local_stimulus, with_like=True),
                                       db.getMouseBrainConds(leafs=True)])
        regions_ids = np.array(db.execute(request)[0])[:, 0]
        ids *= False
        for reg in regions_ids:
            ids = np.logical_or(ids, gid_reg == reg)
        StimulationIDs = np.where(ids)[0]
        import genBrain
        gb = genBrain.genBrainSys()
        coords = gb.orthogonalPointProjection(ptpos, point_ids=StimulationIDs,
                                              camera_coord=[-76.56149291992188, 265.58148193359375, 377.6153259277344,
                                                            -1.1890138387680054, 13.543750762939453, -9.429346084594727,
                                                            0.5399147868156433, -0.6533979773521423, 0.5306252241134644])
        inside_ids_LOCAL = np.nonzero((np.fabs(coords[:, 0]) < 0.04) * (np.fabs(coords[:, 1]) < 0.04))[0]
        # StimulationIDs  = gids[StimulationIDs[inside_ids_LOCAL]]
        StimulationIDs = gids[-1] + 1 + np.arange(inside_ids_LOCAL.size)
        # radius_ = np.sqrt( ((0.8-h5file["IO/x"][:][gids])**2.0) + ((0.47-h5file["IO/y"][:][gids])**2.0) ) # whisker
        # radius_ = radius_[h5file["IO/gid"][:][gids]==11]
        # StimulationIext = 1000.0 *np.exp(-radius_*radius_/0.010)
        if rank == 0: print("Nb stimulation neurons: " + str(len(StimulationIDs)))
        for gid in StimulationIDs:
            nest_sim.create_parrot_neuron(gid)
        nest_sim.create_background_noise(17.3, StimulationIDs, weight=1.0, start=1200., stop=2200.)
        del coords, ptpos

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
        if args.print_progress:
            sys.stdout.write("\n" * mpi_size)
            sys.stdout.flush()
    percent_done = -1
    num_created = 0
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
        if args.print_progress and current_percent > percent_done:
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
            local_excitatory = is_excitatory[inv_gids[pre_ids]]
            receptors = 3 - local_excitatory * 2
            weights = syn_params[4][start:i_win] * args.weight_factor
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
            # nrn_sim.set_receptors(curr_id, postsyn_params)

            nest_sim.create_synapse(pre_synaptic_parameters, pre_ids, curr_id)
            # nrn_sim.create_synapse(pre_synaptic_parameters, pre_ids, curr_id)
            receptors += 1
            weights[np.where(receptors == 2)] *= 0.4 + 0.4 * (1 - is_excitatory[inv_gids[curr_id]])
            weights[np.where(receptors == 4)] *= 0.75 * is_excitatory[inv_gids[curr_id]]
            pre_synaptic_parameters['weight'] = weights.tolist()
            pre_synaptic_parameters['receptor_type'] = receptors.tolist()
            nest_sim.create_synapse(pre_synaptic_parameters, pre_ids, curr_id)
            # nrn_sim.create_synapse(pre_synaptic_parameters, pre_ids, curr_id)
            num_created += 2 * (i_win - start)
        start = i_win
    if args.print_progress:
        progress_bar(100, num_created)
    mean_weight = np.mean(syn_params[4][syn_params[4] > 0])
    h5file.close()
    del pre_synaptic_parameters, h5file, gids, pre_gids, post_gids, pre_ids, filter_post, filter_pre
    ########################### Background Noise ########################################
    comm.Barrier()
    if args.background_noise > 0.0:
        if rank == 0: print("-------------------------- Create background noise ----------------------------")
        nest_sim.create_background_noise(args.background_noise, nest_sim.neurons, mean_weight)
        # nrn_sim.create_background_noise(args.background_noise, nest_sim.neurons, mean_weight)

    if rank == 0: print("-------------------------- Create spike detector ----------------------------")
    nest_sim.create_spike_detector(nest_sim.neurons)
    # nrn_sim.create_spike_detector(nrn_sim.neurons)
    # nest_sim.create_spike_detector(nest_sim.source)

    if args.record_potential:
        if rank == 0: print("-------------------------- Create multimeter ----------------------------")
        nest_sim.create_multimeter(nest_sim.neurons)
        # nrn_sim.create_multimeter(nrn_sim.neurons)
    comm.Barrier()
    loadtime = time.time() - start_time
    if rank == 0: print("Loading time: " + str(loadtime))
    if rank == 0: print("-------------------------- Run simulation ----------------------------")
    nest_sim.run()
    # nrn_sim.run()
    # nrn_sim.spikes_to_file(join(join(args.output_folder, "neuron_output"), "spike_detector-" + str(rank) + ".gdf"))

    if rank == 0: print("---------------------------- Store output data ------------------------------")
    if args.save_to_h5:
        spikes_list = comm.gather(nest_sim.spikes, root=0)
        if rank == 0:
            nest_sim.spikes = dict()
            for x in spikes_list:
                nest_sim.spikes.update(x)
            with h5py.File(join(args.output_folder, 'simulation.h5'), 'w') as f:
                f.attrs['sim_date'] = DATE
                input_group = f.create_group('input')
                input_group.attrs['seed'] = args.seed
                input_group.attrs['version'] = args.brain_version
                input_group.attrs['t_trial'] = args.t_trial
                input_group.attrs['n_trials'] = args.n_trial
                input_group.attrs['dt'] = args.dt
                input_group.attrs['mpi_size'] = mpi_size
                input_group.create_dataset('param_files', data=args.whole_brain_model)
                output_group = f.create_group('output')
                output_group.attrs['loadtime'] = loadtime
                output_group.attrs['nest_runtime'] = nest_sim.run_time
                spikes_group = output_group.create_group("nest_spike_times")
                for (gid, values) in nest_sim.spikes.items():
                    spikes_group.create_dataset(str(gid), data=np.array(values))
                if args.record_potential:
                    potential_group = output_group.create_group("membrane_potentials")
                    for neuron in nest_sim.neurons.keys():
                        potential_group.create_dataset(str(neuron), (args.t_trial * args.n_trial / args.dt,), dtype='f8')
        if args.record_potential:
            comm.Barrier()
            for i in range(mpi_size):
                if rank == i:
                    with h5py.File(join(args.output_folder, 'simulation.h5'), 'r+') as f:
                        for (gid, values) in nest_sim.v_m.items():
                            f["/output/nest_membrane_potentials"][str(gid)][:] = np.array(values)
                comm.Barrier()
    if rank == 0:
        parameters = dict()
        parameters['sim_date'] = DATE
        parameters['seed'] = args.seed
        parameters['version'] = args.brain_version
        parameters['t_trial'] = args.t_trial
        parameters['n_trials'] = args.n_trial
        parameters['dt'] = args.dt
        parameters['nest_runtime'] = nest_sim.run_time
        # parameters['neuron_runtime'] = nrn_sim.run_time
        parameters['loadtime'] = loadtime
        parameters['mpi_size'] = mpi_size
        parameters['param_files'] = args.whole_brain_model
        with open(join(args.output_folder, 'simulation.json'), 'w') as f:
            json.dump(parameters, f, indent=4)


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
    main(parse_args(sys.argv[1:]))
