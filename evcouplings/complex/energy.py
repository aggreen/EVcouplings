"""
Protocol for matching putatively interacting sequences
in protein complexes to create a concatenated sequence
alignment

Authors:
    Anna Green
"""
import ctypes
import numpy as np
import pandas as pd
import numba
import multiprocessing as mp

from copy import deepcopy
from evcouplings.complex.alignment import (
    write_concatenated_alignment, modify_complex_segments
)
from evcouplings.align import Alignment, map_matrix
from evcouplings.align.protocol import modify_alignment
from evcouplings.couplings.model import CouplingsModel
from evcouplings.utils.system import (
    create_prefix_folders, insert_dir
)
from evcouplings.complex.protocol import modify_complex_segments
from evcouplings.couplings.protocol import mean_field

ID_THRESH = .8
CPU_COUNT = 2

def _write_format(item_list):
    return "\t".join(list(map(str, item_list))) + "\n"

def _get_index_for_species(alignment,
                           monomer_information,
                           species):
    """
    Gets the indices of each sequence identifier
    in the alignment matrix

    Parameters
    ----------
    alignment: EVcouplings.align.Alignment
    monomer_information:
    species: str

    Returns
    -------
    pd.Dataframe
        With columns aln_index, id

    """

    species_df = monomer_information.get_group(species)

    if len(species_df)==0 or species_df.empty:
        raise ValueError("species not present in pd.DataFrame")

    alignment_index = [alignment.id_to_index[x] for x in species_df.id]

    return alignment_index

def _E_matrix_to_file(E_matrix, first_species_index,
                      second_species_index, species):
    """
    Writes an N_seqs by M_seqs matrix to a
    csv file


    """
    all_list = []
    for idx_i, id_i in enumerate(first_species_index):
        for idx_j, id_j in enumerate(second_species_index):
            E = E_matrix[idx_i, idx_j]
            l = [species, id_i, id_j, E]
            all_list.append(l)
    return all_list

def inter_energy_per_species(Jij, Jij_dim, species, sequences_X, sequences_Y,
                             first_alignment_indices, second_alignment_indices,
                             positions_i, positions_j):

    """

    Parameters
    ----------
    model
    first_alignment_indices
    second_alignment_indices
    alignment_1
    alignment_2

    Returns
    -------

    """
    # call H
    Jij = np.frombuffer(Jij, dtype=np.float64)
    Jij_dim = np.frombuffer(Jij_dim, dtype=np.int32)

    matrix = inter_sequence_hamiltonians(sequences_X, sequences_Y, Jij, Jij_dim, positions_i, positions_j)

    list_of_E = _E_matrix_to_file(
        matrix, first_alignment_indices, second_alignment_indices, species
    )

    return list_of_E

# def inter_energy_per_species(i,queue):
#     queue.put('ASDF')
#     return 'ASDF'

#@numba.jit(nopython=True)
def inter_sequence_hamiltonians(sequences_X, sequences_Y, J_ij, Jij_dim, positions_i, positions_j):
    """
    Calculates the Hamiltonian of the global probability distribution P(A_1, ..., A_L)
    for a given sequence A_1,...,A_L from J_ij and h_i parameters

    Parameters
    ----------
    sequences_X, sequences_Y : np.array
        Sequence matrix for which Hamiltonians will be computed
    J_ij: np.array
        L x L x num_symbols x num_symbols J_ij pair coupling parameter matrix
    positions: list of tuple of indices of J_ij matrix to sum

    Returns
    -------
    np.array
        Float matrix of size len(sequences_x) x len(sequences_y), where each position i,j corresponds
        to the sum of Jijs between the given positions in sequences_X[i] and sequences_Y[j]
    """
    # iterate over sequences
    N_x, L_x = sequences_X.shape
    N_y, L_y = sequences_Y.shape

    H = np.zeros((N_x, N_y))

    for s_x in range(N_x):
        A_x = sequences_X[s_x]
        for s_y in range(N_y):
            A_y = sequences_Y[s_y]

            Jij_sum = 0.0

            for ali_i, model_i in positions_i:
                for ali_j, model_j in positions_j:
                    #print(model_i,model_j,ali_i,ali_j,A_x[ali_i],A_y[ali_j])
                    #print(Jij_dim)
                    Jij_sum = Jij_sum + J_ij[
                        np.ravel_multi_index((model_i, model_j, A_x[ali_i], A_y[ali_j]), Jij_dim)
                    ]
            H[s_x, s_y] = Jij_sum

    return H

def initialize_alignment(first_monomer_info, second_monomer_info,
                          **kwargs):
    """
    Create a concatenated sequence alignment that combines the best hits from every
    species that share at least 80% identity with the target

    Parameters
    ----------
    first_monomer_info,second_monomer_info: pd.DataFrame
        With columns id, species, identity_to_query

    Returns
    -------
    aln_outcfg: dict
        File name keys and paths to created files
    current_id_pairs: pd.DataFrame
        With columns id_1 and id_2
    """

    def _most_similar(data):

        data = data.query("identity_to_query > @ID_THRESH")
        most_similar_in_species = data.sort_values(by="identity_to_query").groupby("species").last()
        most_similar_in_species["species"] = most_similar_in_species.index

        return most_similar_in_species

    prefix = kwargs["prefix"]

    most_similar_in_species_1 = _most_similar(first_monomer_info)

    most_similar_in_species_2 = _most_similar(second_monomer_info)

    species_intersection = most_similar_in_species_1.merge(
        most_similar_in_species_2,
        how="inner",  # takes the intersection
        on="species",  # merges on species identifiers
        suffixes=("_1", "_2")
    )

    target_seq_id, target_seq_index, raw_ali, mon_ali_1, mon_ali_2 = \
        write_concatenated_alignment(
            species_intersection,
            kwargs["first_alignment_file"],
            kwargs["second_alignment_file"],
            kwargs["first_focus_sequence"],
            kwargs["second_focus_sequence"]
        )

    raw_alignment_file = prefix + "_raw.fasta"
    with open(raw_alignment_file, "w") as of:
        raw_ali.write(of)

    aln_outcfg, _ = modify_alignment(
        raw_ali,
        target_seq_index,
        target_seq_id,
        kwargs["first_region_start"],
        **kwargs
    )
    aln_outcfg["focus_sequence"] = target_seq_id
    aln_outcfg["raw_alignment_file"] = raw_alignment_file

    current_id_pairs = species_intersection[["id_1","id_2","species"]]

    return aln_outcfg, current_id_pairs


class ProcessWriteResult(mp.Process):

    def __init__(self, writer_queue, outfilename):
        mp.Process.__init__(self)
        self.writer_queue = writer_queue
        self.outfile = outfilename

    def run(self):
        while True:
            with open(self.outfile, "w") as of:

                result = self.writer_queue.get()
                if result is None:
                    break
                else:
                    for i in result:
                        of.write(_write_format(i))

class ProcessHamiltonianCalc(mp.Process):

    def __init__(self, worker_queue, writer_queue, shared_Jij_arr, shared_Jij_dim):
        mp.Process.__init__(self)
        self.writer_queue = writer_queue
        self.worker_queue = worker_queue
        self.J_ij = shared_Jij_arr
        self.dim = shared_Jij_dim

    def run(self):
        while True:
            work = self.worker_queue.get()
            if work is None:
                break
            else:
                result = inter_energy_per_species(self.J_ij,self.dim,**work)
                self.writer_queue.put(result)

def best_pairing(first_monomer_info, second_monomer_info,
                 N_pairing_iterations, N_increase_per_iteration,
                 **kwargs):
    """

    Parameters
    ----------
    first_monomer_info
    second_monomer_info
    N_pairing_iterations
    N_increase_per_iteration
    kwargs

    Returns
    -------
    paired_ids: pd.DataFrame
        has columns id_1 and id_2
    """

    COLUMN_NAMES = ["species", "id_1", "id_2", "E"]


    def _create_mapped_seg_tuples(alignment,model,**current_kwargs):

        # get the upper case positions to use
        segment_1_positions = current_kwargs["segments"][0][5]
        segment_2_positions = current_kwargs["segments"][1][5]
        segment_id = ["A"] * len(segment_1_positions) + ["B"] * len(segment_2_positions)

        #
        alignment_positions = list(range(1,alignment.L+1))

        # filter alignment.L positions to make monomer alignment positions to use
        focus_cols = np.array([
            c.isupper() and c not in [
                alignment._insert_gap
            ]
            for c in alignment.matrix[0,:]
        ])
        focus_cols = [np.nan if x is False else True for x in focus_cols]

        model_index = [model.index_map[v] if v in model.index_map else np.nan for v in alignment_positions]

        mapping_table = pd.DataFrame.from_dict({
            "segments_idx1" : segment_1_positions + segment_2_positions,
            "segments_idx0": [x-1 for x in segment_1_positions + segment_2_positions],
            "segment_id": segment_id,
            "alignment_idx1": alignment_positions,
            "alignment_idx0": [x-1 for x in alignment_positions],
            "focus_columns": focus_cols,
            "model_index": model_index
        })
        mapping_table.to_csv("TEST.csv")
        mapping_table = mapping_table.dropna()

        segment_1 = mapping_table.query("segment_id == 'A'")
        segment_2 = mapping_table.query("segment_id == 'B'")

        #return tuple of 0-indexed segment of monomer alingment positions to use
        #plus model positions
        return (
            list(zip(segment_1.segments_idx0.astype(int),segment_1.model_index.astype(int))),
            list(zip(segment_2.segments_idx0.astype(int),segment_2.model_index.astype(int)))
        )

    outcfg = {}

    # read in the monomer alignments

    with open(kwargs["first_alignment_file"],) as inf:
        alignment_1 = Alignment.from_file(inf)
        alignment_1.matrix_mapped = map_matrix(
            alignment_1.matrix, alignment_1.alphabet_map
        )

    with open(kwargs["second_alignment_file"]) as inf:
        alignment_2 = Alignment.from_file(inf)
        alignment_2.matrix_mapped = map_matrix(
            alignment_2.matrix, alignment_2.alphabet_map
        )

    # prepare the prefix logic
    prefix = kwargs["prefix"]
    aux_prefix = insert_dir(prefix, "aux", rootname_subdir=False)
    create_prefix_folders(aux_prefix)

    # will create a new prefix for every iteration through pairing
    initial_prefix = aux_prefix + "_iter0"

    # current_kwargs will be used to pass correct kwargs to mean_field protocol
    current_kwargs = deepcopy(kwargs)
    current_kwargs["prefix"] = initial_prefix

    # species set
    species_set = list(set(first_monomer_info.species.dropna()).intersection(
        set(second_monomer_info.species.dropna())
    ))

    outcfg["species_union_list_file"] = aux_prefix + "_species_union.csv"
    with open(outcfg["species_union_list_file"], "w") as of:
        for idx, species in enumerate(species_set):
            of.write(str(idx) + "," + species + "\n")

    # create and write a starting alignment
    starting_aln_cfg, current_id_pairs = initialize_alignment(
        first_monomer_info, second_monomer_info,
        **current_kwargs
    )

    # group the monomer information tables by species for easier lookup
    first_monomer_groups = first_monomer_info.groupby("species")
    second_monomer_groups = second_monomer_info.groupby("species")

    # update current kwargs with output of alignment generation
    outcfg["alignment_file_iter0"] = starting_aln_cfg["alignment_file"]
    current_kwargs["focus_sequence"] = starting_aln_cfg["focus_sequence"]
    current_kwargs["alignment_file"] = starting_aln_cfg["alignment_file"]
    current_kwargs = modify_complex_segments(current_kwargs, **current_kwargs)

    for iteration in range(1):

        ### Run the mean field
        print("iteration "+str(iteration))
        # #run the mean field calculation on the alignment
        # mean_field_outcfg = mean_field(**current_kwargs)
        # outcfg["model_file_{}".format(iteration)] = mean_field_outcfg["model_file"]
        # outcfg["raw_ec_file_{}".format(iteration)] = mean_field_outcfg["raw_ec_file"]
        # outcfg["ec_file_{}".format(iteration)] = mean_field_outcfg["ec_file"]

        # read in the parameters of mean field
#        model = CouplingsModel(mean_field_outcfg["model_file"])

        model = CouplingsModel("output/complex_238/concatenate/aux/complex_238_iter0.model")
        shared_Jij_arr = mp.RawArray('d',model.J_ij.flatten())
        shared_Jij_dim = mp.RawArray(ctypes.c_int,np.array(model.J_ij.shape))
        print(model.J_ij.shape)

        ### Set up variable names
        current_iteration_E_table = "energy_output_file_iter{}".format(iteration)
        outcfg[current_iteration_E_table] = prefix + "_energy_iter{}.csv".format(iteration)

        # read in the concatenated alignment and figure out which positions
        # to use for energy calculation
        with open(current_kwargs["alignment_file"]) as inf:
            concat_aln = Alignment.from_file(inf)

        filtered_segment_1, filtered_segment_2 = _create_mapped_seg_tuples(
            concat_aln, model, **current_kwargs
        )

        ### Set up our parallel processing magic
        worker_queue = mp.JoinableQueue() #queue of things to be calculated
        writer_queue = mp.Queue() #queue of results to be written
        processes = [] # list of processes

        for i in range(CPU_COUNT):
            p = ProcessHamiltonianCalc(worker_queue, writer_queue, shared_Jij_arr, shared_Jij_dim)
            processes.append(p)
            p.start()

        # Now generate jobs to put in the worker queue
        for species in species_set:
            print(species)
            # get the indices in the alignment matrix of our sequences of interest
            first_alignment_indices = _get_index_for_species(
                alignment_1, first_monomer_groups, species
            )
            second_alignment_indices = _get_index_for_species(
                alignment_2, second_monomer_groups, species
            )

            # get the X sequences
            sequences_X = alignment_1.matrix_mapped[first_alignment_indices, :]

            # get the Y sequences
            sequences_Y = alignment_2.matrix_mapped[second_alignment_indices, :]

            worker_queue.put({
                "species":species,
                "sequences_X": sequences_X,
                "sequences_Y": sequences_Y,
                "first_alignment_indices": first_alignment_indices,
                "second_alignment_indices": second_alignment_indices,
                "positions_i": filtered_segment_1,
                "positions_j": filtered_segment_2
            }
            )
        print("waiting for workers")
        # make sure all worker processes are done
        #worker_queue.join()

        print("workers done")
        # put a termination signal for each processes in the worker queue
        for i in range(CPU_COUNT):
            worker_queue.put(None)

        energy_df = pd.read_csv(
            outcfg[current_iteration_E_table], index_col=None, names=COLUMN_NAMES, sep="\t"
        )

        energy_df = energy_df.sort_values("E", ascending=False)
        print(energy_df.head())

        # take the top M E
        M = N_increase_per_iteration * (iteration + 1)
        top_E = energy_df.sort_values("E",ascending=False).iloc[0:M,:]

        # change the index back to identifier to writing of alignment
        def _index_to_id(alignment, df, COLUMN):

            indices = list(map(int, df.loc[:, COLUMN].dropna()))
            ids = alignment.ids[indices]
            df.loc[:, COLUMN] = list(ids)
            return df
        top_E = _index_to_id(alignment_1, top_E, "id_1")
        top_E = _index_to_id(alignment_2, top_E, "id_2")
        current_id_pairs = pd.concat([current_id_pairs, top_E])

        # write the new concatenated alignment and filter it
        target_seq_id, target_seq_index, raw_ali, mon_ali_1, mon_ali_2 = \
                    write_concatenated_alignment(
                        current_id_pairs,
                        kwargs["first_alignment_file"],
                        kwargs["second_alignment_file"],
                        kwargs["first_focus_sequence"],
                        kwargs["second_focus_sequence"]
                    )

        # update prefixes
        current_prefix = aux_prefix + "_iter{}".format(iteration + 1)
        current_kwargs["prefix"] = current_prefix

        raw_alignment_file = current_prefix + "_raw.fasta"
        with open(raw_alignment_file, "w") as of:
            raw_ali.write(of)

        aln_outcfg, _ = modify_alignment(
                    raw_ali,
                    target_seq_index,
                    target_seq_id,
                    kwargs["first_region_start"],
                    **current_kwargs
                )

        # update the configuration file for input into MFDCA
        current_kwargs["alignment_file"] = aln_outcfg["alignment_file"]
    print("all done")
    # retun the paired ids and output configuration
    return current_id_pairs, outcfg

