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
from evcouplings.couplings.mapping import (
    Segment, SegmentIndexMapper
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

    def ravel_idx(idx_tuple, array_shape):
        current_num = 0
        for idx, pos in enumerate(idx_tuple):
            array_to_multiply = array_shape[idx + 1::]
            product = 1
            for i in array_to_multiply:
                product = product * i
            current_num += pos * product
        return (int(current_num))

    # iterate over sequences
    N_x, L_x = sequences_X.shape
    N_y, L_y = sequences_Y.shape

    H = np.zeros((N_x, N_y))

    for s_x in range(N_x):
        A_x = sequences_X[s_x, :]
        print("the real sequences",A_x,sequences_X)

        for s_y in range(N_y):
            A_y = sequences_Y[s_y, :]

            Jij_sum = 0.0
            print(positions_i)
            for ali_i, model_i in positions_i:
                print(ali_i, A_x[ali_i])
                for ali_j, model_j in positions_j:
                    if A_x[ali_i] == -1:
                        continue
                    else:
                        i = A_x[ali_i]
                    print("ended up with", i, ali_i, A_x[ali_i])
                    if A_y[ali_j] == -1:
                        continue
                    else:
                        j = A_y[ali_j]

                    current_num = 0

                    for idx, pos in enumerate([model_i, model_j, i, j]):
                        array_to_multiply = Jij_dim[idx + 1::]
                        product = 1
                        for i in array_to_multiply:
                            product = product * i
                        current_num += pos * product
                    index = int(current_num)
                    #print(index, i)

                    #print(model_i, model_j, ali_i, i, j)
                    Jij_sum = Jij_sum + J_ij[index]
                    i = np.nan
                    j = np.nan

            break

            H[s_x, s_y] = Jij_sum
        break
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

# get the energy difference
def _energy_diff(df):
    """
    For each potential pairing of id_1, calculate the difference
    in statistical energy between that pairing and then next highest
    scoring pairing, and store as E_diff_1

    For each potential pairing of id_1, calculate the difference
    in statistical energy between that pairing and then next highest
    scoring pairing, and store as E_diff_2

    Params
    ------
    df: pd.DataFrame
        contains columns [id_1, id_2, E]

    Returns
    -------
    df: pd.DataFrame
        contains columns [id_1, id_2, E, E_diff_1, E_diff_2]
    """

    final_df = pd.DataFrame(columns=list(df.columns) + ["E_diff_1", "E_diff_2"])
    grouped_df = df.groupby("id_1")
    for id_1, _df in grouped_df:

        # sort the dataframe by statistical energy
        _sorted_df = _df.sort_values("E", ascending=False)
        _sorted_df.loc[:, "E_diff"] = 0.
        _sorted_df = _sorted_df.reset_index(drop=True)

        # subtract the statE of each row from the previous row
        for index in _sorted_df.index[0:-1]:
            print(_sorted_df.loc[index, "E"],_sorted_df.loc[index+1, "E"])
            #print()
            _sorted_df.loc[index, "E_diff_1"] = _sorted_df.loc[index, "E"] - _sorted_df.loc[index + 1, "E"]

        final_df = pd.concat([final_df, _sorted_df])

    grouped_df = final_df.groupby("id_2")
    for id_2, _df in grouped_df:

        # sort the dataframe by statistical energy
        _sorted_df = _df.sort_values("E", ascending=False)
        _sorted_df.loc[:, "E_diff"] = 0.
        _sorted_df = _sorted_df.reset_index(drop=True)

        # subtract the statE of each row from the previous row
        for index in _sorted_df.index[0:-1]:
            print(_sorted_df.loc[index, "E"],_sorted_df.loc[index+1, "E"])
            #print()
            _sorted_df.loc[index, "E_diff_2"] = _sorted_df.loc[index, "E"] - _sorted_df.loc[index + 1, "E"]

    return final_df.sort_values("E_diff_1", ascending=False)


# get the energy difference
def df_to_assigned_pairs(df):
    """
    implements the Bitbol 2016 Figure S12 pairing algorithm

    Params
    ------
    df: pd.DataFrame
        contains columns [id_1, id_2, E]

    Returns
    -------
    df: pd.DataFrame
        contains columns [id_1, id_2, E, E_diff, E_delta]
        the best pairings as determined by the alorgithm
    """
    final_df = pd.DataFrame(columns=list(df.columns) + ["E_delta"])
    grouped_df = df.groupby("species")

    for species, _df in grouped_df:

        # END CONDITION: all of id_1 are paired
        _to_be_paired = _df.copy()
        n_paired = 0
        latest_E_delta = np.nan

        while len(_to_be_paired) > 0:

            # get the highest E pair
            sort = _to_be_paired.sort_values("E", ascending=False)
            highest_pair = sort.iloc[0, :]
            first_id = int(highest_pair.id_1)
            second_id = int(highest_pair.id_2)

            # remove all related
            _to_be_paired = _to_be_paired.query("id_1 != @first_id and id_2 != @second_id")

            # score based on Bitbol confidence score
            if len(_to_be_paired) > 0:
                highest_pair["E_delta"] = min(highest_pair["E_diff_1"], highest_pair["E_diff_2"]) / (n_paired + 1)
                latest_E_delta = highest_pair["E_delta"]
            else:
                highest_pair["E_delta"] = latest_E_delta

            n_paired += 1

            final_df = pd.concat([final_df, highest_pair.to_frame().transpose()])

    return final_df.sort_values("E_delta", ascending=False)

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

    def _create_mapped_seg_tuples(params, **current_kwargs):

        first_segment = Segment.from_list(current_kwargs["segments"][0])
        second_segment = Segment.from_list(current_kwargs["segments"][1])

        index_start = first_segment.region_start
        r = SegmentIndexMapper(
            True,  # use focus mode
            index_start,  # first index of first segment
            first_segment,
            second_segment
        )
        c = r.patch_model(model=params)

        index_map = c.index_map
        segment_1 = {k: v for k, v in index_map.items() if "A_1" in k}
        segment_2 = {k: v for k, v in index_map.items() if "B_1" in k}

        segment_1_aln_positions = [k[1] - 1 for k, v in segment_1.items()]
        segment_2_aln_positions = [k[1] - 1 for k, v in segment_2.items()]
        segment_1_model_positions = [v for k, v in segment_1.items()]
        segment_2_model_positions = [v for k, v in segment_2.items()]

        # return tuple of 0-indexed segment of monomer alingment positions to use
        # plus model positions
        return (
            list(sorted(zip(segment_1_aln_positions, segment_1_model_positions))),
            list(sorted(zip(segment_2_aln_positions, segment_2_model_positions)))
        )

    # change the index back to identifier to writing of alignment
    def _index_to_id(alignment, df, COLUMN):

        indices = list(map(int, df.loc[:, COLUMN].dropna()))
        ids = alignment.ids[indices]
        df.loc[:, COLUMN] = list(ids)
        return df

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
    print("alignment initialized")
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

        model = CouplingsModel("complex_238_dist_paired/couplings/complex_238.model")
        shared_Jij_arr = mp.RawArray('d',model.J_ij.flatten())
        shared_Jij_dim = mp.RawArray(ctypes.c_int,np.array(model.J_ij.shape))
 
        ### Set up variable names
        current_iteration_E_table = "energy_output_file_iter{}".format(iteration)
        outcfg[current_iteration_E_table] = prefix + "_energy_iter{}.csv".format(iteration)

        # read in the concatenated alignment and figure out which positions
        # to use for energy calculation
        with open(current_kwargs["alignment_file"]) as inf:
            concat_aln = Alignment.from_file(inf)

        filtered_segment_1, filtered_segment_2 = _create_mapped_seg_tuples(
            model, **current_kwargs
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
            sequences_X = alignment_1.matrix_mapped[first_alignment_indices, :] - 1

            # get the Y sequences
            sequences_Y = alignment_2.matrix_mapped[second_alignment_indices, :] - 1
            print(sequences_X.shape, sequences_X[0,:])
            print(sequences_Y.shape, sequences_Y[0,:])
            print(filtered_segment_1)

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

        # read in the energy dataframe and determine which pairs to take
        energy_df = pd.read_csv(
            outcfg[current_iteration_E_table], index_col=None, names=COLUMN_NAMES, sep="\t"
        )
        energy_df = energy_df.sort_values("E", ascending=False)
        energy_df_with_diff = _energy_diff(energy_df)
        top_E = df_to_assigned_pairs(energy_df_with_diff)
        top_E = _index_to_id(alignment_1, top_E, "id_1")
        top_E = _index_to_id(alignment_2, top_E, "id_2")
        current_iteration_diff_E_table = "energy_output_file_iter{}_diff".format(iteration)
        outcfg[current_iteration_diff_E_table] = prefix + "_energy_iter{}_diff.csv".format(iteration)
        top_E.to_csv(outcfg[current_iteration_diff_E_table], index=False)

        M = N_increase_per_iteration * iteration
        current_id_pairs = top_E.iloc[0:M,:]

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

