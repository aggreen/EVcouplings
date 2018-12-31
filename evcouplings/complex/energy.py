"""
Protocol for matching putatively interacting sequences
in protein complexes to create a concatenated sequence
alignment

Authors:
    Anna Green
"""
import numpy as np
import pandas as pd
import numba

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
from evcouplings.couplings.protocol import mean_field, standard, infer_plmc, complex

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

    matrix = inter_sequence_hamiltonians(sequences_X, sequences_Y, Jij, Jij_dim, positions_i, positions_j)
    print(matrix)
    list_of_E = _E_matrix_to_file(
        matrix, first_alignment_indices, second_alignment_indices, species
    )

    return list_of_E


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

def get_pairing_idx(h):
    """

    Gets the indices of the best sequence pairings in the statistical energy matrix.
    N. B. This assumes that the MOST NEGATIVE pairs in terms of statistical energy are the most favorable pairs
    (mean field convention, inverse of plmc convention)

    Parameters
    ----------
    h: np.ndarray
        N1 x N2 dimensional array, where N1 is the number of homologs of sequence 1 in the species,
        and N2 is the number of homologs of sequence 2 in the species

    Returns
    -------


    """

    k = deepcopy(h)
    pairings_idx = []
    E = []

    # the number of pairings returned is equal to min(N1, N2)
    for i in range(0, min(k.shape)):

        #
        best_pair = np.unravel_index(k.argmin(), k.shape)

        E.append(k.min())
        pairings_idx.append(best_pair)
        k[best_pair[0],:] = LARGE_VALUE
        k[:,best_pair[1]] = LARGE_VALUE

    if len(pairings_idx) ==0:
        return ([],[]),E

    return zip(*pairings_idx), E

def get_indices(model, alignment, config):
    """
    Parameters
    ----------

    Returns
    -------
    """

    # create a segment index mapper for the two segments in the alignmmen
    segments = [Segment.from_list(config["segments"][0]), Segment.from_list(config["segments"][1])]
    first_segment = segments[0]
    index_start = first_segment.region_start
    r = SegmentIndexMapper(
        True,  # use focus mode
        index_start,  # first index of first segment
        *segments
    )

    positions_alignment_index = []
    positions_model_index = []

    # iterate through all pairs of positions in the alignment
    for i in r.target_pos:
        for j in r.target_pos:

            # for all inter sequence pairs
            if "A_1" in i and "B_1" in j:

                # get the index in the MultiSegment name - eg (A_1, i)
                model_index_i = r.target_to_model[i]
                model_index_j = r.target_to_model[j]

                if model_index_i in model.index_map and model_index_j in model.index_map:

                    positions_model_index.append(
                        (model.index_map[model_index_i], model.index_map[model_index_j])
                    )
                    positions_alignment_index.append(
                        (i[1] - 1, j[1] - 1)
                    )

    return positions_alignment_index, positions_model_index

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

    ## if a seed alignment is provided, use that as a place to start
    if kwargs["seed_alignment"] is not None:
        current_kwargs["alignment_file"] = kwargs["seed_alignment"]

    for iteration in range(0,N_pairing_iterations):

        ### Run the mean field
        print("iteration "+str(iteration))

        if kwargs["input_model_file"] is None:

            if kwargs["ec_calculation_method"] is "mean_field":
                print("running mean field")
                #run the mean field calculation on the alignment
                temporary_outcfg = mean_field(**current_kwargs)

            elif kwargs["ec_calculation_method"] is "plmc":
                print("running plmc")
                # run the plmc calculation on the alignment
                temporary_outcfg = complex(**current_kwargs)

            else:
                raise(NotImplementedError, "ec calculation method provided is not implemented")

            outcfg["model_file_{}".format(iteration)] = temporary_outcfg["model_file"]
            outcfg["raw_ec_file_{}".format(iteration)] = temporary_outcfg["raw_ec_file"]
            outcfg["ec_file_{}".format(iteration)] = temporary_outcfg["ec_file"]

            # read in the parameters of mean field
            model = CouplingsModel(temporary_outcfg["model_file"])

        else:
            model = CouplingsModel(kwargs["input_model_file"])
 
        ### Set up variable names
        current_iteration_E_table = "energy_output_file_iter{}".format(iteration)
        outcfg[current_iteration_E_table] = prefix + "_energy_iter{}.csv".format(iteration)

        # read in the concatenated alignment and figure out which positions
        # to use for energy calculation
        with open(current_kwargs["alignment_file"]) as inf:
            concat_aln = Alignment.from_file(inf)

        positions_aln_index, positions_model_index = get_indices(model, concat_aln, current_kwargs)
        #print(positions_aln_index, positions_model_index)

        #variables to captures output of sequence hamiltonians
        seq_i, seq_j, species_list, E = [],[],[],[]

        for species in species_set:
        #for species in ["Burkholderiales bacterium RIFCSPLOWO2_02_FULL_57_36"]:
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

            h = inter_sequence_hamiltonians(sequences_X, sequences_Y, model, positions_aln_index, positions_model_index)
            (pairing_idx_i, pairing_idx_j), _E = get_pairing_idx(h)

            seq_i += [alignment_1.ids[first_alignment_indices[x]] for x in pairing_idx_i]
            seq_j += [alignment_2.ids[second_alignment_indices[x]] for x in pairing_idx_j]
            E += _E
            species_list += [species]*len(_E)

            print(h)

        #print(len(seq_i), len(seq_j), len(E), len(species_list))
        #print(seq_i, seq_j, E, species_list)

        # read in the energy dataframe and determine which pairs to take
        top_E = pd.DataFrame({
            "species": species_list,
            "id_1": seq_i,
            "id_2": seq_j,
            "E": E
        })

        current_iteration_diff_E_table = "energy_output_file_iter{}".format(iteration)
        outcfg[current_iteration_diff_E_table] = prefix + "_energy_iter{}.csv".format(iteration)
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


def inter_sequence_hamiltonians(sequences_X, sequences_Y, m, positions_aln_index, positions_model_index):
    """
    Calculates the Hamiltonian of the global probability distribution P(A_1, ..., A_L)
    for a given sequence A_1,...,A_L from J_ij and h_i parameters

    Parameters
    ----------
    sequences_X, sequences_Y : np.array
        Sequence matrix for which Hamiltonians will be computed
    m: CouplingsModel
    positions_aln_index: list of tuple of int
        list of pairs of positions for which inter sequence energy will be calculated
        will be used to index into alignment matrix
    positions_model_index: list of tuple of int
        list of pairs of positions for which inter sequence energy will be calculated
        will be used to index into model matrix

    Returns
    -------
    np.array
        Float matrix of size len(sequences_x) x len(sequences_y), where each position i,j corresponds
        to the sum of Jijs between the given positions in sequences_X[i] and sequences_Y[j]
    """
    # iterate over sequences
    N_x, L_x = sequences_X.shape
    N_y, L_y = sequences_Y.shape


    # ALIGNMENT INDICES
    i_aln, j_aln = zip(*positions_aln_index)
    i_aln = np.array(i_aln)
    j_aln = np.array(j_aln)

    # model index
    i,j = zip(*positions_model_index)
    i = np.array(i)
    j = np.array(j)

    H = np.zeros((N_x, N_y))
    for s_x, A_x in enumerate(sequences_X):
        for s_y, A_y in enumerate(sequences_Y):

            ax = A_x[np.array(i_aln)]
            ay = A_y[np.array(j_aln)]

            H[s_x, s_y] = m.J_ij[i, j, ax, ay].sum()

    return np.round(H, 4)


# # get the energy difference
# def _energy_diff(df):
#     """
#     For each potential pairing of id_1, calculate the difference
#     in statistical energy between that pairing and then next highest
#     scoring pairing, and store as E_diff_1
#
#     For each potential pairing of id_1, calculate the difference
#     in statistical energy between that pairing and then next highest
#     scoring pairing, and store as E_diff_2
#
#     Params
#     ------
#     df: pd.DataFrame
#         contains columns [id_1, id_2, E]
#
#     Returns
#     -------
#     df: pd.DataFrame
#         contains columns [id_1, id_2, E, E_diff_1, E_diff_2]
#     """
#
#     final_df = pd.DataFrame(columns=list(df.columns) + ["E_diff_1", "E_diff_2"])
#     grouped_df = df.groupby("id_1")
#     for id_1, _df in grouped_df:
#
#         # sort the dataframe by statistical energy
#         _sorted_df = _df.sort_values("E", ascending=False)
#         _sorted_df.loc[:, "E_diff"] = 0.
#         _sorted_df = _sorted_df.reset_index(drop=True)
#
#         # subtract the statE of each row from the previous row
#         for index in _sorted_df.index[0:-1]:
#             print(_sorted_df.loc[index, "E"],_sorted_df.loc[index+1, "E"])
#             #print()
#             _sorted_df.loc[index, "E_diff_1"] = _sorted_df.loc[index, "E"] - _sorted_df.loc[index + 1, "E"]
#
#         final_df = pd.concat([final_df, _sorted_df])
#
#     grouped_df = final_df.groupby("id_2")
#     for id_2, _df in grouped_df:
#
#         # sort the dataframe by statistical energy
#         _sorted_df = _df.sort_values("E", ascending=False)
#         _sorted_df.loc[:, "E_diff"] = 0.
#         _sorted_df = _sorted_df.reset_index(drop=True)
#
#         # subtract the statE of each row from the previous row
#         for index in _sorted_df.index[0:-1]:
#             print(_sorted_df.loc[index, "E"],_sorted_df.loc[index+1, "E"])
#             #print()
#             _sorted_df.loc[index, "E_diff_2"] = _sorted_df.loc[index, "E"] - _sorted_df.loc[index + 1, "E"]
#
#     return final_df.sort_values("E_diff_1", ascending=False)
#
#
# # get the energy difference
# def df_to_assigned_pairs(df):
#     """
#     implements the Bitbol 2016 Figure S12 pairing algorithm
#
#     Params
#     ------
#     df: pd.DataFrame
#         contains columns [id_1, id_2, E]
#
#     Returns
#     -------
#     df: pd.DataFrame
#         contains columns [id_1, id_2, E, E_diff, E_delta]
#         the best pairings as determined by the alorgithm
#     """
#     final_df = pd.DataFrame(columns=list(df.columns) + ["E_delta"])
#     grouped_df = df.groupby("species")
#
#     for species, _df in grouped_df:
#
#         # END CONDITION: all of id_1 are paired
#         _to_be_paired = _df.copy()
#         n_paired = 0
#         latest_E_delta = np.nan
#
#         while len(_to_be_paired) > 0:
#
#             # get the highest E pair
#             sort = _to_be_paired.sort_values("E", ascending=False)
#             highest_pair = sort.iloc[0, :]
#             first_id = int(highest_pair.id_1)
#             second_id = int(highest_pair.id_2)
#
#             # remove all related
#             _to_be_paired = _to_be_paired.query("id_1 != @first_id and id_2 != @second_id")
#
#             # score based on Bitbol confidence score
#             if len(_to_be_paired) > 0:
#                 highest_pair["E_delta"] = min(highest_pair["E_diff_1"], highest_pair["E_diff_2"]) / (n_paired + 1)
#                 latest_E_delta = highest_pair["E_delta"]
#             else:
#                 highest_pair["E_delta"] = latest_E_delta
#
#             n_paired += 1
#
#             final_df = pd.concat([final_df, highest_pair.to_frame().transpose()])
#
#     return final_df.sort_values("E_delta", ascending=False)
