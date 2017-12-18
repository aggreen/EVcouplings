"""
Protocol for matching putatively interacting sequences
in protein complexes to create a concatenated sequence
alignment

Authors:
    Anna Green
    Mark Chonofsky
"""

import numpy as np
import pandas as pd
from copy import deepcopy
from evcouplings.complex.alignment import (
    write_concatenated_alignment, modify_complex_segments
)
from evcouplings.align import Alignment
from evcouplings.align.protocol import modify_alignment
from evcouplings.couplings.protocol import mean_field
from evcouplings.couplings.model import CouplingsModel
from evcouplings.utils.system import (
    create_prefix_folders, insert_dir
)
from evcouplings.complex.protocol import modify_complex_segments

ID_THRESH = .8

def inter_energy():
    pass


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


    def _get_index_for_species(monomer_information,
                               alignment, species):
        """
        Gets the indices of each sequence identifier
        in the alignment matrix

        Parameters
        ----------
        monomer_information:
        alignment: EVcouplings.align.Alignment
        species: str

        Returns
        -------
        pd.Dataframe
            With columns aln_index, id

        """
        species_df = monomer_information.query("species == @species")
        if len(species_df)==0 or species_df.empty():
            raise ValueError("")
        alignment_index = [alignment.id_to_index[x] for x in species_df.id]
        species_df.loc[:,"aln_index"] = alignment_index

        return species_df

    def _E_matrix_to_df(E_matrix, first_species_index,
                        second_species_index):
        """
        Converts an N_seqs by M_seqs matrix into
        a pd.DataFrame

        Returns
        -------
        pd.DataFrame with columns id_1, id_2, E

        """
        pass

    outcfg = {}

    # read in the monomer alignments
    with open(kwargs["first_alignment_file"],) as inf:
        alignment_1 = Alignment.from_file(inf)
    with open(kwargs["second_alignment_file"]) as inf:
        alignment_2 = Alignment.from_file(inf)

    species_set = set(first_monomer_info.species).intersection(
        set(second_monomer_info.species)
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

    # create and write a starting alignment
    starting_aln_cfg, current_id_pairs = initialize_alignment(
        first_monomer_info, second_monomer_info,
        **current_kwargs
    )

    # update current kwargs with output of alignment generation
    outcfg["alignment_file_iter0"] = starting_aln_cfg["alignment_file"]
    current_kwargs["focus_sequence"] = starting_aln_cfg["focus_sequence"]
    current_kwargs["alignment_file"] = starting_aln_cfg["alignment_file"]
    current_kwargs = modify_complex_segments(current_kwargs, **current_kwargs)

    for iteration in range(N_pairing_iterations):

        # run the mean field calculation on the alignment
        mean_field_outcfg = mean_field(**current_kwargs)
        outcfg["model_file_{}".format(iteration)] = mean_field_outcfg["model_file"]
        outcfg["raw_ec_file_{}".format(iteration)] = mean_field_outcfg["raw_ec_file"]
        outcfg["ec_file_{}".format(iteration)] = mean_field_outcfg["ec_file"]

        energy_df = pd.DataFrame(columns=["species", "first_id", "second_id", "E"])

        # read in the parameters of mean field
        with open(mean_field_outcfg["model_file"]) as inf:
            model = CouplingsModel(inf)

        # calculate inter sequence energy
            for species in species_set:
                first_alignment_indices = _get_index_for_species(
                    alignment_1,first_monomer_info, species
                )
                second_alignment_indices = _get_index_for_species(
                    alignment_2, second_monomer_info, species
                )
                # calculate inter-protein energy for all A ot B
                # TODO: BE EXTRA SURE INDICES ARE NOT FUCKED
                matrix = inter_energy(
                    model,first_alignment_indices,second_alignment_indices,
                    alignment_1,alignment_2
                )

                # Make df of species, A, B, E
                _energy_df = _E_matrix_to_df(matrix)
                energy_df = pd.concatenate([energy_df,_energy_df])

        # take the top M E
        M = N_increase_per_iteration * (iteration + 1)
        top_E = energy_df.sort_values("E",ascending=False).iloc[0:M,:]
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

        # update prefixes
        current_prefix = aux_prefix + "_iter{}".format(iteration+1)

        # update the configuration file for input into MFDCA
        current_kwargs["alignment_file"]
        current_kwargs["prefix"] = current_prefix

#save the major outcfg
# retun the correct shit
