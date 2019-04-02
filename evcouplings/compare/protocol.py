"""
EC to 3D structure comparison protocols/workflows.

Authors:
  Thomas A. Hopf
  Anna G. Green (complex and _make_complex_contact_maps)
"""

from copy import deepcopy
from math import ceil
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from evcouplings.align.alignment import (
    read_fasta, parse_header
)
from evcouplings.utils.config import (
    check_required, InvalidParameterError
)

from evcouplings.utils.system import (
    create_prefix_folders, insert_dir, verify_resources, valid_file
)
from evcouplings.couplings import Segment, add_mixture_probability
from evcouplings.compare.pdb import load_structures
from evcouplings.compare.distances import (
    intra_dists, multimer_dists, remap_chains,
    inter_dists, remap_complex_chains
)
from evcouplings.compare.sifts import SIFTS, SIFTSResult
from evcouplings.compare.ecs import (
    coupling_scores_compared, add_precision
)
from evcouplings.visualize import pairs, misc
from evcouplings.compare.enrichment import create_enrichment_table, add_enrichment
from evcouplings.compare.asa import combine_asa, add_asa

from evcouplings.align import ALPHABET_PROTEIN

HYDROPHOBIC_WEIGHTS = {
    "G": -.4,
    "A": 1.8,
    "P": -1.6,
    "V": 4.2,
    "L": 3.8,
    "I": 4.5,
    "M": 1.9,
    "F": 2.8,
    "Y": -1.3,
    "W": -0.9, 
    "S": -0.8,
    "T": -0.7,
    "C": 2.5,
    "N": -3.5,
    "Q": -3.5,
    "K": -3.9,
    "H": -3.2,
    "R": -4.5,
    "D": -3.5,
    "E": -3.5,
    "-": 0
}


def double_window_enrichment(ecs, num_pairs=1.0, score="cn", min_seqdist=0, window_size=5):
    """
    Calculate a modified version of EC "enrichment" as first described in
    Hopf et al., Cell, 2012. Here, enrichment is calculated for each EC i,j
    including all other ECs in the range of (i-window_size, i+window_size), 
    (j-window_size, j+window size), effectively drawing a square around
    each point i,j and looking for other ECs within that point. 


    Parameters
    ----------
    ecs : pd.DataFrame
        Dataframe containing couplings
    num_pairs : int or float, optional (default: 1.0)
        Number of ECs to use for enrichment calculation.
        - If float, will be interpreted as fraction of the
        length of the sequence (e.g. 1.0*L)
        - If int, will be interpreted as
        absolute number of pairs
    score : str , optional (default: cn)
        Pair coupling score used for calculation
    min_seqdist : int, optional (default: 6)
        Minimum sequence distance of couplings that will
        be included in the calculation

    Returns
    -------
    enrichment_table : pd.DataFrame
        Sorted table with enrichment values for each
        position in the sequence
    """

    # check if the provided table has segments...
    if "segment_i" in ecs.columns and "segment_j" in ecs.columns:
        has_segments = True
    # ... and if not, create them
    else:
        has_segments = False
        ecs["segment_i"] = "A_1"
        ecs["segment_j"] = "A_1"

    # stack dataframe so it contains each
    # EC twice as forward and backward pairs
    # (i, j) and (j, i)
    flipped = ecs.rename(
        columns={
            "i": "j", "j": "i", "A_i": "A_j", "A_j": "A_i",
            "segment_i": "segment_j", "segment_j": "segment_i"
        }
    )

    stacked_ecs = ecs.append(flipped)

    # determine how many positions ECs are over using the combined dataframe
    num_pos = len(stacked_ecs.groupby(["i", "A_i", "segment_i"]))

    # calculate absolute number of pairs if
    # fraction of length is given
    if isinstance(num_pairs, float):
        num_pairs = int(math.ceil(num_pairs * num_pos))

    # sort the ECs
    ecs = ecs.query(
        "abs(i-j) >= {}".format(min_seqdist)
    ).sort_values(
        by=score, ascending=False
    )

    # take the top num 
    top_ecs = ecs[0:num_pairs]

    sliding_window_data = []

    # calculate sum of EC scores for each position
    def _window_enrich(i, A_i, segment_i):
        segment_ecs = top_ecs.query("segment_i == @segment_i")
        nearby = segment_ecs.query("abs(i - @i)<@window_size")
        summed = sum(nearby.loc[:, score])
        sliding_window_data.append([i, A_i, segment_i, summed])
        return sliding_window_data
        
    for idx, row in top_ecs.iterrows():
        # enrichment on position i
        i, A_i, segment_i = row.i, row.A_i, row.segment_i
        j, A_j, segment_j = row.j, row.A_j, row.segment_j
        
        segment_ecs = top_ecs.query("segment_i == @segment_i and segment_j==@segment_j")
        nearby = segment_ecs.query("abs(i - @i)<@window_size and abs(j-@j)<@window_size")
        summed = sum(nearby.loc[:, score]) 
        sliding_window_data.append([i, A_i, segment_i, j, A_j, segment_j, summed])


    ec_sums = pd.DataFrame(
        sliding_window_data,
        columns = ["i", "A_i", "segment_i", "j", "A_j", "segment_j", "enrichment"]
    )
    #print(ec_sums)

    # average EC strength for top ECs
    avg_degree = top_ecs.loc[:, score].sum() / len(top_ecs)

    # "enrichment" is ratio how much EC strength on
    # an individual position exceeds average strength in top
    ec_sums.loc[:, "enrichment"] = ec_sums.loc[:, "enrichment"] / avg_degree

    return ec_sums.sort_values(by="enrichment", ascending=False)


def complex_probability(ecs, scoring_model, use_all_ecs=False,
                        score="cn", N_effL=None):
    """
    Adds confidence measure for complex evolutionary couplings

    Parameters
    ----------
    ecs : pandas.DataFrame
        Table with evolutionary couplings
    scoring_model : {"skewnormal", "normal", "evcomplex"}
        Use this scoring model to assign EC confidence measure
    use_all_ecs : bool, optional (default: False)
        If true, fits the scoring model to all ECs;
        if false, fits the model to only the inter ECs.
    score : str, optional (default: "cn")
        Use this score column for confidence assignment
        
    Returns
    -------
    ecs : pandas.DataFrame
        EC table with additional column "probability"
        containing confidence measure
    """
    from evcouplings.couplings.pairs import add_mixture_probability

    if use_all_ecs:
        ecs = add_mixture_probability(
            ecs, model=scoring_model
        )
    else:
        inter_ecs = ecs.query("segment_i != segment_j")
        intra_ecs = ecs.query("segment_i == segment_j")

        intra_ecs = add_mixture_probability(
            intra_ecs, model=scoring_model, score=score, N_effL=N_effL
        )

        inter_ecs = add_mixture_probability(
            inter_ecs, model=scoring_model, score=score, N_effL=N_effL
        )

        ecs = pd.concat(
            [intra_ecs, inter_ecs]
        ).sort_values(
            score, ascending=False
        )

    return ecs


def _filter_structures(sifts_map, pdb_ids=None, max_num_hits=None, max_num_structures=None):

    """
    Filters input SIFTSResult for specific pdb ids and/or number of hits

    Parameters
    ----------
    sifts_map: SIFTSResult
        Identified structures and residue index mappings
    pdb_ids: list of str, optional (default: None)
        List of PDB ids to be used for comparison
    max_num_hits: int, optional (default: None)
        Number of PDB hits to be used for comparison.
        Different chains from the same PDB count as multiple hits.
    max_num_structures: int, optional (default: None)
        Number of unique PDB ids to be used for comparison.
        Different chains from the same PDB count as one hit.

    Returns
    -------
    SIFTSResult
        Filtered identified structures and residue index mappings
    """
    def _filter_by_id(x, id_list):
        x = deepcopy(x)
        x.hits = x.hits.loc[
            x.hits.pdb_id.isin(id_list)
        ]
        return x


    # filter ID list down to manually selected PDB entries
    if pdb_ids is not None:
        pdb_ids = pdb_ids

        # make sure we have a list of PDB IDs
        if not isinstance(pdb_ids, list):
            pdb_ids = [pdb_ids]

        pdb_ids = [x.lower() for x in pdb_ids]

        sifts_map = _filter_by_id(sifts_map, pdb_ids)

    # limit number of hits and structures
    if max_num_hits is not None:
        sifts_map.hits = sifts_map.hits.iloc[:max_num_hits]

    if max_num_structures is not None:
        keep_ids = sifts_map.hits.pdb_id.unique()
        keep_ids = keep_ids[:max_num_structures]
        sifts_map = _filter_by_id(sifts_map, keep_ids)

    return sifts_map

def _identify_structures(**kwargs):
    """
    Identify set of 3D structures for comparison

    Parameters
    ----------
    **kwargs
        See check_required in code below

    Returns
    -------
    SIFTSResult
        Identified structures and residue index mappings
    """

    check_required(
        kwargs,
        [
            "prefix", "pdb_ids", "compare_multimer",
            "max_num_hits", "max_num_structures",
            "pdb_mmtf_dir",
            "sifts_mapping_table", "sifts_sequence_db",
            "by_alignment", "pdb_alignment_method",
            "alignment_min_overlap",
            "sequence_id", "sequence_file", "region",
            "use_bitscores", "domain_threshold",
            "sequence_threshold"
        ]
    )
    # get SIFTS mapping object/sequence DB
    s = SIFTS(
        kwargs["sifts_mapping_table"],
        kwargs["sifts_sequence_db"]
    )

    reduce_chains = not kwargs["compare_multimer"]

    # determine if we need to find structures
    # by sequence search or just fetching
    # based on Uniprot/PDB identifier
    if kwargs["by_alignment"]:

        # if searching by alignment, verify that
        # user selected jackhmmer or hmmsearch
        SEARCH_METHODS = ["jackhmmer", "hmmsearch"]

        if kwargs["pdb_alignment_method"] not in SEARCH_METHODS:
            raise InvalidParameterError(
                "Invalid pdb search method: " +
                "{}. Valid selections are: {}".format(
                    ", ".join(SEARCH_METHODS.keys())
                )
            )

        sifts_map = s.by_alignment(
            reduce_chains=reduce_chains,
            min_overlap=kwargs["alignment_min_overlap"],
            **kwargs
        )
    else:
        sifts_map = s.by_uniprot_id(
            kwargs["sequence_id"], reduce_chains=reduce_chains
        )

    # Save the pre-filtered SIFTs map
    sifts_map_full = deepcopy(sifts_map)

    #Filter the SIFTS map
    sifts_map = _filter_structures(
       sifts_map, kwargs["pdb_ids"], kwargs["max_num_hits"], kwargs["max_num_structures"]
    )

    return sifts_map, sifts_map_full


def _make_contact_maps(ec_table, d_intra, d_multimer, **kwargs):
    """
    Plot contact maps with all ECs above a certain probability threshold,
    or a given count of ECs

    Parameters
    ----------
    ec_table : pandas.DataFrame
        Full set of evolutionary couplings (all pairs)
    d_intra : DistanceMap
        Computed residue-residue distances inside chain
    d_multimer : DistanceMap
        Computed residue-residue distances between homomultimeric
        chains
    **kwargs
        Further plotting parameters, see check_required in code
        for necessary values.

    Returns
    -------
    cm_files : list(str)
        Paths of generated contact map files
    """

    def plot_cm(ecs, output_file=None):
        """
        Simple wrapper for contact map plotting
        """
        with misc.plot_context("Arial"):
            fig = plt.figure(figsize=(8, 8))
            if kwargs["scale_sizes"]:
                ecs = ecs.copy()
                ecs.loc[:, "size"] = ecs.cn.values / ecs.cn.max()

            pairs.plot_contact_map(
                ecs, d_intra, d_multimer,
                distance_cutoff=kwargs["distance_cutoff"],
                show_secstruct=kwargs["draw_secondary_structure"],
                margin=5,
                boundaries=kwargs["boundaries"]
            )

            plt.suptitle("{} evolutionary couplings".format(len(ecs)), fontsize=14)

            if output_file is not None:
                plt.savefig(output_file, bbox_inches="tight")
                plt.close(fig)

    check_required(
        kwargs,
        [
            "prefix", "min_sequence_distance",
            "plot_probability_cutoffs",
            "boundaries", "plot_lowest_count",
            "plot_highest_count", "plot_increase",
            "draw_secondary_structure"
        ]
    )

    prefix = kwargs["prefix"]

    cm_files = []

    ecs_longrange = ec_table.query(
        "abs(i - j) >= {}".format(kwargs["min_sequence_distance"])
    )

    # based on significance cutoff
    if kwargs["plot_probability_cutoffs"]:
        cutoffs = kwargs["plot_probability_cutoffs"]
        if not isinstance(cutoffs, list):
            cutoffs = [cutoffs]

        for c in cutoffs:
            ec_set = ecs_longrange.query("probability >= @c")
            # only can plot if we have any significant ECs above threshold
            if len(ec_set) > 0:
                output_file = prefix + "_significant_ECs_{}.pdf".format(c)
                plot_cm(ec_set, output_file=output_file)
                cm_files.append(output_file)

    # based on number of long-range ECs

    # identify number of sites in EC model
    num_sites = len(
        set.union(set(ec_table.i.unique()), set(ec_table.j.unique()))
    )

    # transform fraction of number of sites into discrete number of ECs
    def _discrete_count(x):
        if isinstance(x, float):
            x = ceil(x * num_sites)
        return int(x)

    # range of plots to make
    lowest = _discrete_count(kwargs["plot_lowest_count"])
    highest = _discrete_count(kwargs["plot_highest_count"])
    step = _discrete_count(kwargs["plot_increase"])

    # create individual plots
    for c in range(lowest, highest + 1, step):
        ec_set = ecs_longrange.iloc[:c]
        output_file = prefix + "_{}_ECs.pdf".format(c)
        plot_cm(ec_set, output_file=output_file)
        cm_files.append(output_file)

    # give back list of all contact map file names
    return cm_files


def _make_complex_contact_maps(ec_table, d_intra_i, d_multimer_i,
                               d_intra_j, d_multimer_j,
                               d_inter, first_segment_name,
                               second_segment_name, **kwargs):
    """
    Plot contact maps with all ECs above a certain probability threshold,
    or a given count of ECs

    Parameters
    ----------
    ec_table : pandas.DataFrame
        Full set of evolutionary couplings (all pairs)
    d_intra_i, d_intra_j: DistanceMap
        Computed residue-residue distances within chains for
        monomers i and j
    d_multimer_i, d_multimer_j : DistanceMap
        Computed residue-residue distances between homomultimeric
        chains for monomers i and j
    d_inter: DistanceMap
        Computed residue-residue distances between heteromultimeric
        chains i and j
    first_segment_name, second_segment_name: str
        Name of segment i and segment j in the ec_table
    **kwargs
        Further plotting parameters, see check_required in code
        for necessary values.

    Returns
    -------
    cm_files : list(str)
        Paths of generated contact map files
    """

    def plot_complex_cm(ecs_i, ecs_j, ecs_inter, 
                        first_segment_name,
                        second_segment_name, output_file=None):
        """
        Simple wrapper for contact map plotting
        """
        with misc.plot_context("Arial"):
            if kwargs["scale_sizes"]:
                # to scale sizes, combine all ecs to rescale together
                ecs = pd.concat([ecs_i, ecs_j, ecs_inter])
                ecs.loc[:, "size"] = ecs.cn.values / ecs.cn.max()

                # split back into three separate DataFrames
                ecs_i = ecs.query("segment_i == segment_j == @first_segment_name")
                ecs_j = ecs.query("segment_i == segment_j == @second_segment_name")
                ecs_inter = ecs.query("segment_i != segment_j")

                # if any of these groups are entry, replace with None
                if len(ecs_i) == 0:
                    ecs_i = None
                if len(ecs_j) == 0:
                    ecs_j = None
                if len(ecs_inter) == 0:
                    ecs_inter = None

            # Currently, we require at least one of the monomer 
            # to have either ECs or distances in order to make a plot
            if ((ecs_i is None or ecs_i.empty) and d_intra_i is None and d_multimer_i is None) \
                    or ((ecs_j is None or ecs_j.empty) and d_intra_j is None and d_multimer_i is None):
                return False

            fig = plt.figure(figsize=(8, 8))

            # create the contact map
            pairs.complex_contact_map(
                ecs_i, ecs_j, ecs_inter,
                d_intra_i, d_multimer_i,
                d_intra_j, d_multimer_j,
                d_inter,
                margin=5,
                distance_cutoff=kwargs["distance_cutoff"],
                show_secstruct=kwargs["draw_secondary_structure"],
                boundaries=kwargs["boundaries"]
            )

            # Add title to the plot
            if ecs_inter is None:
                ec_len = '0'
            else:
                ec_len = len(ecs_inter)
            plt.suptitle(
                "{} inter-molecule evolutionary couplings".format(ec_len), 
                fontsize=14
            )

            # save to output
            if output_file is not None:
                plt.savefig(output_file, bbox_inches="tight")
                plt.close(fig)

            return True

    check_required(
        kwargs,
        [
            "prefix", "min_sequence_distance",
            "plot_probability_cutoffs",
            "boundaries",
            "draw_secondary_structure", "plot_lowest_count",
            "plot_highest_count", "plot_increase",
            "scale_sizes"
        ]
    )

    prefix = kwargs["prefix"]

    cm_files = []

    ecs_longrange = ec_table.query(
        "abs(i - j) >= {} or segment_i != segment_j".format(kwargs["min_sequence_distance"])
    )

    # create plots based on significance cutoff
    if kwargs["plot_probability_cutoffs"]:
        cutoffs = kwargs["plot_probability_cutoffs"]
        if not isinstance(cutoffs, list):
            cutoffs = [cutoffs]

        for c in cutoffs:
            ec_set = ecs_longrange.query("probability >= @c")

            # only can plot if we have any significant ECs above threshold
            if len(ec_set) > 0:
                ec_set_i = ec_set.query("segment_i == segment_j == @first_segment_name")
                ec_set_j = ec_set.query("segment_i == segment_j == @second_segment_name")
                ec_set_inter = ec_set.query("segment_i != segment_j")

                output_file = prefix + "_significant_ECs_{}.pdf".format(c)
                plot_completed = plot_complex_cm(
                    ec_set_i, ec_set_j, ec_set_inter,
                    first_segment_name, second_segment_name,
                    output_file=output_file
                )
                if plot_completed:
                    cm_files.append(output_file)

    # transform fraction of number of sites into discrete number of ECs
    def _discrete_count(x):
        if isinstance(x, float):
            num_sites = 0
            for seg_name in [first_segment_name, second_segment_name]:
                num_sites += len(
                    set.union(
                        set(ec_table.query("segment_i == @seg_name").i.unique()),
                        set(ec_table.query("segment_j == @seg_name").j.unique())
                    )
                )

            x = ceil(x * num_sites)

        return int(x)

    # range of plots to make
    lowest = _discrete_count(kwargs["plot_lowest_count"])
    highest = _discrete_count(kwargs["plot_highest_count"])
    step = _discrete_count(kwargs["plot_increase"])

    for c in range(lowest, highest + 1, step):
        # get the inter ECs to plot
        ec_set_inter = ecs_longrange.query("segment_i != segment_j")[0:c]

        # if there are no inter ecs to be plotted, continue
        if ec_set_inter.empty:
            continue

        # get the index of the lowest inter EC
        last_inter_index = ec_set_inter.index[-1]

        # take all intra-protein ECs that score higher than the lowest plotted inter-protein EC
        ec_set_i = ecs_longrange.iloc[0:last_inter_index].query(
            "segment_i == segment_j == @first_segment_name"
        )
        ec_set_j = ecs_longrange.iloc[0:last_inter_index].query(
            "segment_i == segment_j == @second_segment_name"
        )

        output_file = prefix + "_{}_ECs.pdf".format(c)
        plot_completed = plot_complex_cm(
            ec_set_i, ec_set_j, ec_set_inter,
            first_segment_name, second_segment_name,
            output_file=output_file
        )
        if plot_completed:
            cm_files.append(output_file)

    # give back list of all contact map file names
    return cm_files


def standard(**kwargs):
    """
    Protocol:
    Compare ECs for single proteins (or domains)
    to 3D structure information

    Parameters
    ----------
    Mandatory kwargs arguments:
        See list below in code where calling check_required

    Returns
    -------
    outcfg : dict
        Output configuration of the pipeline, including
        the following fields:

        * ec_file_compared_all
        * ec_file_compared_all_longrange
        * pdb_structure_hits
        * distmap_monomer
        * distmap_multimer
        * contact_map_files
        * remapped_pdb_files
    """
    check_required(
        kwargs,
        [
            "prefix", "ec_file", "min_sequence_distance",
            "pdb_mmtf_dir", "atom_filter", "compare_multimer",
            "distance_cutoff", "target_sequence_file",
            "scale_sizes",
        ]
    )

    prefix = kwargs["prefix"]

    outcfg = {
        "ec_compared_all_file": prefix + "_CouplingScoresCompared_all.csv",
        "ec_compared_longrange_file": prefix + "_CouplingScoresCompared_longrange.csv",
        "pdb_structure_hits_file": prefix + "_structure_hits.csv",
        "pdb_structure_hits_unfiltered_file": prefix + "_structure_hits_unfiltered.csv",
        # cannot have the distmap files end with "_file" because there are
        # two files (.npy and .csv), which would cause problems with automatic
        # checking if those files exist
        "distmap_monomer": prefix + "_distance_map_monomer",
        "distmap_multimer": prefix + "_distance_map_multimer",
    }

    # make sure EC file exists
    verify_resources(
        "EC file does not exist",
        kwargs["ec_file"]
    )

    # make sure output directory exists
    create_prefix_folders(prefix)

    # store auxiliary files here (too much for average user)
    aux_prefix = insert_dir(prefix, "aux", rootname_subdir=False)
    create_prefix_folders(aux_prefix)

    # Step 1: Identify 3D structures for comparison
    sifts_map, sifts_map_full = _identify_structures(**{
        **kwargs,
        "prefix": aux_prefix,
    })

    # save selected PDB hits
    sifts_map.hits.to_csv(
        outcfg["pdb_structure_hits_file"], index=False
    )

    # also save full list of hits
    sifts_map_full.hits.to_csv(
        outcfg["pdb_structure_hits_unfiltered_file"], index=False
    )

    # Step 2: Compute distance maps

    # load all structures at once
    structures = load_structures(
        sifts_map.hits.pdb_id,
        kwargs["pdb_mmtf_dir"],
        raise_missing=False
    )

    # compute distance maps and save
    # (but only if we found some structure)
    if len(sifts_map.hits) > 0:
        d_intra = intra_dists(
            sifts_map, structures, atom_filter=kwargs["atom_filter"],
            output_prefix=aux_prefix + "_distmap_intra"
        )
        d_intra.to_file(outcfg["distmap_monomer"])

        # save contacts to separate file
        outcfg["monomer_contacts_file"] = prefix + "_contacts_monomer.csv"
        d_intra.contacts(
            kwargs["distance_cutoff"]
        ).to_csv(
            outcfg["monomer_contacts_file"], index=False
        )

        # compute multimer distances, if requested;
        # note that d_multimer can be None if there
        # are no structures with multiple chains
        if kwargs["compare_multimer"]:
            d_multimer = multimer_dists(
                sifts_map, structures, atom_filter=kwargs["atom_filter"],
                output_prefix=aux_prefix + "_distmap_multimer"
            )
        else:
            d_multimer = None

        # if we have a multimer contact mapin the end, save it
        if d_multimer is not None:
            d_multimer.to_file(outcfg["distmap_multimer"])
            outcfg["multimer_contacts_file"] = prefix + "_contacts_multimer.csv"

            # save contacts to separate file
            d_multimer.contacts(
                kwargs["distance_cutoff"]
            ).to_csv(
                outcfg["multimer_contacts_file"], index=False
            )
        else:
            outcfg["distmap_multimer"] = None

        # at this point, also create remapped structures (e.g. for
        # later comparison of folding results)
        verify_resources(
            "Target sequence file does not exist",
            kwargs["target_sequence_file"]
        )

        # create target sequence map for remapping structure
        with open(kwargs["target_sequence_file"]) as f:
            header, seq = next(read_fasta(f))

        seq_id, seq_start, seq_end = parse_header(header)
        seqmap = dict(zip(range(seq_start, seq_end + 1), seq))

        # remap structures, swap mapping index and filename in
        # dictionary so we have a list of files in the dict keys
        outcfg["remapped_pdb_files"] = {
            filename: mapping_index for mapping_index, filename in
            remap_chains(sifts_map, aux_prefix, seqmap).items()
        }
    else:
        # if no structures, can not compute distance maps
        d_intra = None
        d_multimer = None
        outcfg["distmap_monomer"] = None
        outcfg["distmap_multimer"] = None
        outcfg["remapped_pdb_files"] = None

    # Step 3: Compare ECs to distance maps

    ec_table = pd.read_csv(kwargs["ec_file"])

    # identify number of sites in EC model
    num_sites = len(
        set.union(set(ec_table.i.unique()), set(ec_table.j.unique()))
    )

    for out_file, min_seq_dist in [
        ("ec_compared_longrange_file", kwargs["min_sequence_distance"]),
        ("ec_compared_all_file", 0),
    ]:
        # compare ECs only if we minimally have intra distance map
        if d_intra is not None:
            coupling_scores_compared(
                ec_table, d_intra, d_multimer,
                dist_cutoff=kwargs["distance_cutoff"],
                output_file=outcfg[out_file],
                min_sequence_dist=min_seq_dist
            )
        else:
            outcfg[out_file] = None

    # also create line-drawing script if we made the csv
    if outcfg["ec_compared_longrange_file"] is not None:
        ecs_longrange = pd.read_csv(outcfg["ec_compared_longrange_file"])

        outcfg["ec_lines_compared_pml_file"] = prefix + "_draw_ec_lines_compared.pml"
        pairs.ec_lines_pymol_script(
            ecs_longrange.iloc[:num_sites, :],
            outcfg["ec_lines_compared_pml_file"],
            distance_cutoff=kwargs["distance_cutoff"]
        )

    # Step 4: Make contact map plots
    # if no structures available, defaults to EC-only plot

    outcfg["contact_map_files"] = _make_contact_maps(
        ec_table, d_intra, d_multimer, **kwargs
    )

    return outcfg


def complex(**kwargs):
    """
    Protocol:
    Compare ECs for a complex to
    3D structure

    Parameters
    ----------
    Mandatory kwargs arguments:
        See list below in code where calling check_required

    Returns
    -------
    outcfg : dict
        Output configuration of the pipeline, including
        the following fields:

        * ec_file_compared_all
        * ec_file_compared_all_longrange
        * pdb_structure_hits
        * distmap_monomer
        * distmap_multimer
        * contact_map_files
        * remapped_pdb_files
    """
    check_required(
        kwargs,
        [
            "prefix", "ec_file", "min_sequence_distance",
            "pdb_mmtf_dir", "atom_filter",
            "first_compare_multimer", "second_compare_multimer",
            "distance_cutoff", "segments",
            "first_sequence_id", "second_sequence_id",
            "first_sequence_file", "second_sequence_file",
            "first_target_sequence_file", "second_target_sequence_file",
            "scale_sizes"
        ]
    )

    prefix = kwargs["prefix"]

    outcfg = {
        # initialize output EC files
        "ec_compared_all_file": prefix + "_CouplingScoresCompared_all.csv",
        "ec_compared_longrange_file": prefix + "_CouplingScoresCompared_longrange.csv",
        "ec_compared_inter_file": prefix + "_CouplingsScoresCompared_inter.csv",

        # initialize output inter distancemap files
        "distmap_inter": prefix + "_distmap_inter",
        "inter_contacts_file": prefix + "_inter_contacts.csv"
    }

    # Add PDB comparison files for first and second monomer
    for monomer_prefix in ["first", "second"]:
        outcfg = {
            **outcfg,
            monomer_prefix + "_pdb_structure_hits_file":
                "{}_{}_structure_hits.csv".format(prefix, monomer_prefix),
            monomer_prefix + "_pdb_structure_hits_unfiltered_file":
                "{}_{}_structure_hits_unfitered.csv".format(prefix, monomer_prefix),
            monomer_prefix + "_distmap_monomer":
                "{}_{}_distance_map_monomer".format(prefix, monomer_prefix),
            monomer_prefix + "_distmap_multimer":
                "{}_{}_distance_map_multimer".format(prefix, monomer_prefix),
        }

    # make sure EC file exists
    verify_resources(
        "EC file does not exist",
        kwargs["ec_file"]
    )

    # make sure output directory exists
    create_prefix_folders(prefix)

    # store auxiliary files here (too much for average user)
    aux_prefix = insert_dir(prefix, "aux", rootname_subdir=False)
    create_prefix_folders(aux_prefix)

    # store auxiliary files here (too much for average user)
    first_aux_prefix = insert_dir(aux_prefix, "first_monomer", rootname_subdir=False)
    create_prefix_folders(first_aux_prefix)

    # store auxiliary files here (too much for average user)
    second_aux_prefix = insert_dir(aux_prefix, "second_monomer", rootname_subdir=False)
    create_prefix_folders(second_aux_prefix)

    # Step 1: Identify 3D structures for comparison
    def _identify_monomer_structures(name_prefix, outcfg, aux_prefix):
        # create a dictionary with kwargs for just the current monomer
        # any prefix that starts with a name_prefix will overwrite prefixes that do not start
        # eg, "first_sequence_file" will overwrite "sequence_file"
        monomer_kwargs = deepcopy(kwargs)
        for k,v in kwargs.items():
            if name_prefix + "_" in k:
                # only replace first occurrence of name_prefix
                monomer_kwargs[k.replace(name_prefix + "_", "", 1)] = v

        # remove the "prefix" kwargs so that we can replace with the
        # aux prefix when calling _identify_structures
        monomer_kwargs = {
            k: v for k, v in monomer_kwargs.items() if "prefix" not in k
        }

        # identify structures for that monomer
        sifts_map, sifts_map_full = _identify_structures(
            **monomer_kwargs,
            prefix=aux_prefix
        )

        # save full list of hits
        sifts_map_full.hits.to_csv(
            outcfg[name_prefix + "_pdb_structure_hits_unfiltered_file"], index=False
        )

        return outcfg, sifts_map, sifts_map_full

    outcfg, first_sifts_map, first_sifts_map_full = _identify_monomer_structures("first", outcfg, first_aux_prefix)
    outcfg, second_sifts_map, second_sifts_map_full = _identify_monomer_structures("second", outcfg, second_aux_prefix)

    # Determine the inter-protein PDB hits based on the full sifts map for each monomer
    inter_protein_hits_full = first_sifts_map_full.hits.merge(
        second_sifts_map_full.hits, on="pdb_id", how="inner", suffixes=["_1", "_2"]
    )
    outcfg["structure_hits_unfiltered_file"] = prefix + "_inter_structure_hits_unfiltered.csv"
    inter_protein_hits_full.to_csv(outcfg["structure_hits_unfiltered_file"])

    # Filter for the number of PDB ids to use
    inter_protein_sifts = SIFTSResult(hits=inter_protein_hits_full, mapping=None)
    inter_protein_sifts = _filter_structures(
        inter_protein_sifts,
        kwargs["inter_pdb_ids"],
        kwargs["inter_max_num_hits"],
        kwargs["inter_max_num_structures"]
    )

    outcfg["structure_hits_file"] = prefix + "_inter_structure_hits.csv"
    inter_protein_sifts.hits.to_csv(outcfg["structure_hits_file"])
    
    def _add_inter_pdbs(inter_protein_sifts, sifts_map, sifts_map_full, name_prefix,):
        """
        ensures that all pdbs used for inter comparison end up in the monomer SIFTS hits
        """

        lines_to_keep = sifts_map_full.hits.query("pdb_id in @inter_protein_sifts.hits.pdb_id").index
        sifts_map.hits = pd.concat([
            sifts_map.hits, sifts_map_full.hits.loc[lines_to_keep, :]
        ]).drop_duplicates()

        # save selected PDB hits
        sifts_map.hits.to_csv(
            outcfg[name_prefix + "_pdb_structure_hits_file"], index=False
        )
        return sifts_map

    first_sifts_map = _add_inter_pdbs(inter_protein_sifts, first_sifts_map, first_sifts_map_full, "first")
    second_sifts_map = _add_inter_pdbs(inter_protein_sifts, second_sifts_map, second_sifts_map_full, "second")

    # get the segment names from the kwargs
    segment_list = kwargs["segments"]

    # Make sure user provided exactly two segments
    if len(segment_list) != 2:
        raise InvalidParameterError(
            "Compare stage for protein complexes requires exactly two segments"
        )

    first_segment_name = Segment.from_list(kwargs["segments"][0]).segment_id
    second_segment_name = Segment.from_list(kwargs["segments"][1]).segment_id

    first_chain_name = Segment.from_list(kwargs["segments"][0]).default_chain_name()
    second_chain_name = Segment.from_list(kwargs["segments"][1]).default_chain_name()

    # load all structures at once
    all_ids = set(first_sifts_map.hits.pdb_id).union(
        set(second_sifts_map.hits.pdb_id)
    )

    structures = load_structures(
        all_ids,
        kwargs["pdb_mmtf_dir"],
        raise_missing=False
    )

    # Step 2: Compute distance maps
    def _compute_monomer_distance_maps(sifts_map, name_prefix, chain_name):

        # prepare a sequence map to remap the structures we have found
        verify_resources(
            "Target sequence file does not exist",
            kwargs[name_prefix + "_target_sequence_file"]
        )

        # create target sequence map for remapping structure
        with open(kwargs[name_prefix + "_target_sequence_file"]) as f:
            header, seq = next(read_fasta(f))

        # create target sequence map for remapping structure
        seq_id, seq_start, seq_end = parse_header(header)
        seqmap = dict(zip(range(seq_start, seq_end + 1), seq))

        # compute distance maps and save
        # (but only if we found some structure)
        if len(sifts_map.hits) > 0:
            d_intra = intra_dists(
                sifts_map, structures, atom_filter=kwargs["atom_filter"],
                output_prefix=aux_prefix + "_" + name_prefix + "_distmap_intra"
            )
            d_intra.to_file(outcfg[name_prefix + "_distmap_monomer"])

            # save contacts to separate file
            outcfg[name_prefix + "_monomer_contacts_file"] = prefix + "_" + name_prefix + "_contacts_monomer.csv"
            d_intra.contacts(
                kwargs["distance_cutoff"]
            ).to_csv(
                outcfg[name_prefix + "_monomer_contacts_file"], index=False
            )

            # compute multimer distances, if requested;
            # note that d_multimer can be None if there
            # are no structures with multiple chains
            if kwargs[name_prefix + "_compare_multimer"]:
                d_multimer = multimer_dists(
                    sifts_map, structures, atom_filter=kwargs["atom_filter"],
                    output_prefix=aux_prefix + "_" + name_prefix + "_distmap_multimer"
                )
            else:
                d_multimer = None

            # if we have a multimer contact map, save it
            if d_multimer is not None:
                d_multimer.to_file(outcfg[name_prefix + "_distmap_multimer"])
                outcfg[name_prefix + "_multimer_contacts_file"] = prefix + "_" + name_prefix + "_contacts_multimer.csv"

                # save contacts to separate file
                d_multimer.contacts(
                    kwargs["distance_cutoff"]
                ).to_csv(
                    outcfg[name_prefix + "_multimer_contacts_file"], index=False
                )
            else:
                outcfg[name_prefix + "_distmap_multimer"] = None

            # create remapped structures (e.g. for
            # later comparison of folding results)
            # remap structures, swap mapping index and filename in
            # dictionary so we have a list of files in the dict keys
            outcfg[name_prefix + "_remapped_pdb_files"] = {
                filename: mapping_index for mapping_index, filename in
                remap_chains(
                    sifts_map, aux_prefix, None, chain_name=chain_name,
                    raise_missing=kwargs["raise_missing"], atom_filter=None
                ).items()
            }

        else:
            # if no structures, cannot compute distance maps
            d_intra = None
            d_multimer = None
            outcfg[name_prefix + "_distmap_monomer"] = None
            outcfg[name_prefix + "_distmap_multimer"] = None
            outcfg[name_prefix + "remapped_pdb_files"] = None

        return d_intra, d_multimer, seqmap

    d_intra_i, d_multimer_i, seqmap_i = _compute_monomer_distance_maps(
        first_sifts_map, "first", first_chain_name
    )
    d_intra_j, d_multimer_j, seqmap_j = _compute_monomer_distance_maps(
        second_sifts_map, "second", second_chain_name
    )

    # compute inter distance map if sifts map for each monomer exists
    if len(first_sifts_map.hits) > 0 and len(second_sifts_map.hits) > 0:
        d_inter = inter_dists(
            first_sifts_map, second_sifts_map,
            raise_missing=kwargs["raise_missing"]
        )
        # if there were overlapping PDBs, save the results
        if d_inter is not None:
            d_inter.to_file(outcfg["distmap_inter"])

            # save contacts to separate file
            d_inter.contacts(
                kwargs["distance_cutoff"]
            ).to_csv(
                outcfg["inter_contacts_file"], index=False
            )

    else:
        outcfg["inter_contacts_file"] = None
        d_inter = None

    # # Step 3: Compare ECs to distance maps
    ec_table = pd.read_csv(kwargs["ec_file"])

    for out_file, min_seq_dist in [
        ("ec_compared_longrange_file", kwargs["min_sequence_distance"]),
        ("ec_compared_all_file", 0),
    ]:

        # compare ECs only if we have an intra distance map
        # for at least one monomer - inter can't exist unless
        # we have both monomers
        if (d_intra_i is not None) or (d_intra_j is not None):
            # compare distances individually for each segment pair
            ecs_intra_i = ec_table.query("segment_i == segment_j == @first_segment_name")
            if d_intra_i is not None:
                ecs_intra_i_compared = coupling_scores_compared(
                    ecs_intra_i, d_intra_i, d_multimer_i,
                    dist_cutoff=kwargs["distance_cutoff"],
                    output_file=None,
                    min_sequence_dist=min_seq_dist
                )
            else:
                # If no distance map, the distance is saved as np.nan
                ecs_intra_i_compared = ecs_intra_i.assign(dist=np.nan)

            ecs_intra_j = ec_table.query("segment_i == segment_j == @second_segment_name")
            if d_intra_j is not None:
                ecs_intra_j_compared = coupling_scores_compared(
                    ecs_intra_j, d_intra_j, d_multimer_j,
                    dist_cutoff=kwargs["distance_cutoff"],
                    output_file=None,
                    min_sequence_dist=min_seq_dist
                )
            else:
                ecs_intra_j_compared = ecs_intra_j.assign(dist=np.nan)

            ecs_inter = ec_table.query("segment_i != segment_j")
            if d_inter is not None:
                ecs_inter_compared = coupling_scores_compared(
                    ecs_inter, d_inter, dist_map_multimer=None,
                    dist_cutoff=kwargs["distance_cutoff"],
                    output_file=None,
                    min_sequence_dist=None  # does not apply for inter-protein ECs
                )
            else:
                ecs_inter_compared = ecs_inter.assign(dist=np.nan)

            # combine the tables
            ec_table_compared = pd.concat([
                ecs_inter_compared,
                ecs_intra_i_compared,
                ecs_intra_j_compared
            ])

            # rename the precision column to "segmentwise_precision"
            # because we calculated precision for each segment independently
            ec_table_compared = ec_table_compared.rename(
                columns={"precision": "segmentwise_precision"}
            )
            # TODO: change "cn" to "score" eventually
            ec_table_compared = ec_table_compared.sort_values("cn", ascending=False)

            # add the total precision
            # TODO: implement different cutoffs for intra vs inter contacts
            ec_table_compared = add_precision(
                ec_table_compared,
                dist_cutoff=kwargs["distance_cutoff"]
            )

            # save to file
            # all ecs
            ec_table_compared.to_csv(outcfg[out_file])

            # save the inter ECs to a file
            ecs_inter_compared.to_csv(outcfg["ec_compared_inter_file"])

    # create an inter-ecs file with extra information for calibration purposes
    def _calibration_file(prefix, ec_file):

        if not valid_file(ec_file):
            return None

        ecs = pd.read_csv(ec_file)

        #add the evcomplex score
        ecs = complex_probability(
            ecs, "evcomplex_uncorrected", False
        )
        ecs.loc[:,"evcomplex_raw"] = ecs.loc[:,"probability"]

        enrichment_table = create_enrichment_table(
           ecs, d_intra_i, d_intra_j
        )
        enrichment_table.to_csv(prefix + "_enrichment_complete.csv")
        ecs = add_enrichment(enrichment_table, ecs)

        # get only the top 100 inter ECs

        ecs = ecs.query("segment_i != segment_j")
        mean_ec = ecs.cn.mean()
        std_ec = ecs.cn.std()
        L = len(ecs.i.unique()) + len(ecs.j.unique())
        ecs = ecs[0:100]
        
        # calculate the z-score
        ecs["Z_score"] = (ecs.cn - mean_ec) / std_ec

        # add rank
        ecs["inter_relative_rank_longrange"] = ecs.index / L

        # accessible surface area
        if not "first_remapped_pdb_files" in outcfg:
            outcfg["first_remapped_pdb_files"] = []

        if not "second_remapped_pdb_files" in outcfg:
            outcfg["second_remapped_pdb_files"] = []

        first_asa = combine_asa(outcfg["first_remapped_pdb_files"], kwargs["prefix"])
        first_asa["segment_i"] = "A_1"

        second_asa = combine_asa(outcfg["second_remapped_pdb_files"], kwargs["prefix"])
        second_asa["segment_i"] = "B_1"

        asa = pd.concat([first_asa, second_asa])
        asa.to_csv(prefix + "_surface_area.csv")

        ecs = add_asa(ecs, asa, asa_column="mean")

        ecs["asa_max"] = [
            max([x,y]) for x,y in zip(ecs.asa_i, ecs.asa_j)
        ]
        ecs["asa_min"] = [
            min([x,y]) for x,y in zip(ecs.asa_i, ecs.asa_j)
        ]

        #conservation
        frequency_file = prefix.replace("compare", "concatenate") + "_frequencies.csv"
        print(frequency_file)
        d = pd.read_csv(frequency_file)
        d["j"] = d["i"]
        d["segment_j"] = d["segment_i"]
        conservation = {(x,y):z for x,y,z in zip(d.segment_i, d.i, d.conservation)}

        ecs["conservation_i"] = [conservation[(x,y)] if (x,y) in conservation else np.nan for x,y in zip(ecs.segment_i, ecs.i)]
        ecs["conservation_j"] = [conservation[(x,y)] if (x,y) in conservation else np.nan for x,y in zip(ecs.segment_j, ecs.j)]

        ecs["conservation_max"] = [
            max([x,y]) for x,y in zip(ecs.conservation_i, ecs.conservation_j)
        ]
        ecs["conservation_min"] = [
            min([x,y]) for x,y in zip(ecs.conservation_i, ecs.conservation_j)
        ]
        
        # amino acid frequencies
        for char in list(ALPHABET_PROTEIN):
            ecs = ecs.merge(d[["i", "segment_i", char]], on=["i","segment_i"], how="left")
            ecs = ecs.rename({char: f"f{char}_i"}, axis=1)

            ecs = ecs.merge(d[["j", "segment_j", char]], on=["j", "segment_j"], how="left")
            ecs = ecs.rename({char: f"f{char}_j"}, axis=1)

        for i in list(ALPHABET_PROTEIN):
            ecs[f'f{i}'] = ecs[f'f{i}_i'] + ecs[f'f{i}_j']

        hydrophilicity = []
        for idx,row in ecs.iterrows():
            hydro = sum([
                HYDROPHOBIC_WEIGHTS[i] * float(row[[f'f{i}']]) for i in list(ALPHABET_PROTEIN)
            ])
            hydrophilicity.append(hydro)

        ecs["f_hydrophilicity"] = hydrophilicity

        #enrichment
        inter_ecs = ecs.query("segment_i != segment_j")

        enrich_range_to_calculate = [1, 5, 10]
        for size in enrich_range_to_calculate:
            enrich = double_window_enrichment(ecs=inter_ecs, min_seqdist=0, num_pairs=20, window_size=size, score="Z_score")
            enrich = enrich.rename({"enrichment": f"enrichment_{size}"}, axis=1)
            inter_ecs = inter_ecs.merge(enrich, on=["i", "A_i", "segment_i", "j", "A_j", "segment_j"])   

        #write the file (top 50 only)
        
        outcfg["calibration_file"] = prefix + "_CouplingScores_inter_calibration.csv"
        inter_ecs.iloc[0:50, :].to_csv(outcfg["calibration_file"])

    # if valid_file(outcfg["ec_compared_longrange_file"]):
    #     _calibration_file(prefix, outcfg["ec_compared_longrange_file"])
    # else:
    #     _calibration_file(prefix, kwargs["ec_longrange_file"])

    # create the inter-ecs line drawing script
    if outcfg["ec_compared_inter_file"] is not None and kwargs["plot_highest_count"] is not None:
        inter_ecs = ec_table.query("segment_i != segment_j")

        outcfg["ec_lines_compared_pml_file"] = prefix + "_draw_ec_lines_compared.pml"

        pairs.ec_lines_pymol_script(
            inter_ecs.iloc[:kwargs["plot_highest_count"], :],
            outcfg["ec_lines_compared_pml_file"],
            distance_cutoff=kwargs["distance_cutoff"],
            chain={
                first_segment_name: first_chain_name,
                second_segment_name: second_chain_name
            }
        )

    # Remap the complex crystal structures, if available
    if len(first_sifts_map.hits) > 0 and len(second_sifts_map.hits) > 0:
        outcfg["complex_remapped_pdb_files"] = {
            filename: mapping_index for mapping_index, filename in
            remap_complex_chains(
                first_sifts_map, second_sifts_map,
                seqmap_i, seqmap_j, output_prefix=aux_prefix,
                raise_missing=kwargs["raise_missing"], atom_filter=None
            ).items()
        }

    # Step 4: Make contact map plots
    # if no structures available, defaults to EC-only plot
    outcfg["contact_map_files"] = _make_complex_contact_maps(
        ec_table, d_intra_i, d_multimer_i,
        d_intra_j, d_multimer_j,
        d_inter, first_segment_name,
        second_segment_name, **kwargs
    )

    return outcfg


def double_window_enrichment(ecs, num_pairs=1.0, score="cn", min_seqdist=0, window_size=5):
    """
    Calculate a modified version of EC "enrichment" as first described in
    Hopf et al., Cell, 2012. Here, enrichment is calculated for each EC i,j
    including all other ECs in the range of (i-window_size, i+window_size), 
    (j-window_size, j+window size), effectively drawing a square around
    each point i,j and looking for other ECs within that point. 


    Parameters
    ----------
    ecs : pd.DataFrame
        Dataframe containing couplings
    num_pairs : int or float, optional (default: 1.0)
        Number of ECs to use for enrichment calculation.
        - If float, will be interpreted as fraction of the
        length of the sequence (e.g. 1.0*L)
        - If int, will be interpreted as
        absolute number of pairs
    score : str , optional (default: cn)
        Pair coupling score used for calculation
    min_seqdist : int, optional (default: 6)
        Minimum sequence distance of couplings that will
        be included in the calculation

    Returns
    -------
    enrichment_table : pd.DataFrame
        Sorted table with enrichment values for each
        position in the sequence
    """

    # check if the provided table has segments...
    if "segment_i" in ecs.columns and "segment_j" in ecs.columns:
        has_segments = True
    # ... and if not, create them
    else:
        has_segments = False
        ecs["segment_i"] = "A_1"
        ecs["segment_j"] = "A_1"

    # stack dataframe so it contains each
    # EC twice as forward and backward pairs
    # (i, j) and (j, i)
    flipped = ecs.rename(
        columns={
            "i": "j", "j": "i", "A_i": "A_j", "A_j": "A_i",
            "segment_i": "segment_j", "segment_j": "segment_i"
        }
    )

    stacked_ecs = ecs.append(flipped)

    # determine how many positions ECs are over using the combined dataframe
    num_pos = len(stacked_ecs.groupby(["i", "A_i", "segment_i"]))

    # calculate absolute number of pairs if
    # fraction of length is given
    if isinstance(num_pairs, float):
        num_pairs = int(math.ceil(num_pairs * num_pos))

    # sort the ECs
    ecs = ecs.query(
        "abs(i-j) >= {}".format(min_seqdist)
    ).sort_values(
        by=score, ascending=False
    )

    # take the top num 
    top_ecs = ecs[0:num_pairs]

    sliding_window_data = []

    # calculate sum of EC scores for each position
    def _window_enrich(i, A_i, segment_i):
        segment_ecs = top_ecs.query("segment_i == @segment_i")
        nearby = segment_ecs.query("abs(i - @i)<@window_size")
        summed = sum(nearby.loc[:, score])
        sliding_window_data.append([i, A_i, segment_i, summed])
        return sliding_window_data
        
    for idx, row in top_ecs.iterrows():
        # enrichment on position i
        i, A_i, segment_i = row.i, row.A_i, row.segment_i
        j, A_j, segment_j = row.j, row.A_j, row.segment_j
        
        segment_ecs = top_ecs.query("segment_i == @segment_i and segment_j==@segment_j")
        nearby = segment_ecs.query("abs(i - @i)<@window_size and abs(j-@j)<@window_size")
        summed = sum(nearby.loc[:, score]) 
        sliding_window_data.append([i, A_i, segment_i, j, A_j, segment_j, summed])


    ec_sums = pd.DataFrame(
        sliding_window_data,
        columns = ["i", "A_i", "segment_i", "j", "A_j", "segment_j", "enrichment"]
    )
    #print(ec_sums)

    # average EC strength for top ECs
    avg_degree = top_ecs.loc[:, score].sum() / len(top_ecs)

    # "enrichment" is ratio how much EC strength on
    # an individual position exceeds average strength in top
    ec_sums.loc[:, "enrichment"] = ec_sums.loc[:, "enrichment"] / avg_degree

    return ec_sums.sort_values(by="enrichment", ascending=False)


# list of available EC comparison protocols
PROTOCOLS = {
    # standard monomer comparison protocol
    "standard": standard,

    # comparison for protein complexes
    "complex": complex
}


def run(**kwargs):
    """
    Run inference protocol to calculate ECs from
    input sequence alignment.

    Parameters
    ----------
    Mandatory kwargs arguments:
        protocol: EC protocol to run
        prefix: Output prefix for all generated files

    Returns
    -------
    outcfg : dict
        Output configuration of stage
        (see individual protocol for fields)
    """
    check_required(kwargs, ["protocol"])

    if kwargs["protocol"] not in PROTOCOLS:
        raise InvalidParameterError(
            "Invalid protocol selection: " +
            "{}. Valid protocols are: {}".format(
                kwargs["protocol"], ", ".join(PROTOCOLS.keys())
            )
        )

    return PROTOCOLS[kwargs["protocol"]](**kwargs)
