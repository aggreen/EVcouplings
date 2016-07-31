"""
Protein sequence alignment creation protocols/workflows.

Authors:
  Thomas A. Hopf
"""

import evcouplings.align.tools as at
from evcouplings.align.alignment import (
    read_fasta, write_fasta, Alignment
)
from evcouplings.utils.config import check_required, MissingParameterError
from evcouplings.utils.system import (
    create_prefix_folders, get, file_not_empty, ResourceError
)


def fetch_sequence(sequence_id, sequence_file,
                   sequence_download_url, out_file):
    """
    Get sequence.

    Parameters
    ----------
    sequence_id : str
        Identifier of sequence that should be retrieved
    sequence_file : str
        File containing sequence. If None, sqeuence will
        be downloaded from sequence_download_url
    sequence_download_url : str
        URL from which to download missing sequence. Must
        contain "{}" at the position where sequence ID will
        be inserted into download URL (using str.format).
    out_file : str
        Output file in which sequence will be stored, if
        sequence_file is not existing.

    Returns
    -------
    str
        Path of file with stored sequence (can be sequence_file
        or out_file)
    tuple (str, str)
        Identifier of sequence as stored in file, and sequence
    """
    if sequence_file is None:
        get(
            sequence_download_url.format(sequence_id),
            out_file,
            allow_redirects=True
        )
    else:
        # if we have sequence file, pass it through
        out_file = sequence_file

    # also make sure input file has something in it
    if not file_not_empty(out_file):
        raise ResourceError(
            "Input sequence missing: {}".format(out_file)
        )

    with open(out_file) as f:
        seq = next(read_fasta(f))

    return out_file, seq


def cut_sequence(sequence, sequence_id, region=None, first_index=None, out_file=None):
    """
    Cut a given sequence to sub-range and save it in a file

    Parameters
    ----------
    sequence : str
        Full sequence that will be cut
    sequence_id : str
        Identifier of sequence, used to construct header
        in output file
    region : tuple(int, int), optional (default: None)
        Region that will be cut out of full sequence.
        If None, full sequence will be returned.
    first_index : int, optional (default: None)
        Define index of first position in sequence.
        Will be set to 1 if None.
    out_file : str, optional (default: None)
        Save sequence in a FASTA file (header:
        >sequence_id/start_region-end_region)

    Returns
    ------
    str
        Subsequence contained in region
    tuple(int, int)
        Region. If no input region is given, this will be
        (1, len(sequence)); otherwise, the input region is
        returned.

    Raises
    ------
    ValueError
        Upon invalid region specification (violating boundaries
        of sequence)
    """
    cut_seq = None

    # (not using 1 as default value to allow parameter
    # to be unspecified in config file)
    if first_index is None:
        first_index = 1

    # last index is *inclusive*!
    if region is None:
        region = (first_index, first_index + len(sequence) - 1)
        cut_seq = sequence
    else:
        start, end = region
        str_start = start - first_index
        str_end = end - first_index + 1
        cut_seq = sequence[str_start:str_end]

        # make sure bounds are valid given the sequence that we have
        if str_start < 0 or str_end > len(sequence):
            raise ValueError(
                "Illegal sequence range: "
                "region={} first_index={} len(sequence)={}".format(
                    region,
                    first_index,
                    len(sequence)
                )
            )

    # save sequence to file
    if out_file is not None:
        with open(out_file, "w") as f:
            header = "{}/{}-{}".format(sequence_id, *region)
            write_fasta([(header, cut_seq)], f)

    return region, cut_seq


def create_segment(sequence_id, region_start, region_end,
                   segment_id="1", segment_type="aa"):
    """
    Create a segment for a monomer sequence search run

    Parameters
    ----------
    sequence_id : str
        Identifier of sequence
    region_start : int
        Start index of sequence segment
    region_end : int
        End index of sequence segment
    segment_id : str
        Identifier for segment (must be unique)
    segment_type : {"aa", "dna", "rna"}
        Type of sequence

    Returns
    -------
    tuple
        Segment description
    """
    return (
        segment_type,
        segment_id,
        sequence_id,
        region_start,
        region_end
    )


def search_thresholds(use_bitscores, seq_threshold, domain_threshold, seq_len):
    """
    Set homology search inclusion parameters.

    HMMER hits get included in the HMM according to a two-step rule
    1) sequence passes sequence-level treshold
    2) domain passes domain-level threshold

    Therefore, search thresholds are set based on the following logic:
    1) If only sequence threshold is given, a MissingParameterException is raised
    2) If only bitscore threshold is given, sequence threshold is set to the same
    3) If both thresholds are given, they are according to defined values

    Valid inputs for bitscore thresholds:
    1) int or str: taken as absolute score threshold
    2) float: taken as relative threshold (absolute threshold derived by
       multiplication with domain length)

    Valid inputs for integer thresholds:
    1) int: Used as negative exponent, threshold will be set to 1E-<exponent>
    2) float or str: Interpreted literally

    Parameters
    ----------
    use_bitscores : bool
        Use bitscore threshold instead of E-value threshold
    domain_threshold : str or int or float
        Domain-level threshold. See rules above.
    seq_threshold : str or int or float
        Sequence-level threshold. See rules above.
    seq_len : int
        Length of sequence. Used to calculate absolute bitscore
        threshold for relative bitscore thresholds.

    Returns
    -------
    tuple (str, str)
        Sequence- and domain-level thresholds ready to be fed into HMMER
    """
    def transform_bitscore(x):
        if isinstance(x, float):
            # float: interpret as relative fraction of length
            return "{:.1f}".format(x * seq_len)
        else:
            # otherwise interpret as absolute score
            return str(x)

    def transform_evalue(x):
        if isinstance(x, int):
            # if integer, interpret as negative exponent
            return "1E{}".format(-x)
        else:
            # otherwise interpret literally
            # (mantissa-exponent string or float)
            return str(x).upper()

    if domain_threshold is None:
        raise MissingParameterError(
            "domain_threshold must be explicitly defined "
            "and may not be None/empty"
        )

    if use_bitscores:
        transform = transform_bitscore
    else:
        transform = transform_evalue

    if seq_threshold is not None:
        seq_threshold = transform(seq_threshold)

    if domain_threshold is not None:
        domain_threshold = transform(domain_threshold)

    # set "outer" sequence threshold so that it matches domain threshold
    if domain_threshold is not None and seq_threshold is None:
        seq_threshold = domain_threshold

    return seq_threshold, domain_threshold


def modify_alignment(config):
    """
    Prepare alignment to be ready for EC calculation
    """
    return


def describe(config):
    """
    Get parameters of alignment such as gaps, coverage,
    conservation
    """
    return


def external(**kwargs):
    """
    Protocol:
    Use external sequence alignment and extract all relevant
    information from there (e.g. sequence, region, etc.),
    then apply gap & fragment filtering as usual
    """
    print("Start from existing alignment")
    return {}


def standard(**kwargs):
    """
    Protocol:
    Standard buildali4 workflow

    Parameters
    ----------

    Returns
    -------

    """
    check_required(
        kwargs,
        [
            "prefix", "sequence_id", "sequence_file",
            "sequence_download_url", "reuse_alignment",
            "region", "first_index",
            "use_bitscores", "domain_threshold", "sequence_threshold",
            "database", "iterations", "cpu", "nobias", "reuse_alignment",
            "checkpoints_hmm", "checkpoints_ali", "jackhmmer"
        ]
    )

    prefix = kwargs["prefix"]

    # prepare output dictionary with result files
    outcfg = {
        "alignment_file": prefix + ".a2m",
        "statistics_file": prefix + "_alignment_statistics.csv",
        "sequence_file": prefix + ".fa",
        "specieslist_file": prefix + "_specieslist.csv",
        "focus_mode": True
    }

    # check if stage should be skipped and if so, return
    if kwargs.get("skip", False):
        # get information about sequence range from existing file
        with open(outcfg["sequence_file"]) as f:
            seq_id, seq = next(read_fasta(f))
            start, end = seq_id.split("/", maxsplit=1)[1].split("-")

        outcfg["segments"] = [
            create_segment(kwargs["sequence_id"], start, end)
        ]

        return outcfg

    # Otherwise, now run the protocol...
    # make sure output directory exists
    create_prefix_folders(prefix)

    # make sure search sequence is defined and load it
    full_seq_file, (full_seq_id, full_seq) = fetch_sequence(
        kwargs["sequence_id"],
        kwargs["sequence_file"],
        kwargs["sequence_download_url"],
        kwargs["prefix"] + "_full.fa"
    )

    # cut sequence to target region and save in sequence_file
    # (this is the main sequence file used downstream)
    region, cut_seq = cut_sequence(
        full_seq,
        kwargs["sequence_id"],
        kwargs["region"],
        kwargs["first_index"],
        outcfg["sequence_file"]
    )

    outcfg["segments"] = [
        create_segment(kwargs["sequence_id"], *region)
    ]

    # run jackhmmer... allow to reuse pre-exisiting
    # Stockholm alignment file here
    if not kwargs["reuse_alignment"]:
        # run iterative jackhmmer search
        check_required(kwargs, [kwargs["database"]])

        seq_threshold, domain_threshold = search_thresholds(
            kwargs["use_bitscores"],
            kwargs["sequence_threshold"],
            kwargs["domain_threshold"],
            len(cut_seq)
        )

        ali = at.run_jackhmmer(
            query=outcfg["sequence_file"],
            database=kwargs[kwargs["database"]],
            prefix=prefix,
            use_bitscores=kwargs["use_bitscores"],
            domain_threshold=domain_threshold,
            seq_threshold=seq_threshold,
            iterations=kwargs["iterations"],
            nobias=kwargs["nobias"],
            cpu=kwargs["cpu"],
            checkpoints_hmm=kwargs["checkpoints_hmm"],
            checkpoints_ali=kwargs["checkpoints_ali"],
            binary=kwargs["jackhmmer"],
        )
        ali_raw_file = ali.alignment
    else:
        ali_raw_file = prefix + ".sto"
        if not file_not_empty(ali_raw_file):
            raise ResourceError(
                "Tried to reuse alignment, but file does not exist "
                "or have any contents: {}".format(ali_raw_file)
            )

    # read in stockholm format (with full annotation)
    with open(ali_raw_file) as a:
        ali_raw = Alignment.from_file(a, "stockholm")

    # TODO: save species information here from annotation
    # generate specieslist (copy into sequence headers?)

    ali_raw_fasta_file = prefix + "_raw.fasta"
    with open(ali_raw_fasta_file, "w") as ao:
        ali_raw.write(ao, "fasta")

    # focus sequence
    # apply id filter, gap threshold
    # set correct headers (make ready for plmc)

    # print(kwargs)
    return outcfg

    # output gap statistics, conservation of columns

    # visualize distributions?

    # TODO: how to get alignment statistics and plots?
    # (modularize this into an independent function too)

    # dump YAML for debugging

    # in the end, return both alignment object (if in memory)
    # and path to final alignment file

    return outcfg


# list of available alignment protocols
PROTOCOLS = {
    # standard buildali protocol (iterative hmmer search)
    "standard": standard,

    # start from an existing (external) alignment
    "external": external,
}


def run(**kwargs):
    """
    Run alignment protocol to generate multiple sequence
    alignment from input sequence.

    Parameters
    ----------
    Mandatory kwargs arguments:
        protocol: Alignment protocol to run
        prefix: Output prefix for all generated files

    Optional:
        skip: If True, only return stage results but do
        not run actual calculation.

    Returns
    -------
    Dictionary with results of stage in following fields:
        alignment_file
        statistics_file
        sequence_file
        search_sequence_file
        sequence_id
        segments
        focus_mode
    """
    check_required(kwargs, ["protocol"])

    return PROTOCOLS[kwargs["protocol"]](**kwargs)