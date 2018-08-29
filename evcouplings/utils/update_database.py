"""
command-line app to update the necessary databases

Authors:
  Benjamin Schubert
  Anna G. Green
"""
import ftplib
import datetime
import os
import errno
import zlib
import tempfile
import pandas as pd
from functools import partial
from pathlib import Path
from Bio import SeqIO

import click

from evcouplings.compare import SIFTS
from evcouplings.utils import Progressbar
from evcouplings.utils import verify_resources

UNIPROT_URL = "ftp.uniprot.org"
UNIPROT_CWD = "/pub/databases/uniprot/current_release/knowledgebase/complete/"
UNIPROT_FILE = "uniprot_{type}.fasta.gz"

DB_URL = "ftp.uniprot.org"
DB_CWD = "/pub/databases/uniprot/uniref/{type}/"
DB_FILE = "{type}.fasta.gz"

DB_SUFFIX = "{type}_{year}_{month}.fasta"
DB_CURRENT = "{type}_current.fasta"

SIFTS_SUFFIX = "pdb_chain_uniprot_plus_{year}_{month}_{day}.{extension}"
SIFTS_CURRENT = "pdb_chain_uniprot_plus_current.{extension}"

UNIPROT_IDMAPPING_CWD = "/pub/databases/uniprot/current_release/knowledgebase/idmapping/"
UNIPROT_IDMAPPING_FILE = "idmapping_selected.tab.gz"
UNIPROT_IDMAPPING_TABLE = "idmapping_uniprot_embl_{year}_{month}_{day}.txt"

EMBL_CDS_URL = "ftp.ebi.ac.uk"
EMBL_CDS_CWD = "/pub/databases/ena/coding/"
EMBL_SUBDIRS = [
    "update/std", "update/wgs", "update/con", "update/tsa",
    "release/std", "release/wgs", "release/con", "release/tsa"
]
CDS_OUTPUT_FILE = "embl_genome_location_table_{year}_{month}_{day}.txt"

def symlink_force(target, link_name):
    """
    Creates or overwrites an existing symlink

    Parameters
    ----------
    target : str
        the target file path
    link_name : str
        the symlink name

    """
    try:
        os.symlink(target, link_name)
    except OSError as e:
        if e.errno == errno.EEXIST:
            os.remove(link_name)
            os.symlink(target, link_name)
        else:
            raise e


def download_ftp_file(ftp_url, ftp_cwd, file_url, output_path, file_handling="wb", gziped=False, verbose=False):
    """
    Downloads a gzip file from a remote ftp server and
    decompresses it on the fly into an output file

    Parameters
    ----------
    ftp_url : str
        the FTP server url
    ftp_cwd : str
        the FTP directory of the file to download
    file_url : str
        the file name that gets downloaded
    output_path : str
        the path to the output file on the local system
    file_handling : str
        the file handling mode (default: 'wb')
    verbose : bool
        determines whether a progressbar is printed
    """
    def _callback(_bar, decompressor, chunk):
        if gziped:
            out.write(decompressor.decompress(chunk))
        else:
            out.write(chunk)
        if verbose:
            _bar += len(chunk)

    ftp = ftplib.FTP(ftp_url)
    ftp.login()
    ftp.cwd(ftp_cwd)
    with open(output_path, file_handling) as out:
        print(ftp_cwd, file_url)
        filesize = ftp.size(file_url)
        pbar = Progressbar(filesize) if verbose else None

        # automatic header detection
        decompressor = zlib.decompressobj(zlib.MAX_WBITS | 32)
        callback = partial(_callback, pbar, decompressor)
        ftp.retrbinary('RETR %s' % file_url, callback,
                       blocksize=8192)
    ftp.quit()

def parse_uniprot_idmapping_table(table_file, output_file):
    """
    Parameters
    ----------
    table_file: str
        path to downloaded, decompressed uniprot id mapping table
    output_file: str
        path to file to write
    """
    verify_resources(
        "Invalid uniprot IDmapping file {}".format(table_file), table_file
    )

    with open(output_file, "w") as of:
        with open(table_file) as inf:
            for line in inf:

                contents = line.split("\t")

                # check if line has all proper contents
                if len(contents) > 17:

                    uniprot_ac = contents[0]
                    uniprot_id = contents[1]

                    # TODO: saving uniref100 id for mapping later
                    uniref100_id = contents[7]

                    embl_genome_id_string = contents[16].replace(" ","")
                    embl_transcript_id_string = contents[17].replace(" ","")

                    embl_genome_ids = embl_genome_id_string.split(";")
                    embl_transcript_ids = embl_transcript_id_string.split(";")

                    # pair up genome and transcript ids
                    paired_ids = [
                        (g, t) for (g, t) in zip(embl_genome_ids, embl_transcript_ids)
                    ]

                    # remove the pairs that are missing information
                    cleaned_pairs = [
                        pair for pair in paired_ids if not "-" in pair and not "" in pair
                    ]

                    if len(cleaned_pairs) > 0:
                        string_to_write = "{} {} {}".format(
                            uniprot_ac, uniprot_id, ",".join(map(lambda x: ":".join(x), cleaned_pairs))
                        )

                        of.write(string_to_write)


def extract_embl_cds_list(input_file):
    """
    Extracts
    Parameters
    ----------
    input_file: str
        the open ENA-EMBL file
    Returns
    -------
    """


    # extract the relevant information
    cds_list = []

    for record in SeqIO.parse(open(input_file, "r"), "embl"):
        cds_per_record = []
        for feature in record.features:
            if feature.type == "CDS":

                start, end = feature.location.start.position, feature.location.end.position
                uniprot_id_list = []
                if "db_xref" in feature.qualifiers:
                    uniprot_id_list = [
                        r.split(":")[1] for r in feature.qualifiers["db_xref"] if r.startswith("UniProt")
                    ]

                cds_per_record.append((record.id, feature.ref, ",".join(uniprot_id_list), start, end))

        if len(cds_per_record) == 1:
            cds_list += cds_per_record
    return cds_list


def embl_mapping_by_directory(ftp_url, ftp_cwd, cds_output_path, cds_output_file, gziped=False, verbose=False):
    """
    Parameters
    ----------
    ftp_url: str
    ftp_cwd: str
    cds_output_file: str
    gziped
    verbose
    """

    def _callback(out, decompressor, chunk):
        if gziped:
            out.write(decompressor.decompress(chunk))
        else:
            out.write(chunk)

    ftp = ftplib.FTP(ftp_url)
    ftp.login()
    ftp.cwd(ftp_cwd)
    files = []
    decompressor = zlib.decompressobj(zlib.MAX_WBITS | 32)

    # each directory contains many files
    # get a complete list of files in the directory
    ftp.dir(files.append)

    full_cds_output_file = os.path.join(cds_output_path, cds_output_file)
    with open(full_cds_output_file, "a") as master_output_file:

        # iterate through each file, download and extract info
        for f in files:
            #file is the last argument in the returned string
            file_to_download = f.split(" ")[-1]

            # download and unzip the file to a tempfile
            temp = tempfile.NamedTemporaryFile(prefix=cds_output_path + "/", delete=False)
            with open(temp.name, "wb") as ena_temp_file:

                callback = partial(_callback, ena_temp_file, decompressor)
                ftp.retrbinary("RETR {}".format(file_to_download), callback, blocksize=8192)

            # get the genome location information and uniprot mapping
            # from the CDS file

            cds_list = extract_embl_cds_list(temp.name)

            # Ensure that the file only had one unique mapping
            for cds_annotation in cds_list:
                cds_string = "\t".join(map(str, cds_annotation))

                # write to the master mapping table
                master_output_file.write(cds_string + "\n")

            # clean up the output file
            os.unlink(temp.name)



def run(**kwargs):
    """
    Exposes command line interface as a Python function.

    Parameters
    ----------
    kwargs
        See click.option decorators for app() function
    """
    now = datetime.datetime.now()
    year = now.year
    month = now.month
    day = now.day
    verbose = kwargs.get("verbose", False)
    symlink = kwargs.get("symlink", False)

    # # update SIFTS file
    # if verbose:
    #     print("Updating SIFTS")
    #
    # SIFTS_dir = os.path.abspath(kwargs.get("sifts", os.path.realpath(__file__)))
    # # create directory if it does not exist
    # # ignores if directory on the way already exist
    # dir = Path(SIFTS_dir)
    # dir.mkdir(parents=True, exist_ok=True)
    # sifts = os.path.join(SIFTS_dir, SIFTS_SUFFIX)
    # sifts_curr = os.path.join(SIFTS_dir, SIFTS_CURRENT)
    # sifts_table = sifts.format(year=year, month=month, day=day, extension="csv")
    # sifts_fasta = sifts.format(year=year, month=month, day=day, extension="fasta")
    # s_new = SIFTS(sifts.format(year=year, month=month, day=day, extension="csv"))
    # s_new.create_sequence_file(sifts.format(year=year, month=month, day=day, extension="fasta"))
    #
    # # set symlink to "<file>_current"
    # if symlink:
    #     symlink_force(sifts_table, sifts_curr.format(extension="csv"))
    #     symlink_force(sifts_fasta, sifts_curr.format(extension="fasta"))
    #
    # # update uniref
    # db_path = os.path.abspath(kwargs.get("db", os.path.realpath(__file__)))
    # for db_type in ["uniprot", "uniref100", "uniref90" ]:
    #
    #     if verbose:
    #         print("Updating", db_type)
    #
    #     # if not existent create folder db_path/db_type
    #     db_full_path = os.path.join(db_path, db_type)
    #     dir = Path(db_full_path)
    #     dir.mkdir(parents=True, exist_ok=True)
    #
    #     if db_type == "uniprot":
    #         # download Swiss and TrEMBL and concatinate both
    #         out_path = os.path.join(db_full_path, DB_SUFFIX.format(type=db_type, year=year, month=month))
    #         db_curr = os.path.join(db_full_path, DB_CURRENT.format(type=db_type))
    #         for i, type_d in enumerate(["sprot", "trembl"]):
    #             if i:
    #                 file_url = UNIPROT_FILE.format(type=type_d)
    #                 download_ftp_file(UNIPROT_URL, UNIPROT_CWD, file_url, out_path, gziped=True,
    #                                   file_handling="ab", verbose=verbose)
    #             else:
    #                 file_url = UNIPROT_FILE.format(type=type_d)
    #                 download_ftp_file(UNIPROT_URL, UNIPROT_CWD, file_url, out_path, gziped=True, verbose=verbose)
    #     else:
    #         # download uniref db
    #         db_file = DB_FILE.format(type=db_type)
    #         db_cwd = DB_CWD.format(type=db_type)
    #         out_path = os.path.join(db_full_path, DB_SUFFIX.format(type=db_type, year=year, month=month))
    #         db_curr = os.path.join(db_full_path, DB_CURRENT.format(type=db_type))
    #         download_ftp_file(DB_URL, db_cwd, db_file, out_path, gziped=True, verbose=verbose)
    #
    #     if symlink:
    #         symlink_force(out_path, db_curr)

    # update EVcomplex ENA mapping databases
    # if verbose:
    #     print("Updating Uniprot to ENA ID Mapping table")
    #
    # # download the ID mapping file (Uniprot to ENA)
    #
    # # temporary file to download
    # idmapping_dir = os.path.abspath(kwargs.get("idmapping", os.path.realpath(__file__)))
    # tf = tempfile.NamedTemporaryFile(delete=False, prefix=idmapping_dir)
    # out_path = tf.name
    #
    # download_ftp_file(
    #   UNIPROT_URL, UNIPROT_IDMAPPING_CWD, UNIPROT_IDMAPPING_FILE,
    #   out_path, gziped=True, file_handling="wb", verbose=verbose
    # )
    #
    # full_mapping_table = os.path.join(
    #     idmapping_dir, UNIPROT_IDMAPPING_TABLE.format(year=year, month=month, day=day)
    # )
    #
    # # parse into correct format
    # print("beginning parsing")
    # parse_uniprot_idmapping_table(
    #     out_path,
    #     full_mapping_table
    # )
    # os.unlink(out_path)

    # download EMBL-ENA CDS data
    if verbose:
        print("Updating EMBL-ENA CDS data")

    cds_output_dir = os.path.abspath(kwargs.get("embl_ena", os.path.realpath(__file__)))
    cds_output_file = CDS_OUTPUT_FILE.format(year=year, month=month, day=day)

    for subdirectory in EMBL_SUBDIRS:
        ftp_url = os.path.join(EMBL_CDS_CWD, subdirectory)
        embl_mapping_by_directory(EMBL_CDS_URL, ftp_url, cds_output_dir, cds_output_file,  gziped=True, verbose=verbose)
        break



CONTEXT_SETTINGS = dict(help_option_names=["-h", "--help"])

@click.command(context_settings=CONTEXT_SETTINGS)

# run settings
@click.option("-s", "--sifts", default="/groups/marks/databases/SIFTS/", help="SIFTS output directory")
@click.option("-d", "--db", default="/groups/marks/databases/jackhmmer/", help="Uniprot output directory")
@click.option(
    "-i", "--idmapping", default="/groups/marks/databases/complexes/idmapping/",
    help="Uniprot idmapping output directory"
)
@click.option(
    "-e", "--embl_ena", default="/groups/marks/databases/complexes/ena/",
    help="EMBL-ENA genome location output directory"
)
@click.option(
    "-l", "--symlink", default=False, is_flag=True,
    help="Creates symlink with ending '_current.' pointing to the newly created db files"
)
@click.option("-v", "--verbose", default=False, is_flag=True, help="Enables verbose output")
def app(**kwargs):
    """
    Update database command line interface
    """
    run(**kwargs)

if __name__ == "__main__":
    app()
