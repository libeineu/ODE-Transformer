from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
import os
import parser
import shutil
from tempfile import mkdtemp
import pyrouge
from argparse import ArgumentParser
from pathlib import Path
import sys

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=os.environ.get("LOGLEVEL", "INFO").upper(),
    stream=sys.stdout,
)
logger = logging.getLogger()


def write_to_file(filename, data):
    data = ".\n".join(data.decode().split(". "))
    with open(filename, "w") as fp:
        fp.write(data)


def prep_data(args,decode_dir, target_dir):
    with open(args.decodes_filename, "rb") as fdecodes:
        with open(args.targets_filename, "rb") as ftargets:
            for i, (d, t) in enumerate(zip(fdecodes, ftargets)):
                write_to_file(os.path.join(decode_dir, "rouge.%06d.txt" % (i + 1)), d)
                write_to_file(os.path.join(target_dir, "rouge.A.%06d.txt" % (i + 1)), t)
                if (i + 1 % 1000) == 0:
                    logger.info("Written %d examples to file" % i)


def init_args(args: ArgumentParser):
    args.add_argument("--decodes_filename", type=Path)
    args.add_argument("--targets_filename", type=Path)
    return args.parse_args()


def main(args):
    rouge = pyrouge.Rouge155()
    rouge.log.setLevel(logging.ERROR)
    rouge.system_filename_pattern = "rouge.(\\d+).txt"
    rouge.model_filename_pattern = "rouge.[A-Z].#ID#.txt"

    tmpdir = mkdtemp()
    logger.info("tmpdir: %s" % tmpdir)
    # system = decodes/predictions
    system_dir = os.path.join(tmpdir, "system")
    # model = targets/gold
    model_dir = os.path.join(tmpdir, "model")
    os.mkdir(system_dir)
    os.mkdir(model_dir)

    rouge.system_dir = system_dir
    rouge.model_dir = model_dir

    prep_data(args,rouge.system_dir, rouge.model_dir)

    rouge_scores = rouge.convert_and_evaluate()
    rouge_scores = rouge.output_to_dict(rouge_scores)
    for prefix in ["rouge_1", "rouge_2", "rouge_l"]:
        for suffix in ["f_score"]:
            key = "_".join([prefix, suffix])
            logger.info("%s: %.4f" % (key, rouge_scores[key]))

    # clean up after pyrouge
    shutil.rmtree(tmpdir)
    shutil.rmtree(rouge._config_dir)  # pylint: disable=protected-access
    shutil.rmtree(
        os.path.split(rouge._system_dir)[0]
    )  # pylint: disable=protected-access


if __name__ == "__main__":
    args = ArgumentParser()
    args=init_args(args)
    main(args)
