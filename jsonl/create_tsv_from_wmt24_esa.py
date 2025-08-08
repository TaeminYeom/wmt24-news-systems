import json
import copy
import random
import logging
import argparse
import pandas as pd
from datasets import load_dataset
from collections import Counter
from collections import defaultdict

logger = logging.getLogger(__name__)


def generate_from_jsonl(input_path):
    with open(input_path) as f_in:
        for line in f_in:
            yield json.loads(line)


def assert_span_type(esa_span):
    assert isinstance(esa_span["start_i"], int) or (
        (esa_span["start_i"] == "missing") and (esa_span["end_i"] == "missing")
    )
    assert (isinstance(esa_span["end_i"], int)) or (
        (esa_span["end_i"] == "missing") and (esa_span["start_i"] == "missing")
    )
    assert esa_span["severity"] in ("major", "minor", "undecided")


def correct_span(error_span, hypothesis):
    error_span = copy.deepcopy(error_span)
    if error_span["severity"] == "critical":
        error_span["severity"] = "major"

    if isinstance(error_span["start_i"], int) and isinstance(error_span["end_i"], int):
        error_span["start_i"] = min(max(0, error_span["start_i"]), len(hypothesis))
        error_span["end_i"] = min(max(0, error_span["end_i"]), len(hypothesis))

    if error_span["start_i"] > error_span["end_i"]:
        error_span["start_i"], error_span["end_i"] = (
            error_span["end_i"],
            error_span["start_i"],
        )

    if error_span["start_i"] == error_span["end_i"]:
        error_span["start_i"] = "missing"
        error_span["end_i"] = "missing"

    return error_span


def is_invalid_span(error_span, hypothesis):
    start_i = error_span["start_i"]
    end_i = error_span["end_i"]
    severity = error_span["severity"]
    if isinstance(start_i, str) and start_i != "missing":
        return True
    if isinstance(end_i, str) and end_i != "missing":
        return True
    if isinstance(start_i, int) and ((start_i < 0) or (len(hypothesis) < start_i)):
        return True
    if isinstance(end_i, int) and ((end_i < 0) or (len(hypothesis) < end_i)):
        return True
    if (isinstance(start_i, int) and isinstance(end_i, int)) and (start_i > end_i):
        return True
    if severity not in ("major", "minor", "critical"):
        return True
    return False


def random_valid_span(hypothesis):
    start_i = random.randrange(len(hypothesis) + 1)
    end_i = random.randrange(len(hypothesis) + 1)
    while start_i == end_i:
        end_i = random.randrange(len(hypothesis) + 1)
    if end_i < start_i:
        start_i, end_i = end_i, start_i
    error_type = "major" if random.randrange(2) == 0 else "minor"
    return start_i, end_i, error_type


langs_to_wmt24pp_code = {
    "en-cs": "en-cs_CZ",
    "en-ja": "en-ja_JP",
    "en-zh": "en-zh_CN",
    "en-is": "en-is_IS",
    "en-uk": "en-uk_UA",
    "en-ru": "en-ru_RU",
}


def main(
    wmt24_esa_jsonl,
    output_path,
    filter_data_with_invalid_span,
    seed,
):
    if filter_data_with_invalid_span:
        logger.info(f"filter data with invalid span: {filter_data_with_invalid_span}")

    random.seed(seed)
    log_counter = Counter()

    langs_to_data = defaultdict(lambda: defaultdict(list))
    for datum in generate_from_jsonl(wmt24_esa_jsonl):
        assert len(datum["src"]) > 0
        if len(datum["tgt"]) == 0:
            log_counter["target segment is empty"] += 1
            continue
        langs = datum["langs"]
        signature = f"{datum['doc_id']}-{datum['line_id']}-{datum['system']}"
        langs_to_data[langs][signature].append(datum)

    valid_signature_counter = Counter()
    for langs in langs_to_data:
        valid_signature_counter[langs] = len(langs_to_data[langs])

    langs_to_statistics = defaultdict(Counter)
    data = defaultdict(list)
    for langs in langs_to_data:
        if langs not in langs_to_wmt24pp_code:
            continue

        # prepare references
        wmt24pp = load_dataset("google/wmt24pp", langs_to_wmt24pp_code[langs])["train"]
        src_to_tgt = {}
        for datum in wmt24pp:
            if datum["is_bad_source"]:
                continue
            src_to_tgt[datum["source"]] = datum["target"]

        num_langs_data = 0
        for signature in langs_to_data[langs]:
            for datum in langs_to_data[langs][signature]:
                # skip if there is no reference
                if datum["src"] not in src_to_tgt:
                    continue

                start_indices = []
                end_indices = []
                error_types = []

                has_invalid_span = False
                esa_counter = Counter()
                for esa_span in datum["esa_spans"]:
                    # type of span
                    # 1. Span indices should be "missing" or integers
                    # 2. severity should be in ["major", "minor", "undecided"]
                    assert_span_type(esa_span)

                    # Valid span:
                    # 1. has valid span indices
                    # 2. has severity with major, minor, or critical
                    if filter_data_with_invalid_span and is_invalid_span(
                        esa_span, datum["tgt"]
                    ):
                        has_invalid_span = True
                        break

                    # missing
                    if esa_span["start_i"] == esa_span["end_i"]:
                        esa_counter["missing"] += 1
                    else:
                        esa_counter[esa_span["severity"]] += 1

                    start_indices.append(f'{esa_span["start_i"]}')
                    end_indices.append(f'{esa_span["end_i"]}')
                    error_types.append(esa_span["severity"])

                if has_invalid_span:
                    log_counter["skip_datum_with_invalid_span"] += 1
                    continue

                if len(start_indices) == 0:
                    start_indices.append("-1")
                    end_indices.append("-1")
                    error_types.append("no-error")

                data["doc_id"].append(datum["doc_id"])
                data["segment_id"].append(datum["line_id"])
                source_lang, target_lang = datum["langs"].split("-")
                data["source_lang"].append(source_lang)
                data["target_lang"].append(target_lang)
                data["set_id"].append("official")
                data["system_id"].append(datum["system"])
                data["source_segment"].append(datum["src"])
                data["hypothesis_segment"].append(datum["tgt"])
                data["reference_segment"].append(src_to_tgt[datum["src"]])
                data["domain_name"].append(datum["domain"])
                data["method"].append("ESA")
                data["start_indices"].append(" ".join(start_indices))
                data["end_indices"].append(" ".join(end_indices))
                data["error_types"].append(" ".join(error_types))

                if error_types[0] == "no-error":
                    langs_to_statistics[langs]["no-error"] += 1

                langs_to_statistics[langs]["major"] += esa_counter["major"]
                langs_to_statistics[langs]["minor"] += esa_counter["minor"]
                langs_to_statistics[langs]["missing"] += esa_counter["missing"]

                num_langs_data += 1
                # skip the same signature
                break

        logger.info(
            f"# {langs} data: {num_langs_data} from {valid_signature_counter[langs]} valid signatures:"
        )
        for error_type in ["no-error", "missing", "major", "minor"]:
            logger.info(f"- {error_type}: {langs_to_statistics[langs][error_type]}")

        log_counter["num_skipped_valid_signatures"] += (
            valid_signature_counter[langs] - num_langs_data
        )

    for option_name in log_counter:
        logger.info(f"# {option_name}: {log_counter[option_name]}")

    pd.DataFrame(data).to_csv(output_path, sep="\t", index=False, header=True)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--wmt24_esa_jsonl")
    parser.add_argument("-o", "--output_tsv")
    parser.add_argument("--filter_data_with_invalid_span", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()

if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=logging.INFO,
    )
    args = parse_args()
    main(
        args.wmt24_esa_jsonl,
        args.output_tsv,
        args.filter_data_with_invalid_span,
        args.seed,
    )
