import sys
import argparse

import pandas as pd
import numpy as np

from tools.settings import REF_LABELS

from experiments import only_static_mixed_loc, custom_train_test_split


def parse_args():
    class ExtractorArgsParser(argparse.ArgumentParser):
        def error(self, message):
            self.print_help()
            sys.exit(2)

        def format_help(self):
            formatter = self._get_formatter()
            formatter.add_text(self.description)
            formatter.add_usage(self.usage, self._actions,
                                self._mutually_exclusive_groups)
            for action_group in self._action_groups:
                formatter.start_section(action_group.title)
                formatter.add_text(action_group.description)
                formatter.add_arguments(action_group._group_actions)
                formatter.end_section()
            formatter.add_text(self.epilog)

            return formatter.format_help()

    usage = """
    Training a deep-learning based acoustic model to predict cars coming around corners
    """
    parser = ExtractorArgsParser(description='python main.py',
                                 usage=usage)
    parser.add_argument('--test',
                        help='Only test the desired model',
                        action='store_true')

    try:
        parsed = parser.parse_args()
    except:
        sys.exit(2)

    return parsed


if __name__ == '__main__':

    parsed = parse_args()

    # only_static_mixed_loc(parsed)
    train_test_combos = [
        ("static", None),
        ("driving", None),
        ("SA", None),
        ("SB", None),
        ("DA", None),
        ("DB", None),
        # ("static", "driving"),
        # ("static", "DA"),
        # ("static", "DB"),
        # ("SA", "driving"),
        # ("SB", "driving"),
        # ("SA", "DA"),
        # ("SB", "DB"),
        # ("SA2", "driving"),
        # ("SB2", "driving"),
        # ("SA2", "DA"),
        # ("SB2", "DB"),
        # ("driving", None),
        # ("DA", None),
        # ("DB", None),
        # # # ===============
        # (("static", "DA"), "DB"),
        # (("static", "DB"), "DA"),
        # (("SA", "DB"), "DA"),
        # (("SB", "DA"), "DB"),
        # (("SA2", "DB"), "DA"),
        # (("SB2", "DA"), "DB"),
        # # # =================
        # (("SA2", "DA"), "DA"),
        # (("SB2", "DB"), "DB"),
        # (("static", "driving"), "driving"),
    ]

    random_seeds = [1, 56, 93]  # , 783, 832]

    # only for single category in train and test each
    for combo in train_test_combos:
        f1_score_avg = []
        conf_mat_avg = pd.DataFrame()
        over_acc_avg = []

        for seed in random_seeds:
            f1_score, conf_mat, over_acc = custom_train_test_split(parsed, combo[0], combo[1], seed)
            f1_score_avg.append(f1_score)
            conf_mat_avg = conf_mat_avg.add(conf_mat, fill_value=0)
            over_acc_avg.append(over_acc)

        f1_score_avg = np.array(f1_score_avg)
        over_acc_avg = np.array(over_acc_avg)

        print("Average Statistics:")
        print("Mean F1 score (strong labels): {} +/- {}".format(np.mean(f1_score_avg), np.std(f1_score_avg)))
        print("Confusion matrix (summed):")
        print(conf_mat_avg)
        print("Mean accuracy (weak labels): {} +/- {}".format(np.mean(over_acc_avg), np.std(over_acc_avg)))
        print("===========************************===================")

    # train_test_combos = [
    #     # ("static", "driving"),
    #     # ("driving", "static"),
    #     # ("SA", "SB"),
    #     # ("SB", "SA"),
    #     # ("SA", "DA"),
    #     # ("SA", "DB"),
    #     # ("SB", "DA"),
    #     # ("SB", "DB"),
    #     # ("SA2", "DA"),
    #     # ("SB2", "DB"),
    #     # ("SA1", "SA2"),
    #     # ("SA2", "SA1"),
    #     # ("SB12", "SB3"),
    #     # ("SB23", "SB1"),
    #     # ("SB31", "SB2"),
    #     # ("static", None),
    #     # ("driving", None),
    #     # ("SA", None),
    #     # ("SB", None),
    #     # ("DA", None),
    #     # ("DB", None),
    #     # ("DA", "SA"),
    #     # ("DB", "SB"),
    #     # ("SB2", "SB1"),
    #     # ("SB1", "SB2"),
    #     # ("SB3", "SB1"),
    #     # ("SB1", "SB3"),
    #     # ("SB3", "SB2"),
    #     # ("SB2", "SB3"),
    #     # ("SA", "driving"),
    #     # ("SB", "driving"),
    #     # ("static", "DA"),
    #     # ("static", "DB"),
    #     # ("DA", "DB"),
    #     # ("DB", "DA"),
    #     # ("SA1", None),
    #     # ("SA2", None),
    #     # ("SB1", None),
    #     # ("SB2", None),
    #     # ("SB3", None),
    #     # ("SB12", None),
    #     # ("SB23", None),
    #     # ("SB31", None),
    #     # ("driving", "SA"),
    #     # ("driving", "SB"),
    #     # ("SA2", "SB2"),
    #     # ("SB2", "SA2"),
    #     # # ===============
    #     # # (("static", "DA"), "DB")
    # ]
