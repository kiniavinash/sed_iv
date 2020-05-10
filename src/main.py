import sys
import argparse

from tools.settings import CLASS_TYPE

from experiments import only_static_mixed_loc, only_static_location_split


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
    train_test_combos = [("static", "driving"),
                         ("driving", "static"),
                         ("SA", "SB"),
                         ("SB", "SA"),
                         ("SA", "DA"),
                         ("SA", "DB"),
                         ("SB", "DA"),
                         ("SB", "DB"),
                         ("SA2", "DA"),
                         ("SB2", "DB"),
                         ("SA1", "SA2"),
                         ("SA2", "SA1"),
                         ("SB12", "SB3"),
                         ("SB23", "SB1"),
                         ("SB31", "SB2"),
                         ("static", None),
                         ("driving", None),
                         ("SA", None),
                         ("SB", None),
                         ("DA", None),
                         ("DB", None)]

    for combo in train_test_combos:
        only_static_location_split(parsed, combo[0], combo[1])
