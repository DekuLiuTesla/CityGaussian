import sys

STOP_INDICATOR = "--"


def split_stoppable_args(argv: list):
    for idx, v in enumerate(argv + [STOP_INDICATOR]):
        if v == STOP_INDICATOR:
            break

    return argv[:idx], argv[idx + 1:]


def parser_stoppable_args(parser):
    argvs = split_stoppable_args(sys.argv[1:])

    return parser.parse_args(argvs[0]), argvs[1]


def test_split_stoppable_args():
    assert split_stoppable_args(["-a", "1", "-b", "2"]) == (["-a", "1", "-b", "2"], [])
    assert split_stoppable_args(["-a", "-b", "--", "-c"]) == (["-a", "-b"], ["-c"])
