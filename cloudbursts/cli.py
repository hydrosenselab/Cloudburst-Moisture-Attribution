"""
cloudbursts.cli

"""

from __future__ import annotations

import argparse

from .config import EVENTS, get_event


def base_parser(description: str) -> argparse.ArgumentParser:
    """Argument parser with the options every script accepts."""
    p = argparse.ArgumentParser(description=description)
    p.add_argument("--events", nargs="*", default=None,
                   help="Event keys or numbers to run (default: all ten), e.g. --events 3 Leh_2010")
    return p


def selected_events(args):
    """Events chosen on the command line, in event-number order."""
    if not args.events:
        return list(EVENTS)
    chosen = {get_event(int(e) if str(e).isdigit() else e) for e in args.events}
    return sorted(chosen, key=lambda e: e.number)
