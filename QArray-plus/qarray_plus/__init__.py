# <--- ensures ANSI codes are translated properly


def __print_welcome_message():
    from colorama import init

    init(autoreset=True)

    RESET = "\033[0m"
    BOLD = "\033[1m"
    CYAN = "\033[36m"
    YELLOW = "\033[33m"
    message = (
        f"\n{BOLD}{CYAN}Qarray-Latched{RESET} — successor to Qarray."
        f"\n{YELLOW}Author:{RESET} Barnaby van Straaten"
        f"\n{YELLOW}License:{RESET} GPL"
    )
    print(message)


__print_welcome_message()

from qarray_plus.ChargeSensor.ChargeSensor import ChargeSensor
from qarray_plus.DotArray.DotArray import DotArray
from qarray_plus.DotArray.ground_state.ground_state import ground_state_open
from qarray_plus.GateVoltageComposer import GateVoltageComposer
from qarray_plus.utils import charge_state_to_scalar
