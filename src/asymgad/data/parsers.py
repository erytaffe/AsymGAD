"""Per-dataset field parsers.

Each parser takes the raw ``action_state`` string and returns a dict of
extracted atomic fields suitable for serialisation into the unified
``event_detail`` column.
"""

import re

# -- Auth ---
# LogOn_Success_Network_Kerberos -> 4 parts, underscore-delimited
# TGS_Success_?_? -> unknown fields use '?' sentinel
# AuthMap_Success_?_? -> some actions have no subtype/protocol
# ---
_AUTH_RE = re.compile(r"^([^_]+)_([^_]+)_([^_]+)_([^_]+)$")


def parse_auth_action(action_state: str) -> dict:
    """Parse auth *action_state* into action_category, status, subtype, protocol."""
    m = _AUTH_RE.match(action_state)
    if m:
        return {
            "action_category": m.group(1),
            "status": m.group(2),
            "subtype": m.group(3),
            "protocol": m.group(4),
        }
    return {"raw": action_state}


# -- Flows ---
# NetFlow_17_137>137 -> protocol=17(UDP), src=137, dst=137
# NetFlow_1_N2>N3 -> protocol=1(ICMP), src=N2, dst=N3
# ---
_FLOWS_RE = re.compile(r"^NetFlow_([^_]+)_([^>]+)>([^>]+)$")

# Noise ports - broadcast / infrastructure chatter with no lateral value
_NOISE_PORTS = {"137", "138", "123"}  # NetBIOS NS, NetBIOS DGM, NTP


def parse_flows_action(action_state: str) -> dict:
    """Parse flows *action_state* into protocol, src_port, dst_port."""
    m = _FLOWS_RE.match(action_state)
    if m:
        return {
            "protocol": m.group(1),
            "src_port": m.group(2),
            "dst_port": m.group(3),
        }
    return {"raw": action_state}


def is_noise_flow(protocol: str, dst_port: str) -> bool:
    """Return True for UDP broadcast / infrastructure noise flows."""
    return protocol == "17" and dst_port in _NOISE_PORTS


# -- Proc ---
# Process_Start -> action=Start
# Process_End -> action=End
# ---


def parse_proc_action(action_state: str) -> dict:
    """Parse proc *action_state* - trivial: just strip the 'Process_' prefix."""
    return {"action": action_state.replace("Process_", "", 1)}


# -- Redteam ---
# RedTeam_via_C17693 -> pivot_machine=C17693
# ---
_REDTEAM_RE = re.compile(r"RedTeam_via_(C\d+)")


def parse_redteam_action(action_state: str) -> dict:
    """Parse redteam *action_state* to extract the pivot / jump-host."""
    m = _REDTEAM_RE.search(action_state)
    if m:
        return {"pivot_machine": m.group(1)}
    return {"raw": action_state}
