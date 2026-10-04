"""DEVS↔HLA bridge for pyjevsim coupled models.

Public API:

    from pyjevsim_bridge import HLAFederate, PortMapping

    federate = HLAFederate(
        coupled_model=my_coupled_model,
        federation=FederationSpec(name="demo", fom_modules=["./demo.fom.xml"]),
        federate_name="alice",
        port_mapping=PortMapping({
            "out_position": "Position",   # pyjevsim port -> FOM interaction class
            "in_command": "Command",
        }),
    )
    await federate.run(ticks=100)

Object-class extension: a coupled model may also implement methods on
:class:`ObjectClassFederateProtocol` to participate in HLA OBJECT-CLASS
pub/sub (``register_object_instance`` / ``update_attributes`` /
``ReflectAttributeValues``). Detection is duck-typed via ``hasattr``;
existing interaction-only models keep working unchanged. See
``examples/pyjevsim-dashboard-bridged/`` for a worked example.

The bridge preserves DEVS external-first transition semantics while using
HLA Next Message Request for logical-time advancement.
"""

from __future__ import annotations

from pyjevsim_bridge._protocol import (
    CoupledModelProtocol,
    ObjectClassFederateProtocol,
)
from pyjevsim_bridge.port_mapping import PortMapping
from pyjevsim_bridge.time_advance import HLAFederate

__all__ = [
    "CoupledModelProtocol",
    "HLAFederate",
    "ObjectClassFederateProtocol",
    "PortMapping",
]
