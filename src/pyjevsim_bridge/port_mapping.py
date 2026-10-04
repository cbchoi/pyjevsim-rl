"""Explicit pyjevsim port-to-FOM-class bindings."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PortMapping:
    """Bidirectional mapping between pyjevsim port names and FOM classes.

    out_ports maps {pyjevsim_output_port_name: FOM_interaction_class_name}
    in_ports  maps {FOM_interaction_class_name: pyjevsim_input_port_name}

    Construction takes a single dict and the implementation splits it into
    out/in based on the coupled model's port directions.
    """

    out_ports: dict[str, str]
    in_ports: dict[str, str]

    @classmethod
    def from_dict(cls, mapping: dict[str, str]) -> PortMapping:
        """Build a PortMapping from a flat {port: class} dict.

        Cut-1 direction inference uses the port-name prefix:

        * ``out_*`` -> outgoing port (added to ``out_ports`` as
          ``{port_name: fom_class}``).
        * ``in_*``  -> incoming port (added to ``in_ports`` as
          ``{fom_class: port_name}`` so subscribers can resolve the
          inbound port from the FOM class name).

        Any other prefix is rejected with a :class:`ValueError` to force
        the caller to be explicit. Future revisions may consult
        ``coupled_model.get_ports()`` for direction; until then, the
        ``out_``/``in_`` convention is the contract.
        """
        out_ports: dict[str, str] = {}
        in_ports: dict[str, str] = {}
        for port_name, fom_class in mapping.items():
            if port_name.startswith("out_"):
                out_ports[port_name] = fom_class
            elif port_name.startswith("in_"):
                in_ports[fom_class] = port_name
            else:
                raise ValueError(
                    f"PortMapping.from_dict: port name {port_name!r} lacks an"
                    " 'out_' or 'in_' prefix; cut-1 requires explicit"
                    " direction-prefixed port names."
                )
        return cls(out_ports=out_ports, in_ports=in_ports)

    def fom_class_for_out_port(self, port: str) -> str | None:
        """Lookup helper — None if port is not mapped."""
        return self.out_ports.get(port)

    def in_port_for_fom_class(self, class_name: str) -> str | None:
        """Lookup helper — None if class is not subscribed via this mapping."""
        return self.in_ports.get(class_name)
