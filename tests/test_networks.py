from pathlib import Path

import pytest

from shellsteer.networks import load_network, read_networks

NETWORKS = Path(__file__).parent.parent / "configs" / "networks.yaml"


def test_networks_file_reads():
    assert read_networks(NETWORKS)["shellflow"].width == 128


def test_unknown_network_stops():
    with pytest.raises(ValueError, match="not in"):
        load_network("shellflow-c", NETWORKS)


def test_shellflow_loads_frozen():
    """Downloads the ShellFlow checkpoint from Hugging Face on the first run (about 24 MB)."""
    network = load_network("shellflow", NETWORKS)
    assert len(network.model.blocks) == 6
    assert sum(p.numel() for p in network.model.parameters()) == 3_016_898
    assert not network.model.training
    assert not any(p.requires_grad for p in network.model.parameters())
