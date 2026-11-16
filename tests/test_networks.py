from pathlib import Path

from shellsteer.config import load_config, read_networks
from shellsteer.networks import load_network

MAIN = Path(__file__).parent.parent / "configs" / "paper" / "main_z_block1.yaml"


def test_shellflow_loads_frozen():
    """Downloads the ShellFlow checkpoint from Hugging Face on the first run (about 24 MB)."""
    config = load_config(MAIN)
    network = load_network(config.networks[config.network], "cpu")
    assert len(network.model.blocks) == 6
    assert sum(p.numel() for p in network.model.parameters()) == 3_016_898
    assert not network.model.training
    assert not any(p.requires_grad for p in network.model.parameters())


def test_networks_file_reads_alone():
    entry = read_networks(MAIN.parent.parent / "networks.yaml")["shellflow"]
    assert entry.width == 128
