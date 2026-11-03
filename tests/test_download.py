"""Guards the dataset download call without touching the network."""

import monai.apps

from src import data


def test_download_verifies_md5(monkeypatch, tmp_path):
    """MONAI publishes an MD5 for MSD tarballs, but newer MONAI versions default
    `download_and_extract` to sha256. The hash type must be passed explicitly,
    otherwise every download fails its integrity check."""
    calls = {}
    monkeypatch.setattr(monai.apps, "download_and_extract", lambda **kw: calls.update(kw))
    data.download_spleen(tmp_path)
    assert calls["hash_type"] == "md5"
    assert calls["hash_val"] == monai.apps.DecathlonDataset.md5["Task09_Spleen"]
