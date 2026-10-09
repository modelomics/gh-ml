"""Independent regressions for malformed prefetch completion receipts."""

from datetime import timezone, datetime

import pytest

from gh_ml import gharchive_acquire as acquire


@pytest.mark.parametrize("payload", ["[]", "null", '"receipt"', "17"])
def test_non_object_prefetch_receipt_is_discarded_for_resume(tmp_path, payload):
    hour = datetime(2023, 8, 29, tzinfo=timezone.utc)
    part, receipt = acquire._prefetch_path(tmp_path, hour)
    part.write_bytes(b"interrupted or untrusted bytes")
    receipt.write_text(payload, encoding="utf-8")

    assert acquire._read_prefetch(tmp_path, hour) is None
    assert not part.exists()
    assert not receipt.exists()
