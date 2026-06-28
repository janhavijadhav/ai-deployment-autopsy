"""
Tests for the data layer: supplier lookup, search, schema init.
"""

from __future__ import annotations

import pytest
from unittest.mock import patch


class TestSupplierLookup:

    @pytest.mark.asyncio
    async def test_get_existing_supplier(self, tmp_sqlite):
        with patch("src.data.database.settings") as mock_settings:
            mock_settings.SQLITE_PATH = tmp_sqlite
            from src.data.database import get_supplier
            record = await get_supplier("SUP-0001")

        assert record is not None
        assert record["supplier_id"] == "SUP-0001"
        assert record["name"] == "Apex Industries"
        assert record["country"] == "CN"
        assert record["risk_score"] == pytest.approx(0.82)

    @pytest.mark.asyncio
    async def test_get_nonexistent_supplier_returns_none(self, tmp_sqlite):
        with patch("src.data.database.settings") as mock_settings:
            mock_settings.SQLITE_PATH = tmp_sqlite
            from src.data.database import get_supplier
            record = await get_supplier("SUP-9999")

        assert record is None

    @pytest.mark.asyncio
    async def test_search_by_country(self, tmp_sqlite):
        with patch("src.data.database.settings") as mock_settings:
            mock_settings.SQLITE_PATH = tmp_sqlite
            from src.data.database import search_suppliers
            results = await search_suppliers(country="CN")

        assert len(results) == 1
        assert results[0]["country"] == "CN"

    @pytest.mark.asyncio
    async def test_search_by_risk_threshold(self, tmp_sqlite):
        with patch("src.data.database.settings") as mock_settings:
            mock_settings.SQLITE_PATH = tmp_sqlite
            from src.data.database import search_suppliers
            high_risk = await search_suppliers(min_risk=0.7)

        # Only SUP-0001 has risk_score 0.82 >= 0.7
        assert len(high_risk) == 1
        assert high_risk[0]["supplier_id"] == "SUP-0001"

    @pytest.mark.asyncio
    async def test_search_by_max_risk(self, tmp_sqlite):
        with patch("src.data.database.settings") as mock_settings:
            mock_settings.SQLITE_PATH = tmp_sqlite
            from src.data.database import search_suppliers
            low_risk = await search_suppliers(max_risk=0.3)

        # Only SUP-0002 has risk_score 0.21
        assert all(r["risk_score"] <= 0.3 for r in low_risk)

    @pytest.mark.asyncio
    async def test_search_returns_at_most_limit(self, tmp_sqlite):
        with patch("src.data.database.settings") as mock_settings:
            mock_settings.SQLITE_PATH = tmp_sqlite
            from src.data.database import search_suppliers
            results = await search_suppliers(limit=1)

        assert len(results) <= 1

    @pytest.mark.asyncio
    async def test_get_contract_metadata(self, tmp_sqlite):
        with patch("src.data.database.settings") as mock_settings:
            mock_settings.SQLITE_PATH = tmp_sqlite
            from src.data.database import get_contract_metadata
            contract = await get_contract_metadata("CTR-00001")

        assert contract is not None
        assert contract["contract_id"] == "CTR-00001"
        assert contract["supplier_id"] == "SUP-0001"
        assert contract["auto_renewal"] == 1

    @pytest.mark.asyncio
    async def test_get_nonexistent_contract_returns_none(self, tmp_sqlite):
        with patch("src.data.database.settings") as mock_settings:
            mock_settings.SQLITE_PATH = tmp_sqlite
            from src.data.database import get_contract_metadata
            result = await get_contract_metadata("CTR-99999")

        assert result is None
