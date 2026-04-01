"""Tests for retrieval data models."""

from archivist.retrieval.models import DiffResult, FamilyInfo, SearchResult


class TestSearchResult:
    def test_construction(self) -> None:
        r = SearchResult(
            text="some text",
            score=0.95,
            source_file="doc.pdf",
            family_slug="nginx",
            doc_title="Nginx Admin Guide",
            doc_type="admin_guide",
            version="1.24",
            page_number=5,
            heading_path="Config > TLS",
            chunk_role="base",
        )
        assert r.text == "some text"
        assert r.score == 0.95
        assert r.family_slug == "nginx"
        assert r.version == "1.24"
        assert r.chunk_role == "base"

    def test_frozen(self) -> None:
        r = SearchResult(
            text="t", score=0.5, source_file="f", family_slug="s",
            doc_title="d", doc_type="t", version=None,
            page_number=None, heading_path=None, chunk_role="base",
        )
        try:
            r.score = 0.1  # type: ignore[misc]
            assert False, "Should be frozen"
        except AttributeError:
            pass

    def test_optional_fields_none(self) -> None:
        r = SearchResult(
            text="t", score=0.5, source_file="f", family_slug="s",
            doc_title="d", doc_type="t", version=None,
            page_number=None, heading_path=None, chunk_role="delta",
        )
        assert r.version is None
        assert r.page_number is None
        assert r.heading_path is None


class TestFamilyInfo:
    def test_construction(self) -> None:
        f = FamilyInfo(
            family_slug="nginx",
            doc_types=["admin_guide", "release_notes"],
            versions=["1.22", "1.24"],
            latest_version="1.24",
            total_chunks=150,
        )
        assert f.family_slug == "nginx"
        assert len(f.doc_types) == 2
        assert f.latest_version == "1.24"

    def test_empty_versions(self) -> None:
        f = FamilyInfo(
            family_slug="misc",
            doc_types=["other"],
            versions=[],
            latest_version=None,
            total_chunks=10,
        )
        assert f.versions == []
        assert f.latest_version is None


class TestDiffResult:
    def test_construction(self) -> None:
        d = DiffResult(
            chunk_text="new content",
            change_type="added",
            source_file="doc.pdf",
            chunk_index=3,
            heading_path="Setup",
        )
        assert d.change_type == "added"
        assert d.chunk_index == 3

    def test_change_types(self) -> None:
        for ct in ("added", "modified", "removed"):
            d = DiffResult(
                chunk_text="x", change_type=ct,
                source_file="f", chunk_index=0, heading_path=None,
            )
            assert d.change_type == ct
