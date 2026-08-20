from __future__ import annotations

from langchain_neuraltrust import __all__ as exported

EXPECTED_ALL = [
    "TrustGuardAuthError",
    "TrustGuardBlockedError",
    "TrustGuardEntitlementError",
    "TrustGuardError",
    "TrustGuardMiddleware",
    "TrustGuardRequestError",
    "TrustGuardTransformError",
    "TrustGuardUnknownVerdictError",
    "TrustGuardUnreachableError",
    "TrustGuardVerdict",
    "__version__",
]


def test_all_exports_match() -> None:
    assert sorted(exported) == sorted(EXPECTED_ALL)


def test_version_matches_metadata() -> None:
    from importlib.metadata import version

    from langchain_neuraltrust import __version__

    assert __version__ == version("langchain-neuraltrust")


def test_public_names_are_importable() -> None:
    import langchain_neuraltrust as pkg

    for name in EXPECTED_ALL:
        assert hasattr(pkg, name)
