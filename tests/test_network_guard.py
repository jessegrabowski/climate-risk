"""The offline suite is a stated non-negotiable and `conftest` enforces it, which makes the guard
itself worth a test. It lapsed once: a function-scoped guard leaves module- and session-scoped
fixture bodies free to reach upstream, and a seeded cache that stopped matching downloaded for
weeks without failing anything."""

import socket

import pytest

from tests.conftest import NetworkAccessError


@pytest.fixture(scope="module")
def lookup_from_a_module_scoped_fixture():
    """Resolve a name during module-scoped setup, which runs before any function-scoped fixture."""
    with pytest.raises(NetworkAccessError):
        socket.getaddrinfo("example.com", 80)

    return "refused"


def test_a_module_scoped_fixture_cannot_reach_upstream(lookup_from_a_module_scoped_fixture):
    assert lookup_from_a_module_scoped_fixture == "refused"


def test_a_test_body_cannot_reach_upstream():
    with pytest.raises(NetworkAccessError):
        socket.getaddrinfo("example.com", 80)


def test_a_test_body_cannot_open_a_connection():
    with pytest.raises(NetworkAccessError), socket.socket() as opened:
        opened.connect(("93.184.216.34", 80))
