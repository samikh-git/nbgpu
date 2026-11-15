"""Backend registry."""

from __future__ import annotations

from nbgpu.backends.base import Backend, BackendError

KNOWN_BACKENDS: tuple[str, ...] = ("local", "modal")


def get_backend(name: str) -> Backend:
    """"local" -> LocalBackend(), "modal" -> ModalBackend(). Imports lazily so
    the Modal SDK is only loaded when used. Unknown names raise BackendError
    listing the known ones."""
    if name == "local":
        from nbgpu.backends.local import LocalBackend

        return LocalBackend()
    if name == "modal":
        from nbgpu.backends.modal import ModalBackend

        return ModalBackend()
    raise BackendError(f"unknown backend {name!r}; known backends: {', '.join(KNOWN_BACKENDS)}")


__all__ = ["Backend", "BackendError", "get_backend"]
