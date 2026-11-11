"""v5-specific entry point; the generic Refiner trainer stays unchanged."""

from __future__ import annotations

from . import train


def main():
    train.main()


if __name__ == "__main__":
    main()
