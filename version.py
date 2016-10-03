# coding: utf-8
"""Unique place where the version number is defined."""

# Do not copy into the source folder !

from collections import namedtuple
_version_info = namedtuple(
    "version_info", ["major", "minor", "micro", "releaselevel", "serial"])

MAJOR = 0
MINOR = 0
MICRO = 1
RELEV = "dev"  # <16
SERIAL = 3  # <16

version_info = _version_info(MAJOR, MINOR, MICRO, RELEV, SERIAL)

strict_version = version = "%d.%d.%d" % version_info[:3]

RELEASE_LEVEL_VALUE = {"dev": 0,
                       "alpha": 10,
                       "beta": 11,
                       "rc": 12,
                       "final": 15}

if version_info.releaselevel != "final":
    version += "-%s%s" % version_info[-2:]
    prerel = "a" if RELEASE_LEVEL_VALUE.get(version_info[3], 0) < 10 else "b"
    if prerel not in "ab":
        prerel = "a"
    strict_version += prerel + str(version_info[-1])

hex_version = version_info[4]
hex_version |= RELEASE_LEVEL_VALUE.get(version_info[3], 0) * 1 << 4
hex_version |= version_info[2] * 1 << 8
hex_version |= version_info[1] * 1 << 16
hex_version |= version_info[0] * 1 << 24
