# Copyright (C) 2021 cryzed
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as
# published by the Free Software Foundation, either version 3 of the
# License, or (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

import collections
import os
import typing as T
from collections import abc

from hydrus_api import DEFAULT_API_URL, HYDRUS_METADATA_ENCODING, Client, Permission

X = T.TypeVar("X")


class TextFileLike(T.Protocol):
    def read(self) -> str:
        pass


def verify_permissions(
    client: Client, permissions: abc.Iterable[T.Union[int, Permission]], exact: bool = False
) -> bool:
    granted_permissions = set(client.verify_access_key()["basic_permissions"])
    return granted_permissions == set(permissions) if exact else granted_permissions.issuperset(permissions)


def cli_request_api_key(
    name: str,
    permissions: abc.Iterable[T.Union[int, Permission]],
    verify: bool = True,
    exact: bool = False,
    api_url: str = DEFAULT_API_URL,
) -> str:
    while True:
        input(
            'Navigate to "services->review services->local->client api" in the Hydrus client and click "add->from api '
            'request". Then press enter to continue...'
        )
        access_key = Client(api_url=api_url).request_new_permissions(name, permissions)
        input("Press OK and then apply in the Hydrus client dialog. Then press enter to continue...")

        client = Client(access_key, api_url)
        if verify and not verify_permissions(client, permissions, exact):
            granted = client.verify_access_key()["basic_permissions"]
            print(
                f"The granted permissions ({granted}) differ from the requested permissions ({permissions}), please "
                "grant all requested permissions."
            )
            continue

        return access_key


def parse_hydrus_metadata(text: str) -> collections.defaultdict[T.Optional[str], set[str]]:
    namespaces = collections.defaultdict(set)
    for line in (line.strip() for line in text.splitlines()):
        if not line:
            continue

        parts = line.split(":", 1)
        namespace, tag = (None, line) if len(parts) == 1 else parts
        namespaces[namespace].add(tag)

    # Ignore type, mypy has trouble figuring out that tag isn't optional
    return namespaces  # type: ignore


def parse_hydrus_metadata_file(
    path_or_file: T.Union[str, os.PathLike, TextFileLike]
) -> collections.defaultdict[T.Optional[str], set[str]]:
    if isinstance(path_or_file, (str, os.PathLike)):
        with open(path_or_file, encoding=HYDRUS_METADATA_ENCODING) as file:
            return parse_hydrus_metadata(file.read())

    return parse_hydrus_metadata(path_or_file.read())


# Useful for splitting up requests to get_file_metadata()
def yield_chunks(sequence: T.Sequence[X], chunk_size: int, offset: int = 0) -> T.Generator[T.Sequence[X], None, None]:
    while offset < len(sequence):
        yield sequence[offset : offset + chunk_size]
        offset += chunk_size


__all__ = [
    "verify_permissions",
    "cli_request_api_key",
    "parse_hydrus_metadata",
    "parse_hydrus_metadata_file",
    "yield_chunks",
]
