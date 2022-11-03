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

import enum
import json
import os
import typing as T
from collections import abc

import requests

__version__ = "4.0.0"

DEFAULT_API_URL = "http://127.0.0.1:45869/"
HYDRUS_METADATA_ENCODING = "utf-8"
AUTHENTICATION_TIMEOUT_CODE = 419


class HydrusAPIException(Exception):
    pass


class ConnectionError(HydrusAPIException, requests.ConnectTimeout):
    pass


class APIError(HydrusAPIException):
    def __init__(self, response: requests.Response):
        super().__init__(response.text)
        self.response = response


class MissingParameter(APIError):
    pass


class InsufficientAccess(APIError):
    pass


class DatabaseLocked(APIError):
    pass


class ServerError(APIError):
    pass


# Customize IntEnum, so we can just do str(Enum.member) to get the string representation of its value unmodified,
# without users having to access .value explicitly
class StringableIntEnum(enum.IntEnum):
    def __str__(self):
        return str(self.value)


@enum.unique
class Permission(StringableIntEnum):
    IMPORT_URLS = 0
    IMPORT_FILES = 1
    ADD_TAGS = 2
    SEARCH_FILES = 3
    MANAGE_PAGES = 4
    MANAGE_COOKIES = 5
    MANAGE_DATABASE = 6
    ADD_NOTES = 7


@enum.unique
class URLType(StringableIntEnum):
    POST_URL = 0
    FILE_URL = 2
    GALLERY_URL = 3
    WATCHABLE_URL = 4
    UNKNOWN_URL = 5


@enum.unique
class ImportStatus(StringableIntEnum):
    IMPORTABLE = 0
    SUCCESS = 1
    EXISTS = 2
    PREVIOUSLY_DELETED = 3
    FAILED = 4
    VETOED = 7


@enum.unique
class TagAction(StringableIntEnum):
    ADD = 0
    DELETE = 1
    PEND = 2
    RESCIND_PENDING = 3
    PETITION = 4
    RESCIND_PETITION = 5


@enum.unique
class TagStatus(StringableIntEnum):
    CURRENT = 0
    PENDING = 1
    DELETED = 2
    PETITIONED = 3


@enum.unique
class PageType(StringableIntEnum):
    GALLERY_DOWNLOADER = 1
    SIMPLE_DOWNLOADER = 2
    HARD_DRIVE_IMPORT = 3
    PETITIONS = 5
    FILE_SEARCH = 6
    URL_DOWNLOADER = 7
    DUPLICATES = 8
    THREAD_WATCHER = 9
    PAGE_OF_PAGES = 10


@enum.unique
class FileSortType(StringableIntEnum):
    FILE_SIZE = 0
    DURATION = 1
    IMPORT_TIME = 2
    FILE_TYPE = 3
    RANDOM = 4
    WIDTH = 5
    HEIGHT = 6
    RATIO = 7
    NUMBER_OF_PIXELS = 8
    NUMBER_OF_TAGS = 9
    NUMBER_OF_MEDIA_VIEWS = 10
    TOTAL_MEDIA_VIEWTIME = 11
    APPROXIMATE_BITRATE = 12
    HAS_AUDIO = 13
    MODIFIED_TIME = 14
    FRAMERATE = 15
    NUMBER_OF_FRAMES = 16


class BinaryFileLike(T.Protocol):
    def read(self):
        ...


# The client should accept all objects that either support the iterable or mapping protocol. We must ensure that objects
# are either lists or dicts, so Python's json module can handle them
class JSONEncoder(json.JSONEncoder):
    def default(self, object_: T.Any):
        if isinstance(object_, abc.Mapping):
            return dict(object_)
        if isinstance(object_, abc.Iterable):
            return list(object_)
        return super().default(object_)


class Client:
    VERSION = 31

    # Access Management
    _GET_API_VERSION_PATH = "/api_version"
    _REQUEST_NEW_PERMISSIONS_PATH = "/request_new_permissions"
    _GET_SESSION_KEY_PATH = "/session_key"
    _VERIFY_ACCESS_KEY_PATH = "/verify_access_key"
    _GET_SERVICES_PATH = "/get_services"

    # Adding Files
    _ADD_FILE_PATH = "/add_files/add_file"
    _DELETE_FILES_PATH = "/add_files/delete_files"
    _UNDELETE_FILES_PATH = "/add_files/undelete_files"
    _ARCHIVE_FILES_PATH = "/add_files/archive_files"
    _UNARCHIVE_FILES_PATH = "/add_files/unarchive_files"

    # Adding Tags
    _CLEAN_TAGS_PATH = "/add_tags/clean_tags"
    _SEARCH_TAGS_PATH = "/add_tags/search_tags"
    _ADD_TAGS_PATH = "/add_tags/add_tags"

    # Adding URLs
    _GET_URL_FILES_PATH = "/add_urls/get_url_files"
    _GET_URL_INFO_PATH = "/add_urls/get_url_info"
    _ADD_URL_PATH = "/add_urls/add_url"
    _ASSOCIATE_URL_PATH = "/add_urls/associate_url"

    # Adding Notes
    _SET_NOTES_PATH = "/add_notes/set_notes"
    _DELETE_NOTES_PATH = "/add_notes/delete_notes"

    # Managing Cookies and HTTP Headers
    _GET_COOKIES_PATH = "/manage_cookies/get_cookies"
    _SET_COOKIES_PATH = "/manage_cookies/set_cookies"
    _SET_USER_AGENT_PATH = "/manage_headers/set_user_agent"

    # Managing Pages
    _GET_PAGES_PATH = "/manage_pages/get_pages"
    _GET_PAGE_INFO_PATH = "/manage_pages/get_page_info"
    _ADD_FILES_TO_PAGE_PATH = "/manage_pages/add_files"
    _FOCUS_PAGE_PATH = "/manage_pages/focus_page"

    # Searching and Fetching Files
    _SEARCH_FILES_PATH = "/get_files/search_files"
    _GET_FILE_METADATA_PATH = "/get_files/file_metadata"
    _GET_FILE_PATH = "/get_files/file"
    _GET_THUMBNAIL_PATH = "/get_files/thumbnail"

    # Managing the Database
    _LOCK_DATABASE_PATH = "/manage_database/lock_on"
    _UNLOCK_DATABASE_PATH = "/manage_database/lock_off"
    _MR_BONES_PATH = "/manage_database/mr_bones"

    def __init__(
        self,
        access_key = None,
        api_url: str = DEFAULT_API_URL,
        session = None,
    ):
        """
        See https://hydrusnetwork.github.io/hydrus/help/client_api.html for documentation.
        """

        self.access_key = access_key
        self.api_url = api_url.rstrip("/")
        self.session = session or requests.Session()

    def _api_request(self, method: str, path: str, **kwargs: T.Any):
        if self.access_key is not None:
            kwargs.setdefault("headers", {}).update({"Hydrus-Client-API-Access-Key": self.access_key})

        # Make sure we use our custom JSONEncoder that can serialize all objects that implement the iterable or mapping
        # protocol
        json_data = kwargs.pop("json", None)
        if json_data is not None:
            kwargs["data"] = json.dumps(json_data, cls=JSONEncoder)
            # Since we aren't using the json keyword-argument, we have to set the Content-Type manually
            kwargs["headers"]["Content-Type"] = "application/json"

        try:
            response = self.session.request(method, self.api_url + path, **kwargs)
        except requests.RequestException as error:
            # Re-raise connection and timeout errors as hydrus.ConnectionErrors so these are more easy to handle for
            # client applications
            raise ConnectionError(*error.args)

        try:
            response.raise_for_status()
        except requests.HTTPError:
            if response.status_code == requests.codes.bad_request:
                raise MissingParameter(response)
            elif response.status_code in {
                requests.codes.unauthorized,
                requests.codes.forbidden,
                AUTHENTICATION_TIMEOUT_CODE,
            }:
                raise InsufficientAccess(response)
            elif response.status_code == requests.codes.service_unavailable:
                raise DatabaseLocked(response)
            elif response.status_code == requests.codes.server_error:
                raise ServerError(response)
            raise APIError(response)

        return response

    def get_api_version(self):
        response = self._api_request("GET", self._GET_API_VERSION_PATH)
        return response.json()

    def request_new_permissions(self, name, permissions):
        response = self._api_request(
            "GET",
            self._REQUEST_NEW_PERMISSIONS_PATH,
            params={"name": name, "basic_permissions": json.dumps(permissions, cls=JSONEncoder)},
        )
        return response.json()["access_key"]

    def get_session_key(self):
        response = self._api_request("GET", self._GET_SESSION_KEY_PATH)
        return response.json()["session_key"]

    def verify_access_key(self):
        response = self._api_request("GET", self._VERIFY_ACCESS_KEY_PATH)
        return response.json()

    def get_services(self):
        response = self._api_request("GET", self._GET_SERVICES_PATH)
        return response.json()

    def add_file(self, path_or_file: T.Union[str, os.PathLike, BinaryFileLike]):
        if isinstance(path_or_file, (str, os.PathLike)):
            response = self._api_request("POST", self._ADD_FILE_PATH, json={"path": os.fspath(path_or_file)})
        else:
            response = self._api_request(
                "POST",
                self._ADD_FILE_PATH,
                data=path_or_file.read(),
                headers={"Content-Type": "application/octet-stream"},
            )

        return response.json()

    def delete_files(
        self,
        hashes = None,
        file_ids  = None,
        file_service_name = None,
        file_service_key = None,
        reason = None
    ):
        if hashes is None and file_ids is None:
            raise ValueError("At least one of hashes, file_ids is required")
        if file_service_name is not None and file_service_key is not None:
            raise ValueError("Exactly one of file_service_name, file_service_key is required")

        payload: dict[str, T.Any] = {}
        if hashes is not None:
            payload["hashes"] = hashes
        if file_ids is not None:
            payload["file_ids"] = file_ids
        if file_service_name is not None:
            payload["file_service_name"] = file_service_name
        if file_service_key is not None:
            payload["file_service_key"] = file_service_key
        if reason is not None:
            payload["reason"] = reason

        self._api_request("POST", self._DELETE_FILES_PATH, json=payload)

    def undelete_files(
        self,
        hashes = None,
        file_ids = None,
        file_service_name = None,
        file_service_key = None,
    ):
        if hashes is None and file_ids is None:
            raise ValueError("At least one of hashes, file_ids is required")
        if file_service_name is not None and file_service_key is not None:
            raise ValueError("Exactly one of file_service_name, file_service_key is required")

        payload: dict[str, T.Any] = {}
        if hashes is not None:
            payload["hashes"] = hashes
        if file_ids is not None:
            payload["file_ids"] = file_ids
        if file_service_name is not None:
            payload["file_service_name"] = file_service_name
        if file_service_key is not None:
            payload["file_service_key"] = file_service_key

        self._api_request("POST", self._UNDELETE_FILES_PATH, json=payload)

    def archive_files(
        self,
        hashes = None,
        file_ids = None
    ):
        if hashes is None and file_ids is None:
            raise ValueError("At least one of hashes, file_ids is required")

        payload: dict[str, T.Any] = {}
        if hashes is not None:
            payload["hashes"] = hashes
        if file_ids is not None:
            payload["file_ids"] = file_ids

        self._api_request("POST", self._ARCHIVE_FILES_PATH, json=payload)

    def unarchive_files(
        self,
        hashes = None,
        file_ids = None
    ):
        if hashes is None and file_ids is None:
            raise ValueError("At least one of hashes, file_ids is required")

        payload: dict[str, T.Any] = {}
        if hashes is not None:
            payload["hashes"] = hashes
        if file_ids is not None:
            payload["file_ids"] = file_ids

        self._api_request("POST", self._UNARCHIVE_FILES_PATH, json=payload)

    def clean_tags(self, tags ):
        response = self._api_request("GET", self._CLEAN_TAGS_PATH, params={"tags": json.dumps(tags, cls=JSONEncoder)})
        return response.json()["tags"]

    def search_tags(
        self,
        search: str,
        tag_service_key = None,
        tag_service_name = None
    ):
        if tag_service_name is not None and tag_service_key is not None:
            raise ValueError("Exactly one of tag_service_name, tag_service_key is required")

        payload: dict[str, T.Any] = {"search": search}
        if tag_service_key is not None:
            payload["tag_service_key"] = tag_service_key
        if tag_service_name is not None:
            payload["tag_service_name"] = tag_service_name

        response = self._api_request("GET", self._SEARCH_TAGS_PATH, params=payload)
        return response.json()["tags"]

    def add_tags(
        self,
        hashes = None,
        file_ids = None,
        service_names_to_tags = None,
        service_keys_to_tags = None,
        service_names_to_actions_to_tags = None,
        service_keys_to_actions_to_tags = None,
    ):
        if hashes is None and file_ids is None:
            raise ValueError("At least one of hashes, file_ids is required")
        if (
            service_names_to_tags is None
            and service_keys_to_tags is None
            and service_names_to_actions_to_tags is None
            and service_keys_to_actions_to_tags is None
        ):
            raise ValueError(
                "At least one of service_names_to_tags, service_keys_to_tags, service_names_to_actions_to_tags or "
                "service_keys_to_actions_to_tags is required"
            )

        payload: dict[str, T.Any] = {}
        if hashes is not None:
            payload["hashes"] = hashes
        if file_ids is not None:
            payload["file_ids"] = file_ids
        if service_names_to_tags is not None:
            payload["service_names_to_tags"] = service_names_to_tags
        if service_keys_to_tags is not None:
            payload["service_keys_to_tags"] = service_keys_to_tags
        if service_names_to_actions_to_tags is not None:
            payload["service_names_to_actions_to_tags"] = service_names_to_actions_to_tags
        if service_keys_to_actions_to_tags is not None:
            payload["service_keys_to_actions_to_tags"] = service_keys_to_actions_to_tags

        self._api_request("POST", self._ADD_TAGS_PATH, json=payload)

    def get_url_files(self, url: str):
        response = self._api_request("GET", self._GET_URL_FILES_PATH, params={"url": url})
        return response.json()

    def get_url_info(self, url: str):
        response = self._api_request("GET", self._GET_URL_INFO_PATH, params={"url": url})
        return response.json()

    def add_url(
        self,
        url: str,
        destination_page_key = None,
        destination_page_name = None,
        show_destination_page = None,
        service_names_to_additional_tags = None,
        service_keys_to_additional_tags = None,
        filterable_tags = None,
    ):
        if destination_page_key is not None and destination_page_name is not None:
            raise ValueError("Exactly one of destination_page_key, destination_page_name is required")

        payload: dict[str, T.Any] = {"url": url}
        if destination_page_key is not None:
            payload["destination_page_key"] = destination_page_key
        if destination_page_name is not None:
            payload["destination_page_name"] = destination_page_name
        if show_destination_page is not None:
            payload["show_destination_page"] = show_destination_page
        if service_names_to_additional_tags is not None:
            payload["service_names_to_additional_tags"] = service_names_to_additional_tags
        if service_keys_to_additional_tags is not None:
            payload["service_keys_to_additional_tags"] = service_keys_to_additional_tags
        if filterable_tags is not None:
            payload["filterable_tags"] = filterable_tags

        response = self._api_request("POST", self._ADD_URL_PATH, json=payload)
        return response.json()

    def associate_url(
        self,
        hashes = None,
        file_ids = None,
        urls_to_add = None,
        urls_to_delete = None,
    ):
        if hashes is None and file_ids is None:
            raise ValueError("At least one of hashes, file_ids is required")
        if urls_to_add is None and urls_to_delete is None:
            raise ValueError("At least one of urls_to_add, urls_to_delete is required")

        payload: dict[str, T.Any] = {}
        if hashes is not None:
            payload["hashes"] = hashes
        if file_ids is not None:
            payload["file_ids"] = file_ids
        if urls_to_add is not None:
            urls_to_add = urls_to_add
            payload["urls_to_add"] = urls_to_add
        if urls_to_delete is not None:
            urls_to_delete = urls_to_delete
            payload["urls_to_delete"] = urls_to_delete

        self._api_request("POST", self._ASSOCIATE_URL_PATH, json=payload)

    def set_notes(self, notes , hash_= None, file_id = None):
        if (hash_ is None and file_id is None) or (hash_ is not None and file_id is not None):
            raise ValueError("Exactly one of hash_, file_id is required")

        payload: dict[str, T.Any] = {"notes": notes}
        if hash_ is not None:
            payload["hash"] = hash_
        if file_id is not None:
            payload["file_id"] = file_id

        self._api_request("POST", self._SET_NOTES_PATH, json=payload)

    def delete_notes(
        self,
        note_names ,
        hash_ = None,
        file_id = None
    ):
        if (hash_ is None and file_id is None) or (hash_ is not None and file_id is not None):
            raise ValueError("Exactly one of hash_, file_id is required")

        payload: dict[str, T.Any] = {"note_names": note_names}
        if hash_ is not None:
            payload["hash"] = hash_
        if file_id is not None:
            payload["file_id"] = file_id

        self._api_request("POST", self._DELETE_NOTES_PATH, json=payload)

    def get_cookies(self, domain: str):
        response = self._api_request("GET", self._GET_COOKIES_PATH, params={"domain": domain})
        return response.json()["cookies"]

    def set_cookies(self, cookies ):
        self._api_request("POST", self._SET_COOKIES_PATH, json={"cookies": cookies})

    def set_user_agent(self, user_agent: str):
        self._api_request("POST", self._SET_USER_AGENT_PATH, json={"user-agent": user_agent})

    def get_pages(self):
        response = self._api_request("GET", self._GET_PAGES_PATH)
        return response.json()["pages"]

    def get_page_info(self, page_key: str, simple = None):
        parameters = {"page_key": page_key}
        if simple is not None:
            parameters["simple"] = json.dumps(simple, cls=JSONEncoder)

        response = self._api_request("GET", self._GET_PAGE_INFO_PATH, params=parameters)
        return response.json()["page_info"]

    def add_files_to_page(
        self,
        page_key: str,
        file_ids = None,
        hashes = None
    ):
        if file_ids is None and hashes is None:
            raise ValueError("At least one of file_ids, hashes is required")

        payload: dict[str, T.Any] = {"page_key": page_key}
        if file_ids is not None:
            payload["file_ids"] = file_ids
        if hashes is not None:
            payload["hashes"] = hashes

        self._api_request("POST", self._ADD_FILES_TO_PAGE_PATH, json=payload)

    def focus_page(self, page_key: str):
        self._api_request("POST", self._FOCUS_PAGE_PATH, json={"page_key": page_key})

    def search_files(
        self,
        tags,
        file_service_name = None,
        file_service_key = None,
        tag_service_name = None,
        tag_service_key = None,
        file_sort_type = None,
        file_sort_asc = None,
        return_hashes = None,
    ):
        if file_service_name is not None and file_service_key is not None:
            raise ValueError("Exactly one of file_service_name, file_service_key is required")
        if tag_service_name is not None and tag_service_key is not None:
            raise ValueError("Exactly one of tag_service_name, tag_service_key is required")

        parameters: dict[str, T.Union[str, int]] = {"tags": json.dumps(tags, cls=JSONEncoder)}
        if file_service_name is not None:
            parameters["file_service_name"] = file_service_name
        if file_service_key is not None:
            parameters["file_service_key"] = file_service_key

        if tag_service_name is not None:
            parameters["tag_service_name"] = tag_service_name
        if tag_service_key is not None:
            parameters["tag_service_key"] = tag_service_key

        if file_sort_type is not None:
            parameters["file_sort_type"] = file_sort_type
        if file_sort_asc is not None:
            parameters["file_sort_asc"] = json.dumps(file_sort_asc, cls=JSONEncoder)
        if return_hashes is not None:
            parameters["return_hashes"] = json.dumps(return_hashes, cls=JSONEncoder)

        response = self._api_request("GET", self._SEARCH_FILES_PATH, params=parameters)
        return response.json()["hashes" if return_hashes else "file_ids"]

    def get_file_metadata(
        self,
        hashes = None,
        file_ids = None,
        create_new_file_ids = None,
        only_return_identifiers = None,
        only_return_basic_information = None,
        detailed_url_information = None,
        hide_service_name_tags = None,
        include_notes = None
    ):
        if hashes is None and file_ids is None:
            raise ValueError("At least one of hashes, file_ids is required")

        parameters = {}
        if hashes is not None:
            parameters["hashes"] = json.dumps(hashes, cls=JSONEncoder)
        if file_ids is not None:
            parameters["file_ids"] = json.dumps(file_ids, cls=JSONEncoder)

        if create_new_file_ids is not None:
            parameters["create_new_file_ids"] = json.dumps(create_new_file_ids, cls=JSONEncoder)
        if only_return_identifiers is not None:
            parameters["only_return_identifiers"] = json.dumps(only_return_identifiers, cls=JSONEncoder)
        if only_return_basic_information is not None:
            parameters["only_return_basic_information"] = json.dumps(only_return_basic_information, cls=JSONEncoder)
        if detailed_url_information is not None:
            parameters["detailed_url_information"] = json.dumps(detailed_url_information, cls=JSONEncoder)
        if hide_service_name_tags is not None:
            parameters["hide_service_name_tags"] = json.dumps(hide_service_name_tags, cls=JSONEncoder)
        if include_notes is not None:
            parameters["include_notes"] = json.dumps(include_notes, cls=JSONEncoder)

        response = self._api_request("GET", self._GET_FILE_METADATA_PATH, params=parameters)
        return response.json()["metadata"]

    def get_file(self, hash_ = None, file_id = None):
        if (hash_ is None and file_id is None) or (hash_ is not None and file_id is not None):
            raise ValueError("Exactly one of hash_, file_id is required")

        parameters: dict[str, T.Union[str, int]] = {}
        if hash_ is not None:
            parameters["hash"] = hash_
        if file_id is not None:
            parameters["file_id"] = file_id

        return self._api_request("GET", self._GET_FILE_PATH, params=parameters, stream=True)

    def get_thumbnail(self, hash_ = None, file_id = None):
        if (hash_ is None and file_id is None) or (hash_ is not None and file_id is not None):
            raise ValueError("Exactly one of hash_, file_id is required")

        parameters: dict[str, T.Union[str, int]] = {}
        if hash_ is not None:
            parameters["hash"] = hash_
        if file_id is not None:
            parameters["file_id"] = file_id

        return self._api_request("GET", self._GET_THUMBNAIL_PATH, params=parameters, stream=True)

    def lock_database(self):
        self._api_request("POST", self._LOCK_DATABASE_PATH)

    def unlock_database(self):
        self._api_request("POST", self._UNLOCK_DATABASE_PATH)

    def get_mr_bones(self):
        return self._api_request("GET", self._MR_BONES_PATH).json()["boned_stats"]

    def add_and_tag_files(
        self,
        paths_or_files,
        tags ,
        service_names = None,
        service_keys = None,
    ):
        """Convenience method to add and tag multiple files at the same time.

        If service_names and service_keys aren't specified, the default service name "my tags" will be used. If a file
        already exists in Hydrus, it will also be tagged.

        Returns:
            list[dict[str, T.Any]]: Returns results of all `Client.add_file()` calls, matching the order of the
            paths_or_files iterable
        """
        if service_names is None and service_keys is None:
            service_names = ("my tags",)

        results = []
        hashes = set()
        for path_or_file in paths_or_files:
            result = self.add_file(path_or_file)
            results.append(result)
            if result["status"] != ImportStatus.FAILED:
                hashes.add(result["hash"])

        service_names_to_tags = {name: tags for name in service_names} if service_names is not None else None
        service_keys_to_tags = {key: tags for key in service_keys} if service_keys is not None else None
        # Ignore type, we know that hashes only contains strings
        self.add_tags(hashes, service_names_to_tags=service_names_to_tags, service_keys_to_tags=service_keys_to_tags)  # type: ignore
        return results

    def get_page_list(self):
        """Convenience method that returns a flattened version of the page tree from `Client.get_pages()`.

        Returns:
            list[dict[str, T.Any]]: A list of every "pages" value in the page tree in pre-order (NLR)
        """
        tree = self.get_pages()
        pages = []

        def walk_tree(page: dict[str, T.Any]):
            pages.append(page)
            # Ignore type, we know that pages is always a list
            for sub_page in page.get("pages", ()):  # type: ignore
                # Ignore type, we know that sub_page is always a dict
                walk_tree(sub_page)  # type: ignore

        walk_tree(tree)
        return pages


__all__ = [
    "__version__",
    "DEFAULT_API_URL",
    "HYDRUS_METADATA_ENCODING",
    "HydrusAPIException",
    "ConnectionError",
    "APIError",
    "MissingParameter",
    "InsufficientAccess",
    "DatabaseLocked",
    "ServerError",
    "Permission",
    "URLType",
    "ImportStatus",
    "TagAction",
    "TagStatus",
    "PageType",
    "FileSortType",
    "Client",
]
