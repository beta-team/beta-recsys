import asyncio
import os
from base64 import b64encode

import aiofiles
import aiohttp
import nest_asyncio
from requests import Session

nest_asyncio.apply()


class OneDrive:
    """Download shared file/folder to localhost with persisted structure.

    Download shared file/folder from OneDrive without authentication.

    params:
    `str:url`: url to the shared one drive folder or file
    `str:path`: local filesystem path

    methods:
    `download() -> None`: fire async download of all files found in URL
    """

    def __init__(self, url=None, path=None):
        """Init OneDrive Class."""
        if not (url and path):
            raise ValueError("URL to shared resource or path to download is missing.")

        self.url = url
        self.path = path
        self.prefix = "https://api.onedrive.com/v1.0/shares/"
        self.suffix = "/root?expand=children"
        self.session = Session()
        self.session.headers.update(
            {
                "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36"
                " (KHTML, like Gecko) Chrome/71.0.3578.98 Safari/537.36"
            }
        )

    def _token(self, url):
        """Decode share link."""
        result = "u!" + b64encode(url.encode()).decode()
        result = result.rstrip("=")
        result = result.replace("/", "_")
        result = result.replace("+", "-")
        return result

    def _traverse_url(self, url, name=""):
        """Traverse the folder tree and store leaf urls with filename.

        Get all the shared files given the url.

        Args:
            url: the shared link of files/folder
        """
        r = self.session.get(f"{self.prefix}{self._token(url)}{self.suffix}").json()
        name = name + os.sep + r["name"]

        # shared file
        if not r["children"]:
            file = {}
            file["name"] = name.lstrip(os.sep)
            file["url"] = r["@content.downloadUrl"]
            self.to_download.append(file)
            print(f"Found {file['name']}")

        # shared folder
        for child in r["children"]:
            if "folder" in child:
                self._traverse_url(child["webUrl"], name)

            if "file" in child:
                file = {}
                file["name"] = (name + os.sep + child["name"]).lstrip(os.sep)
                file["url"] = child["@content.downloadUrl"]
                self.to_download.append(file)
                print(f"Found {file['name']}")

    async def _download_file(self, file, session):
        async with session.get(file["url"], timeout=None) as r:
            filename = os.path.join(self.path, file["name"])
            os.makedirs(os.path.dirname(filename), exist_ok=True)
            async with aiofiles.open(filename, "wb") as f:
                async for chunk in r.content.iter_chunked(1024 * 16):
                    if chunk:
                        await f.write(chunk)

        self.downloaded += 1
        progress = int(self.downloaded / len(self.to_download) * 100)
        print(
            f"Download progress: {self.downloaded}/{len(self.to_download)}, {progress}%"
        )

    async def _downloader(self):
        async with aiohttp.ClientSession() as session:
            await asyncio.wait(
                [self._download_file(file, session) for file in self.to_download]
            )

    def download(self):
        """Download files from OneDrive.

        Download files from OneDrive with the given share link.
        """
        print("Traversing public folder\n")
        self.to_download = []
        self.downloaded = 0
        self._traverse_url(self.url)

        print("\nStarting async download\n")
        asyncio.get_event_loop().run_until_complete(self._downloader())
