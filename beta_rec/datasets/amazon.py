import os

import pandas as pd

from ..datasets.dataset_base import DatasetBase
from ..utils.common_util import get_data_frame_from_gzip_file
from ..utils.constants import (
    DEFAULT_ITEM_COL,
    DEFAULT_RATING_COL,
    DEFAULT_TIMESTAMP_COL,
    DEFAULT_USER_COL,
)

# Download URL.
AMAZON_Amazon_Instant_Video_URL = (
    "http://snap.stanford.edu/data/amazon/productGraph/categoryFiles"
    "/reviews_Amazon_Instant_Video.json.gz"
)
AMAZON_Musical_Instruments_URL = (
    "http://snap.stanford.edu/data/amazon/productGraph/categoryFiles"
    "/reviews_Musical_Instruments.json.gz"
)
AMAZON_Digital_Music_URL = (
    "http://snap.stanford.edu/data/amazon/productGraph/categoryFiles"
    "/reviews_Digital_Music.json.gz"
)
AMAZON_Baby_URL = (
    "http://snap.stanford.edu/data/amazon/productGraph/categoryFiles"
    "/reviews_Baby.json.gz"
)
AMAZON_Patio_Lawn_Garden_URL = (
    "http://snap.stanford.edu/data/amazon/productGraph/categoryFiles"
    "/reviews_Patio_Lawn_and_Garden.json.gz"
)
AMAZON_Grocery_Gourmet_Food_URL = (
    "http://snap.stanford.edu/data/amazon/productGraph/categoryFiles"
    "/reviews_Grocery_and_Gourmet_Food.json.gz"
)
AMAZON_Automotive_URL = (
    "http://snap.stanford.edu/data/amazon/productGraph/categoryFiles"
    "/reviews_Automotive.json.gz"
)
AMAZON_Pet_Supplies_URL = (
    "http://snap.stanford.edu/data/amazon/productGraph/categoryFiles"
    "/reviews_Pet_Supplies.json.gz"
)
AMAZON_Cell_Phones_and_Accessories_URL = (
    "http://snap.stanford.edu/data/amazon/productGraph/categoryFiles"
    "/reviews_Cell_Phones_and_Accessories.json.gz"
)
AMAZON_Health_and_Personal_Care_URL = (
    "http://snap.stanford.edu/data/amazon/productGraph/categoryFiles"
    "/reviews_Health_and_Personal_Care.json.gz"
)
AMAZON_Toys_and_Games_URL = (
    "http://snap.stanford.edu/data/amazon/productGraph/categoryFiles"
    "/reviews_Toys_and_Games.json.gz"
)
AMAZON_Video_Games_URL = (
    "http://snap.stanford.edu/data/amazon/productGraph/categoryFiles"
    "/reviews_Video_Games.json.gz"
)
AMAZON_Tools_and_Home_Improvement_URL = (
    "http://snap.stanford.edu/data/amazon/productGraph/categoryFiles"
    "/reviews_Tools_and_Home_Improvement.json.gz"
)
AMAZON_Beauty_URL = (
    "http://snap.stanford.edu/data/amazon/productGraph/categoryFiles"
    "/reviews_Beauty.json.gz"
)
AMAZON_Apps_for_Android_URL = (
    "http://snap.stanford.edu/data/amazon/productGraph/categoryFiles"
    "/reviews_Apps_for_Android.json.gz"
)
AMAZON_Office_Products_URL = (
    "http://snap.stanford.edu/data/amazon/productGraph/categoryFiles"
    "/reviews_Office_Products.json.gz"
)
AMAZON_Books_URL = (
    "http://snap.stanford.edu/data/amazon/productGraph/categoryFiles"
    "/reviews_Books.json.gz"
)
AMAZON_Electronics_URL = (
    "http://snap.stanford.edu/data/amazon/productGraph/categoryFiles"
    "/reviews_Electronics.json.gz"
)
AMAZON_Movies_and_TV_URL = (
    "http://snap.stanford.edu/data/amazon/productGraph/categoryFiles"
    "/reviews_Movies_and_TV.json.gz"
)
AMAZON_CDs_and_Vinyl_URL = (
    "http://snap.stanford.edu/data/amazon/productGraph/categoryFiles"
    "/reviews_CDs_and_Vinyl.json.gz"
)
AMAZON_Clothing_Shoes_and_Jewelry_URL = (
    "http://snap.stanford.edu/data/amazon/productGraph/categoryFiles"
    "/reviews_Clothing_Shoes_and_Jewelry.json.gz"
)
AMAZON_Home_and_Kitchen_URL = (
    "http://snap.stanford.edu/data/amazon/productGraph/categoryFiles"
    "/reviews_Home_and_Kitchen.json.gz"
)
AMAZON_Kindle_Store_URL = (
    "http://snap.stanford.edu/data/amazon/productGraph/categoryFiles"
    "/reviews_Kindle_Store.json.gz"
)
AMAZON_Sports_and_Outdoors_URL = (
    "http://snap.stanford.edu/data/amazon/productGraph/categoryFiles"
    "/reviews_Sports_and_Outdoors.json.gz"
)


class AmazonDataset(DatasetBase):
    r"""AmazonDataset.

    Amazon base dataset.
    """

    def __init__(self, dataset_name, min_u_c=0, min_i_c=3, url=None, root_dir=None):
        r"""Init AmazonDataset Class."""
        super().__init__(
            dataset_name=dataset_name,
            min_u_c=min_u_c,
            min_i_c=min_i_c,
            root_dir=root_dir,
            url=url,
        )

    def preprocess(self):
        """Preprocess the raw file.

        Preprocess the file downloaded via the url, convert it to a dataframe consist of the user-item interaction,
        and save in the processed directory.
        """
        file_name = os.path.join(self.raw_path, f"{self.dataset_name}.json.gz")
        print(f"file_name: {file_name}")
        if not os.path.exists(file_name):
            self.download()

        # parse json data
        data = get_data_frame_from_gzip_file(file_name)

        # rename columns
        data = data.rename(
            columns={
                "reviewerID": DEFAULT_USER_COL,
                "asin": DEFAULT_ITEM_COL,
                "overall": DEFAULT_RATING_COL,
                "unixReviewTime": DEFAULT_TIMESTAMP_COL,
            }
        )

        # select necessary columns
        data = pd.DataFrame(
            data,
            columns=[
                DEFAULT_USER_COL,
                DEFAULT_ITEM_COL,
                DEFAULT_RATING_COL,
                DEFAULT_TIMESTAMP_COL,
            ],
        )

        self.save_dataframe_as_npz(
            data,
            os.path.join(self.processed_path, f"{self.dataset_name}_interaction.npz"),
        )


class AmazonInstantVideo(AmazonDataset):
    r"""AmazonInstantVideo.

    Amazon Review dataset.
    """

    def __init__(self, min_u_c=0, min_i_c=3, root_dir=None):
        r"""Init AmazonInstantVideo Class."""
        super().__init__(
            dataset_name="amazon-amazon-instant-video",
            min_u_c=min_u_c,
            min_i_c=min_i_c,
            root_dir=root_dir,
            url=AMAZON_Amazon_Instant_Video_URL,
        )


class AmazonMusicalInstruments(AmazonDataset):
    r"""AmazonMusicalInstruments.

    Amazon Review dataset.
    """

    def __init__(self, min_u_c=0, min_i_c=3, root_dir=None):
        r"""Init AmazonMusicalInstruments Class."""
        super().__init__(
            dataset_name="amazon-musical-instruments",
            min_u_c=min_u_c,
            min_i_c=min_i_c,
            root_dir=root_dir,
            url=AMAZON_Musical_Instruments_URL,
        )


class AmazonDigitalMusic(AmazonDataset):
    r"""AmazonDigitalMusic.

    Amazon Review dataset.
    """

    def __init__(self, min_u_c=0, min_i_c=3, root_dir=None):
        r"""Init AmazonDigitalMusic Class."""
        super().__init__(
            dataset_name="amazon-digital-music",
            min_u_c=min_u_c,
            min_i_c=min_i_c,
            root_dir=root_dir,
            url=AMAZON_Digital_Music_URL,
        )


class AmazonBaby(AmazonDataset):
    r"""AmazonBaby.

    Amazon Review dataset.
    """

    def __init__(self, min_u_c=0, min_i_c=3, root_dir=None):
        r"""Init AmazonBaby Class."""
        super().__init__(
            dataset_name="amazon-baby",
            min_u_c=min_u_c,
            min_i_c=min_i_c,
            root_dir=root_dir,
            url=AMAZON_Baby_URL,
        )


class AmazonPatioLawnGarden(AmazonDataset):
    r"""AmazonPatioLawnGarden.

    Amazon Review dataset.
    """

    def __init__(self, min_u_c=0, min_i_c=3, root_dir=None):
        r"""Init AmazonPatioLawnGarden Class."""
        super().__init__(
            dataset_name="amazon-patio-lawn-garden",
            min_u_c=min_u_c,
            min_i_c=min_i_c,
            root_dir=root_dir,
            url=AMAZON_Patio_Lawn_Garden_URL,
        )


class AmazonGroceryGourmetFood(AmazonDataset):
    r"""AmazonGroceryGourmetFood.

    Amazon Review dataset.
    """

    def __init__(self, min_u_c=0, min_i_c=3, root_dir=None):
        r"""Init AmazonGroceryGourmetFood Class."""
        super().__init__(
            dataset_name="amazon-grocery-gourmet-food",
            min_u_c=min_u_c,
            min_i_c=min_i_c,
            root_dir=root_dir,
            url=AMAZON_Grocery_Gourmet_Food_URL,
        )


class AmazonAutomotive(AmazonDataset):
    r"""AmazonAutomotive.

    Amazon Review dataset.
    """

    def __init__(self, min_u_c=0, min_i_c=3, root_dir=None):
        r"""Init AmazonAutomotive Class."""
        super().__init__(
            dataset_name="amazon-automotive",
            min_u_c=min_u_c,
            min_i_c=min_i_c,
            root_dir=root_dir,
            url=AMAZON_Automotive_URL,
        )


class AmazonPetSupplies(AmazonDataset):
    r"""AmazonPetSupplies.

    Amazon Review dataset.
    """

    def __init__(self, min_u_c=0, min_i_c=3, root_dir=None):
        r"""Init AmazonPetSupplies Class."""
        super().__init__(
            dataset_name="amazon-pet-suppplies",
            min_u_c=min_u_c,
            min_i_c=min_i_c,
            root_dir=root_dir,
            url=AMAZON_Pet_Supplies_URL,
        )


class AmazonCellPhonesAndAccessories(AmazonDataset):
    r"""AmazonCellPhonesAndAccessories.

    Amazon Review dataset.
    """

    def __init__(self, min_u_c=0, min_i_c=3, root_dir=None):
        r"""Init AmazonPetSupplies Class."""
        super().__init__(
            dataset_name="amazon-cell-phones-and-accessories",
            min_u_c=min_u_c,
            min_i_c=min_i_c,
            root_dir=root_dir,
            url=AMAZON_Cell_Phones_and_Accessories_URL,
        )


class AmazonHealthAndPersonalCare(AmazonDataset):
    r"""AmazonHealthAndPersonalCare.

    Amazon Review dataset.
    """

    def __init__(self, min_u_c=0, min_i_c=3, root_dir=None):
        r"""Init AmazonHealthAndPersonalCare Class."""
        super().__init__(
            dataset_name="amazon-health-and-personal-care",
            min_u_c=min_u_c,
            min_i_c=min_i_c,
            root_dir=root_dir,
            url=AMAZON_Health_and_Personal_Care_URL,
        )


class AmazonToysAndGames(AmazonDataset):
    r"""AmazonToysAndGames.

    Amazon Review dataset.
    """

    def __init__(self, min_u_c=0, min_i_c=3, root_dir=None):
        r"""Init AmazonToysAndGames Class."""
        super().__init__(
            dataset_name="amazon-toys-and-games",
            min_u_c=min_u_c,
            min_i_c=min_i_c,
            root_dir=root_dir,
            url=AMAZON_Toys_and_Games_URL,
        )


class AmazonVideoGames(AmazonDataset):
    r"""AmazonVideoGames.

    Amazon Review dataset.
    """

    def __init__(self, min_u_c=0, min_i_c=3, root_dir=None):
        r"""Init AmazonVideoGames Class."""
        super().__init__(
            dataset_name="amazon-video-games",
            min_u_c=min_u_c,
            min_i_c=min_i_c,
            root_dir=root_dir,
            url=AMAZON_Video_Games_URL,
        )


class AmazonToolsAndHomeImprovement(AmazonDataset):
    r"""AmazonToolsAndHomeImprovement.

    Amazon Review dataset.
    """

    def __init__(self, root_dir=None):
        r"""Init AmazonToolsAndHomeImprovement Class."""
        super().__init__(
            dataset_name="amazon-tools-and-home-improvement",
            root_dir=root_dir,
            url=AMAZON_Tools_and_Home_Improvement_URL,
        )


class AmazonBeauty(AmazonDataset):
    r"""AmazonBeauty.

    Amazon Review dataset.
    """

    def __init__(self, root_dir=None):
        r"""Init AmazonBeauty Class."""
        super().__init__(
            dataset_name="amazon-beauty",
            root_dir=root_dir,
            url=AMAZON_Beauty_URL,
        )


class AmazonAppsForAndroid(AmazonDataset):
    r"""AmazonAppsForAndroid.

    Amazon Review dataset.
    """

    def __init__(self, root_dir=None):
        r"""Init AmazonAppsForAndroid Class."""
        super().__init__(
            dataset_name="amazon-apps-for-android",
            root_dir=root_dir,
            url=AMAZON_Apps_for_Android_URL,
        )


class AmazonOfficeProducts(AmazonDataset):
    r"""AmazonOfficeProducts.

    Amazon Review dataset.
    """

    def __init__(self, root_dir=None):
        r"""Init AmazonOfficeProducts Class."""
        super().__init__(
            dataset_name="amazon-office-products",
            root_dir=root_dir,
            url=AMAZON_Office_Products_URL,
        )


class AmazonBooks(AmazonDataset):
    r"""AmazonBooks.

    Amazon Review dataset.
    """

    def __init__(self, root_dir=None):
        r"""Init AmazonBooks Class."""
        super().__init__(
            dataset_name="amazon-books",
            root_dir=root_dir,
            url=AMAZON_Books_URL,
        )


class AmazonElectronics(AmazonDataset):
    r"""AmazonElectronics.

    Amazon Review dataset.
    """

    def __init__(self, root_dir=None):
        r"""Init AmazonElectronics Class."""
        super().__init__(
            dataset_name="amazon-electronics",
            root_dir=root_dir,
            url=AMAZON_Electronics_URL,
        )


class AmazonMoviesAndTV(AmazonDataset):
    r"""AmazonMoviesAndTV.

    Amazon Review dataset.
    """

    def __init__(self, root_dir=None):
        r"""Init AmazonMoviesAndTV Class."""
        super().__init__(
            dataset_name="amazon-movies-and-tv",
            root_dir=root_dir,
            url=AMAZON_Movies_and_TV_URL,
        )


class AmazonCDsAndVinyl(AmazonDataset):
    r"""AmazonCDsAndVinyl.

    Amazon Review dataset.
    """

    def __init__(self, root_dir=None):
        r"""Init AmazonCDsAndVinyl Class."""
        super().__init__(
            dataset_name="amazon-cds-and-vinyl",
            root_dir=root_dir,
            url=AMAZON_CDs_and_Vinyl_URL,
        )


class AmazonClothingShoesAndJewelry(AmazonDataset):
    r"""AmazonClothingShoesAndJewelry.

    Amazon Review dataset.
    """

    def __init__(self, root_dir=None):
        r"""Init AmazonClothingShoesAndJewelry Class."""
        super().__init__(
            dataset_name="amazon-clothing_shoes_and_jewelry",
            root_dir=root_dir,
            url=AMAZON_Clothing_Shoes_and_Jewelry_URL,
        )


class AmazonHomeAndKitchen(AmazonDataset):
    r"""AmazonHomeAndKitchen.

    Amazon Review dataset.
    """

    def __init__(self, root_dir=None):
        r"""Init AmazonHomeAndKitchen Class."""
        super().__init__(
            dataset_name="amazon-home-and-kitchen",
            root_dir=root_dir,
            url=AMAZON_Home_and_Kitchen_URL,
        )


class AmazonKindleStore(AmazonDataset):
    r"""AmazonKindleStore.

    Amazon Review dataset.
    """

    def __init__(self, root_dir=None):
        r"""Init AmazonKindleStore Class."""
        super().__init__(
            dataset_name="amazon-kindle-store",
            root_dir=root_dir,
            url=AMAZON_Kindle_Store_URL,
        )


class AmazonSportsAndOutdoors(AmazonDataset):
    r"""AmazonSportsAndOutdoors.

    Amazon Review dataset.
    """

    def __init__(self, root_dir=None):
        r"""Init AmazonSportsAndOutdoors Class."""
        super().__init__(
            dataset_name="amazon-sports-and-outdoors",
            root_dir=root_dir,
            url=AMAZON_Sports_and_Outdoors_URL,
        )
