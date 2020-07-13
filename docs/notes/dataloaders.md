# DataLoaders

Beta-recsys provides reusable components (i.e. DataLoaders), which further process the generated train/valid/test data sets by leveraging the BaseDataset component. In particular, given a specific task to address and the implementation of the corresponding model, DataLoaders convert these train/valid/test datasets into usable data structures (e.g. tensors with < 𝑢𝑠𝑒𝑟,𝑖𝑡𝑒𝑚, 𝑟𝑎𝑡𝑖𝑛𝑔 > or < 𝑢𝑠𝑒𝑟, 𝑝𝑜𝑠𝑖𝑡𝑖𝑣𝑒_𝑖𝑡𝑒𝑚, 𝑛𝑒𝑔𝑎𝑡𝑖𝑣𝑒_𝑖𝑡𝑒𝑚(𝑠) >). Therefore, DataLoaders allow users to load data and data-related features to fulfil distinct requirements. 

In this note, we use the *grocery data* as an example to describe the workflow of DataLoaders.

---
## BaseData

First, BaseData is the base class in the DataLoaders workflow. The BaseData provides various general functions to model the input data (namely the generated train/valid/test data). Currently, the BaseData includes the following functions：

- `_binarize(bin_thld)` : It converts the ground truth (e.g. explicit user ratings) into binarize data with given threshold *bin_thld*.
- `_normalize()` : It applies the min-max normalisation to the ground truth data.
- `_re_index()` : It reindexes the identification of users and items to avoid conflictions or user/item indexing error after applying user/item filtering approaches.
- `_intersect()` : It intersects validation and test datasets with the training dataset to remove users or items that only exist in the training dataset but not in the validation and testing dataset.

Additionally, BaseData also includes common types of data loader instances to enable fast implementation of loading data for a recommendation model. It has the following instances at this stage:

- `instance_bce_loader()` : It structured data into < 𝑢𝑠𝑒𝑟,𝑖𝑡𝑒𝑚, 𝑟𝑎𝑡𝑖𝑛𝑔 > to address pointwise tasks. For example, the binary cross-entropy loss can be applied to learn the pointwise prediction results.
- `instance_bpr_loader()` : It structured data into < 𝑢𝑠𝑒𝑟, 𝑝𝑜𝑠𝑖𝑡𝑖𝑣𝑒_𝑖𝑡𝑒𝑚, 𝑛𝑒𝑔𝑎𝑡𝑖𝑣𝑒_𝑖𝑡𝑒𝑚(𝑠) > for a pair-wise comparison usage (e.g. Bayesian Personalised Ranking). 

---
## TaskData
After initialising the BaseData, for different tasks, a model might require additional feature data or data loaded in different structures. Furthermore, users can add extra functions to request data for customised usages.

For example, the GroceryData class inherits the **BaseData** and **Auxiliary** classes to enable the basic data loading requirement and 
the usage of auxiliary data. It also has another two functions, *sample_triple_time* and *sample_triple*, to sample data under indicated criteria.   

---
## Additional DataLoaders
Beta-recsys also enables users to add customised DataLoaders for various usages. However, instead of updating the BaseData class, users can add extra DataLoader classes in the `data_loaders.py` file to organise the code.

---
## More
For any quesitons, please tell us by creating an issue or contact us by sending an email to recsys.beta@gmail.com. We will try to respond it as soon as possible.
