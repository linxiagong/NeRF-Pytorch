from .blender import BlenderDataset


def get_dataset(dataset_params: dict):
    if dataset_params["dataset_type"] == "blender":
        return BlenderDataset(split='train',
                              **dataset_params), BlenderDataset(split='val',
                                                                **dataset_params), BlenderDataset(split='test',
                                                                                                  **dataset_params)
    else:
        raise NotImplementedError()
