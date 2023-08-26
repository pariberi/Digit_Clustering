import hdbscan
import numpy as np
import torch
import umap.umap_ as umap
from torch.utils import data

from src.utils.file_loader import FileLoader
from src.utils.logger import Logger
from src.utils.transform_dataset import TransformDataset
from .base_model import BaseModel


class SelectivePseudoLabelClustering(BaseModel):
    __trained_aes_path = 'D:\\Digit_Clustering\\src\\assets\\autoencoders'
    __umap_path = 'D:\\Digit_Clustering\\src\\assets\\umaps'
    __hdbscan_path = 'D:\\Digit_Clustering\\src\\assets\\hdbscans'
    __mapping_PATH = 'D:\\Digit_Clustering\\src\\assets\\mappings'

    @Logger.time_logger
    def __init__(self) -> None:
        models = self._load_model()
        self.trained_aes = models[0]
        self.umap_models = models[1]
        self.hdbscan_models = models[2]
        self.mappings = models[3]

    def _load_model(self) -> tuple:
        print('Loading models...')
        trained_aes = FileLoader.load_trained_aes(SelectivePseudoLabelClustering.__trained_aes_path)
        umap_models = []
        hdbscan_models = []
        mappings = []
        umap_model = umap.UMAP(n_neighbors=30, min_dist=0.0, n_components=2, random_state=42)

        for i in range(5):
            umap_models.append(FileLoader.load_pickle(f'{SelectivePseudoLabelClustering.__umap_path}\\umap{i}.obj'))
            # print(f'umap model {i} read.')
            hdbscan_models.append(
                FileLoader.load_pickle(f'{SelectivePseudoLabelClustering.__hdbscan_path}\\hdbscan{i}.obj'))
            # print(f'hdbscan model {i} read.')
            mappings.append(np.load(f'{SelectivePseudoLabelClustering.__mapping_PATH}\\map{i}.npy'))
            # print(f'mapping {i} read.')
        return trained_aes, umap_models, hdbscan_models, mappings

    def __build_latent_space(self, X: torch.Tensor) -> list:
        vecs = []
        for i in range(5):
            determin_dl = data.DataLoader(X, batch_size=1, pin_memory=False)
            for j, (xb, yb, idx) in enumerate(determin_dl):
                latent = self.trained_aes[i].enc(xb)
                latent = latent.view(latent.shape[0], -1).detach().cpu().numpy()
                vecs.append(latent)
        return vecs

    def __get_umaps(self, vectors: list) -> list:
        umaps = []
        for i in range(5):
            umap = self.umap_models[i].transform(vectors[i])
            umaps.append(umap)
        return umaps

    def __get_hdbscan_labels(self, umaps: list) -> list:
        labels = []
        for i in range(5):
            label, strengths = hdbscan.approximate_predict(self.hdbscan_models[i], umaps[i])
            labels.append(label[0])

        return labels

    def __map_labels(self, hdbscan_labels: list) -> int:
        final_labels = []
        for i in range(5):
            final_labels.append(self.mappings[i][hdbscan_labels[i]])
        print(f'predicted labels: {final_labels}')
        return max(final_labels, key=final_labels.count)

    @Logger.time_logger
    def predict(self, test_sample: TransformDataset) -> int:
        print("\nStart of prediction...")
        test_vectors = self.__build_latent_space(test_sample)
        test_umap = self.__get_umaps(test_vectors)
        test_hdbscan = self.__get_hdbscan_labels(test_umap)
        # print(test_hdbscan)
        test_label = self.__map_labels(test_hdbscan)

        return test_label
