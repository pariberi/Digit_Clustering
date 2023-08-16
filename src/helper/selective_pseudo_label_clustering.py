import hdbscan
import torch

from base_model import BaseModel
from src import utils


class SelectivePseudoLabelClustering(BaseModel):
    __trained_aes_path = '....\Autoencoders'
    __umap_path = '.....\Umaps'
    __hdbscan_path = '.....\Hdbscan'

    def __init__(self) -> None:
        models = self._load_model()
        self.trained_aes = models[0]
        self.umap_models = models[1]
        self.hdbscan_models = models[2]

    def _load_model(self, **paths) -> tuple:
        trained_aes = utils.load_trained_aes(SelectivePseudoLabelClustering.__trained_aes_path)
        umap_models = []
        hdbscan_models = []
        for i in range(5):
            umap_models.append(utils.load_pickle(f'{SelectivePseudoLabelClustering.__umap_path}/umap{i}.npy'))
        for i in range(5):
            hdbscan_models.append(utils.load_pickle(f'{SelectivePseudoLabelClustering.__hdbscan_path}/hdbscan{i}.npy'))
        return trained_aes, umap_models, hdbscan_models

    def __build_latent_space(self, X: torch.Tensor) -> list:
        vecs = []
        for i in range(5):
            latent = self.trained_aes[i].enc(X)
            latent = latent.view(latent.shape[0], -1).detach().cpu().numpy()
            vecs.append(latent)
        return vecs

    def __get_umaps(self, vectors: list) -> list:
        umaps = []
        for i in range(5):
            umap = self.umap_models[i].transform(vectors[i].squeeze())
            umaps.append(umap)
        return umaps

    def __get_hdbscan_labels(self, umaps: list) -> list:
        labels = []
        for i in range(5):
            label, strengths = hdbscan.approximate_predict(self.hdbscan_model[i], umaps[i])
            labels.append(label)
            
        return labels

    def predict(self, test_sample: torch.Tensor) -> int:
        test_vectors = self.__build_latent_space(test_sample)
        test_umap = self.__get_umaps(test_vectors)
        test_HDBSCAN = self.__get_hdbscan_labels(test_umap)

        return max(test_HDBSCAN, key=test_HDBSCAN.count)
