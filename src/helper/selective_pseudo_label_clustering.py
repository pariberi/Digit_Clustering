import hdbscan
import torch

from base_model import BaseModel
from src import utils


class SelectivePseudoLabelClustering(BaseModel):
    def __init__(self, aes_path: str, umap_path: str, hdbscan_path: str) -> None:
        models = self._load_model(ae=aes_path, umap=umap_path, hdbscan=hdbscan_path)
        self.trained_aes = models[0]
        self.umap_models = models[1]
        self.hdbscan_models = models[2]

    def _load_model(self, **paths) -> tuple:
        trained_aes = utils.load_trained_aes(paths['ae'])
        umap_models = []
        hdbscan_models = []
        path = paths['umap']
        for i in range(5):
            umap_models.append(utils.load_pickle(f'{path}/umap{i}.npy'))
        path = paths['hdbscan']
        for i in range(5):
            hdbscan_models.append(utils.load_pickle(f'{path}/hdbscan{i}.npy'))
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
