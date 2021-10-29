import torch
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
import numpy as np

class DiffractionDataset(Dataset):
    def __init__(self, data, labels=None, unsupervised=False, categorical='Bravais Lattice'):
        assert categorical=='Bravais Lattice' or categorical=='Space Group', "The key word argument categorical should be Bravais Lattice or Space Group, not {}".format(categorical)
        self.data=data
        self.unsupervised= unsupervised
        if labels==None and not unsupervised:
            data=torch.load(data)
            self.data = data['X']
            self.labels= data['Y']
        elif not self.unsupervised:
            self.labels = labels
        self.categorical=categorical
        self.mapping=torch.load('../Data/mapping.pt')[categorical]

    def __getitem__(self, index):
        if self.unsupervised:
            return self.data[index]
        return self.data[index], self.labels[index]

    def __len__(self):
        return len(self.data)

    def batch_u(self, batch_size):
        idx = torch.randint(0,len(self.data),(batch_size,))
        return self.data[idx]
    
    def plot(self, idx=0):
        assert idx<len(self.data), "Index out of range, please input an index between 0 and {}".format(len(self))
        data=self.data[idx].cpu().numpy().flatten()
        xrange=np.arange(3, data.shape[-1]*0.05+3, 0.05)
        fig = plt.plot(xrange,data)
        plt.xlabel("2\u03B8")
        plt.ylabel("Normalized Intensity")
        str_label=self.mapping[int(self.labels[idx].item())]
        plt.title("{}: {}".format(self.categorical,str_label))
        return fig

    def _unencode_labels(self, labels):
        return np.asarray([self.mapping[int(i.item())] for i in labels])
    def compare(self, prediction):
        prediction=self._unencode_labels(prediction)
        labels=self._unencode_labels(self.labels)
        print("{: >3}{: >20}{: >20}".format("Index", "True Label", "Prediction"))
        for i in range(len(prediction)):
            print("{: >3} {: >20} {: >20}".format(i, labels[i], prediction[i]))