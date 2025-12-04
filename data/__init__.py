from data.rico import Rico
from data.publaynet import PubLayNet
from data.magazine import Magazine


def get_dataset(name, split, transform=None):
    if name == 'rico':
        return Rico(split, transform)

    elif name == 'publaynet':
        return PubLayNet(split, transform)

    elif name == 'magazine':
        return Magazine(split, transform)
    
    elif name == "crello":
        # 原本的 5 類 Crello（舊的 imgmap 版本）
        from .crello import CrelloDataset
        return CrelloDataset(split, transform, variant="default")

    elif name == "crello_mainpart":
        # 使用你新的 v3 版本（含 label=5），檔名 crello_{split}_v3.pkl
        from .crello import CrelloDataset
        return CrelloDataset(split, transform, variant="v3")

    raise NotImplementedError(name)
