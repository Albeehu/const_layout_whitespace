from data.rico import Rico
from data.publaynet import PubLayNet
from data.magazine import Magazine
from .crello import CrelloDataset

def get_dataset(name, split, transform=None):
    if name == 'rico':
        return Rico(split, transform)

    elif name == 'publaynet':
        return PubLayNet(split, transform)

    elif name == 'magazine':
        return Magazine(split, transform)
    
    elif name == "crello":
        # 原本的 5 類 Crello（舊的 imgmap 版本）
        return CrelloDataset(split, transform, variant="default")
    
    elif name == "crello_mainpart_face":
    # 1208新增版本：會讀 crello_*_face.pkl，把 face box merge 進來
        return CrelloDataset(
            split=split,
            transform=transform,
            variant="default",
            use_face=True,
        )

    raise NotImplementedError(name)


