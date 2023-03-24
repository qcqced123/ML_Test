# Helper Function =>
def check_cudnn() -> tuple:
    return (torch.backends.cudnn.version(), torch.backends.cudnn.is_available(), torch.backends.cudnn.enabled)


def class2dict(cfg: CFG) -> dict:
    return dict((name, getattr(cfg, name)) for name in dir(cfg) if not name.startswith('__'))


def all_type_seed(cfg: CFG) -> None:
    # python seed
    os.environ['PYTHONHASHSEED'] = str(cfg.seed)  # python Seed
    random.seed(cfg.seed)  # random module Seed
    np.random.seed(cfg.seed)  # numpy module Seed
    # torch.cuda seed
    torch.manual_seed(cfg.seed)  # Pytorch CPU Random Seed Maker
    torch.cuda.manual_seed(cfg.seed)  # Pytorch GPU Random Seed Maker
    torch.cuda.manual_seed_all(cfg.seed)  # Pytorch Multi Core GPU Random Seed Maker
    # torch.cudnn seed
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True


def seed_worker(worker_id) -> None:
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def get_logger(filename: str) -> _Logger:
    logger = getLogger(__name__)
    logger.setLevel(INFO)
    handler1 = StreamHandler()
    handler1.setFormatter(Formatter("%(message)s"))
    handler2 = FileHandler(filename=f"{filename}.log")
    handler2.setFormatter(Formatter("%(message)s"))
    logger.addHandler(handler1)
    logger.addHandler(handler2)
    return logger


if not os.path.exists(CFG.output_dir):
    os.makedirs(CFG.output_dir)

logger = get_logger(filename=CFG.output_dir + 'train')

all_type_seed(CFG)
g = torch.Generator()
g.manual_seed(0)
check_cudnn()