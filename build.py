from models.l2x_cifar import L2XCifar, Explainer, QNet
import models.l2x_stl10
import models.l2x_20ng
import models.l2x_imdb


def l2x_imdb(args, embedding_matrix=None):
    print("building imdb l2x model")
    explainer = models.l2x_imdb.Explainer(embedding_matrix=embedding_matrix)
    q_net = models.l2x_imdb.QNet(embedding_matrix=embedding_matrix)
    model = models.l2x_imdb.L2XIMDB(explainer=explainer, q_net=q_net)

    return model

def l2x_20ng_rr(args, embedding_matrix):
    print("building l2x model subset baseline 20ng")
    explainer = models.l2x_20ng.Explainer(embedding_matrix=embedding_matrix, rr=True, subset_size=args.subset_size, diffk=False)
    q_net = models.l2x_20ng.QNet(embedding_matrix=embedding_matrix)
    model = models.l2x_20ng.L2X20NG(explainer=explainer, q_net=q_net, subset_size=args.subset_size, diffk=False)

    return model

def l2x_20ng(args, embedding_matrix):
    print("building l2x model 20ng")
    explainer = models.l2x_20ng.Explainer(embedding_matrix=embedding_matrix, subset_size=args.subset_size, diffk=args.diffk, correct=args.correct)
    q_net = models.l2x_20ng.QNet(embedding_matrix=embedding_matrix)
    model = models.l2x_20ng.L2X20NG(explainer=explainer, q_net=q_net, subset_size=args.subset_size, diffk=args.diffk)

    return model

def l2x_stl10_subop(args):
    print("building subop baseline l2x model stl10")
    explainer = models.l2x_stl10.Explainer(subop=True, diffk=False)
    q_net = models.l2x_stl10.QNet()
    model = models.l2x_stl10.L2XSTL10(explainer=explainer, q_net=q_net, subop=True, subset_size=args.subset_size, diffk=False)

    return model

def l2x_stl10(args):
    print("building l2x model stl10")
    explainer = models.l2x_stl10.Explainer(subset_size=args.subset_size, diffk=args.diffk)
    q_net = models.l2x_stl10.QNet()
    model = models.l2x_stl10.L2XSTL10(explainer=explainer, q_net=q_net, subset_size=args.subset_size, diffk=args.diffk)

    return model

def l2x_cifar_subop(args):
    print("building subop baseline cifar l2x model")
    explainer = Explainer(subop=True)
    q_net = QNet()
    model = L2XCifar(explainer=explainer, q_net=q_net, subop=True)

    return model

def l2x_cifar(args):
    print("building l2x cifar model")
    explainer = Explainer()
    q_net = QNet()
    model = L2XCifar(explainer=explainer, q_net=q_net)

    return model

