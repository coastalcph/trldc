import torch


'''[2022-Mar-10] https://github.com/allenai/specter/blob/master/specter/model.py#L30'''
class TripletLoss(torch.nn.Module):
    def __init__(self, margin=1.0, distance="l2-norm", reduction="mean"):
        super(TripletLoss, self).__init__()
        self.margin = margin
        self.distance = distance # options include l2-norm, cosine, dot
        self.reduction = reduction # options include mean, sum
        assert distance == "l2-norm" and reduction == "mean"

    def forward(self, query, positive, negative):
        distance_positive = torch.nn.functional.pairwise_distance(query, positive)
        distance_negative = torch.nn.functional.pairwise_distance(query, negative)
        losses = torch.nn.functional.relu(distance_positive - distance_negative + self.margin)
        return losses.mean()