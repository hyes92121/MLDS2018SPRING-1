import torch
import torch.nn as nn


class CustomLoss(nn.Module):
    def __init__(self):
        super(CustomLoss, self).__init__()
        self.loss_fn = nn.CrossEntropyLoss()
        self.loss = 0
        self.avg_loss = None

    def forward(self, x, y, lengths):
        # first dim of x and y is the same (equals to batch size)
        batch_size = len(x)

        predict_cat = None
        groundT_cat = None

        flag = True

        for batch in range(batch_size):
            predict      = x[batch]
            ground_truth = y[batch]
            seq_len = lengths[batch] -1
            # seq_len includes <SOS> and <EOS>,
            # but our predictions and modified ground truths do not have <SOS>

            # 
            predict = predict[:seq_len]
            ground_truth = ground_truth[:seq_len]

            if flag:
                predict_cat = predict
                groundT_cat = ground_truth
                flag = False

            else:
                predict_cat = torch.cat((predict_cat, predict), dim=0)
                groundT_cat = torch.cat((groundT_cat, ground_truth), dim=0)
                # concat so that we can calculate loss even if there exists
                # different sequence lengths in a batch

        try:
            assert len(predict_cat) == len(groundT_cat)

        except AssertionError as error:
            print('prediction length is not same as ground truth length. error location: Custom Loss')
            print('prediction length: {}, ground truth length: {}'.format(len(predict_cat), len(groundT_cat)))


        self.loss = self.loss_fn(predict_cat, groundT_cat)

        self.avg_loss = self.loss/batch_size

        return self.loss


if __name__ == '__main__':
    from torch.autograd import Variable

    x = torch.rand(32, 16, 1799)
    y = torch.LongTensor(32, 16).random_(1799)
    length = list(torch.LongTensor(32).random_(5, 17))
    length.sort(reverse=True)

    x, y = Variable(x, requires_grad=True), Variable(y)

    x = x+2

    print(x)
    print(y)
    print(length)

    ll = CustomLoss()

    loss = ll(x, y, length)
    loss.backward()

    print(loss)