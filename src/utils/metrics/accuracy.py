import torch

def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        if target.dim() > 1 and target.size(1) > 1:
            target = target.argmax(dim=1)

        _, pred = output.topk(maxk, 1, True, True) # topk devuelve (valores, indices)
        pred = pred.t() # cada columna es un ejemplo, la fila i es topi (en nuestro caso
                        # binario la fila de arriba es la prediccion real)
        correct = pred.eq(target.view(1, -1).expand_as(pred)) # view pasa de (batch_size,) a 
                        # (1, batch_size), y luego expand duplica eso k veces
        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res
