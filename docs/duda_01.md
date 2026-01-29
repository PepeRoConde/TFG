si cuando haciendo forward de una red profunda usas una linearidad que empuja a 0 no hay informacion por eso silu y tal o incluso en el caso de las snn gradiente surrogado. 

clipin en ista ignora eso? implementar!

    class FeedForward(nn.Module):
        def __init__(self, dim, hidden_dim, dropout=0., step_size=0.1):
            super().__init__()
            self.weight = nn.Parameter(torch.Tensor(dim, dim))
            with torch.no_grad():
                init.kaiming_uniform_(self.weight)
            self.step_size = step_size
            self.lambd = 0.1
    
        def forward(self, x):
            # compute D^T * D * x
            x1 = F.linear(x, self.weight, bias=None)
            grad_1 = F.linear(x1, self.weight.t(), bias=None)
            # compute D^T * x
            grad_2 = F.linear(x, self.weight.t(), bias=None)
            # compute negative gradient update: step_size * (D^T * x - D^T * D * x)
ojo con lo que viene ahora

esta haciendo

    Relu(x) = max(x + self.lambd, 0)

porque self.lambd es una constante. 
eso tiene sentido porque es la verdadera (unica) restriccion de esparsidad (aunque hable mucho de la norma cero). y el sumarle la constante es para forzar que este a cero y no tambalee cerca. el gradiente tendra que ser suficientemente largo para sacar a alguien del cero. 

idea: substituir por algo smoth para que la derivada sea continua 

e.g. en vez de grad_2 - grad_1 usar (grad_2 - grad_1 )2

ademas: https://www.youtube.com/watch?v=bhqNSjJ_A20 (minuto 4) yi ma no normaliza?

---

por otro lado se podria usar el relu en forwarda y el gradiente que sea una funcion continua (de 0 a lambd el gradiente crece a uno y a partir de ahi es uno)

---
finalmente

            grad_update = self.step_size * (grad_2 - grad_1) - self.step_size * self.lambd
    
            output = F.relu(x + grad_update)
            return output j
